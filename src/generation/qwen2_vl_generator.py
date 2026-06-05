"""
Qwen2.5-VL-7B Model Wrapper.

Implements BaseVLM for Qwen2.5-VL-7B-Instruct with optional quantization.
Uses the Qwen2-VL chat template for structured medical VQA with
strong grounding support.

Supports:
  - Base model inference
  - 4-bit quantization via bitsandbytes (fits in A100 40GB with room)
  - bf16/fp16 full precision
  - Structured grounding prompts with evidence injection

Architecture notes:
  - Qwen2.5-VL uses a different conversation format than LLaVA:
      <|im_start|>system\n{system_prompt}<|im_end|>
      <|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{text}<|im_end|>
      <|im_start|>assistant\n
  - The processor handles image token injection automatically
  - Qwen2.5-VL has much stronger instruction following than LLaVA-1.5
"""

import time
from typing import Optional, Dict, Any, List

import torch
from PIL import Image

from src.generation.base_generator import BaseVLM, VLMOutput
from src.utils.device import get_device, get_vram_usage_gb
from src.utils.logging_utils import setup_logger

logger = setup_logger("models.qwen2_vl")


# ------------------------------------------------------------------ #
#  System prompt for grounded medical VQA                              #
# ------------------------------------------------------------------ #

MEDICAL_SYSTEM_PROMPT = (
    "You are an expert radiologist analyzing chest X-ray images. "
    "You will be given retrieved clinical evidence from similar cases "
    "in a medical knowledge base.\n\n"
    "STRICT RULES:\n"
    "1. Base your answer PRIMARILY on the retrieved evidence and the image.\n"
    "2. Do NOT hallucinate findings that are not supported by evidence.\n"
    "3. If the evidence clearly states a finding is absent "
    "(e.g., 'no pleural effusion'), your answer must reflect that.\n"
    "4. If the evidence conflicts with what you observe in the image, "
    "explicitly state the discrepancy.\n"
    "5. If evidence is insufficient, say so explicitly.\n"
    "6. Always justify your answer by referencing specific evidence."
)


class Qwen2VLModel(BaseVLM):
    """
    Qwen2.5-VL-7B-Instruct wrapper for grounded medical VQA.

    Key advantages over LLaVA-1.5-7B:
      - Stronger instruction following (critical for grounding)
      - Better negation understanding
      - Native multi-turn chat template
      - Dynamic resolution support

    Usage:
        model = Qwen2VLModel()
        model.load(config)
        output = model.generate(image, "Is there pneumonia?", context="...")
    """

    def __init__(self):
        self._model = None
        self._processor = None
        self._config = None
        self._device = None
        self._model_name = "qwen2.5-vl-7b"
        self._loaded = False

    # ------------------------------------------------------------------ #
    #  BaseVLM interface                                                   #
    # ------------------------------------------------------------------ #

    def load(self, config: dict) -> None:
        """Load Qwen2.5-VL-7B with optional quantization."""
        from transformers import (
            Qwen2VLForConditionalGeneration,
            AutoProcessor,
        )

        self._config = config
        model_cfg = config["model"]
        model_id = model_cfg["model_id"]

        logger.info(f"Loading Qwen2.5-VL model: {model_id}")

        # Build quantization config
        quant_config = None
        quant_enabled = model_cfg.get("quantization", {}).get("enabled", False)

        if quant_enabled and not torch.cuda.is_available():
            logger.warning(
                "  Quantization requires CUDA — falling back to CPU mode"
            )
            quant_enabled = False

        if quant_enabled:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(
                    torch,
                    model_cfg["quantization"].get(
                        "bnb_4bit_compute_dtype", "bfloat16"
                    ),
                ),
                bnb_4bit_quant_type=model_cfg["quantization"].get(
                    "bnb_4bit_quant_type", "nf4"
                ),
                bnb_4bit_use_double_quant=model_cfg["quantization"].get(
                    "bnb_4bit_use_double_quant", True
                ),
            )
            logger.info("  4-bit quantization config created")

        # Load processor
        self._processor = AutoProcessor.from_pretrained(model_id)
        logger.info("  Processor loaded")

        # Load model
        load_kwargs = {
            "pretrained_model_name_or_path": model_id,
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
        }

        if quant_config:
            load_kwargs["quantization_config"] = quant_config
            load_kwargs["device_map"] = "auto"
        else:
            self._device = get_device(model_cfg.get("device", "auto"))
            if str(self._device) == "cpu":
                load_kwargs["torch_dtype"] = torch.float32
                logger.info("  Loading in CPU mode (float32)")
            else:
                load_kwargs["device_map"] = "auto"

        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            **load_kwargs
        )

        if "device_map" not in load_kwargs:
            self._model = self._model.to(self._device)

        self._device = self._model.device
        self._loaded = True

        logger.info(f"  Model loaded on device: {self._device}")
        logger.info(f"  Model dtype: {self._model.dtype}")

        mem = self.get_memory_footprint()
        logger.info(f"  VRAM allocated: {mem['allocated_gb']} GB")

    def generate(
        self,
        image: Image.Image,
        question: str,
        context: Optional[str] = None,
        max_new_tokens: int = 512,
    ) -> VLMOutput:
        """Generate answer from image + question using Qwen2.5-VL."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        gen_cfg = self._config["model"].get("generation", {})

        # Build the chat messages
        messages = self._build_messages(question, context)

        # Process inputs using the chat template
        prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self._processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
            padding=True,
        )

        # Move to device
        device = self._model.device
        inputs = {
            k: v.to(device) if hasattr(v, 'to') else v
            for k, v in inputs.items()
        }

        input_token_count = inputs["input_ids"].shape[-1]

        # Generate
        try:
            start_time = time.time()
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=gen_cfg.get("temperature", 0.1),
                    top_p=gen_cfg.get("top_p", 0.9),
                    do_sample=gen_cfg.get("do_sample", False),
                    repetition_penalty=gen_cfg.get(
                        "repetition_penalty", 1.1
                    ),
                )
            generation_time = time.time() - start_time
        except RuntimeError as e:
            logger.error(f"Generation failed: {e}")
            return VLMOutput(
                answer=f"[Generation error: {e}]",
                raw_output="",
                generation_time_sec=0.0,
                input_token_count=input_token_count,
                output_token_count=0,
                metadata={"error": str(e), "model": self._model_name},
            )

        # Decode only new tokens
        generated_ids = output_ids[0, input_token_count:]
        raw_output = self._processor.decode(
            generated_ids, skip_special_tokens=True
        )
        answer = raw_output.strip()

        return VLMOutput(
            answer=answer,
            raw_output=raw_output,
            generation_time_sec=round(generation_time, 2),
            input_token_count=input_token_count,
            output_token_count=len(generated_ids),
            metadata={
                "model": self._model_name,
                "prompt": prompt,
            },
        )

    def caption(self, image: Image.Image) -> str:
        """Generate a clinical caption for a medical image."""
        output = self.generate(
            image=image,
            question=(
                "Describe all clinically significant findings visible "
                "in this chest X-ray image."
            ),
            max_new_tokens=256,
        )
        return output.answer

    def get_memory_footprint(self) -> Dict[str, float]:
        """Report VRAM usage."""
        vram = get_vram_usage_gb()
        return {
            "allocated_gb": vram["allocated"],
            "reserved_gb": vram["reserved"],
            "total_gb": vram["total"],
        }

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------ #
    #  Internal: build chat messages                                       #
    # ------------------------------------------------------------------ #

    def _build_messages(
        self,
        question: str,
        context: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build Qwen2.5-VL chat messages.

        Creates a structured conversation with:
          1. System message: grounding rules
          2. User message: image + evidence + question

        This is a SINGLE prompt construction — no double-wrapping.
        The system prompt establishes grounding rules, and the user
        message contains the evidence and question cleanly separated.

        Args:
            question: The clinical question.
            context:  Optional evidence summary from the aggregator.

        Returns:
            List of message dicts for the chat template.
        """
        messages = [
            {"role": "system", "content": MEDICAL_SYSTEM_PROMPT},
        ]

        # Build user message content (multimodal: image + text)
        user_content = []

        # Image (Qwen2.5-VL uses a content list with type markers)
        user_content.append({"type": "image", "image": "placeholder"})

        # Evidence block (if available)
        if context:
            text_parts = [
                "RETRIEVED EVIDENCE FROM SIMILAR CASES:\n",
                context,
                "\n\nQUESTION: " + question,
                "\n\nProvide your answer. Start with a direct YES or NO "
                "if the question is a yes/no type. Then explain your "
                "reasoning, citing specific evidence.",
            ]
        else:
            text_parts = [
                "QUESTION: " + question,
                "\n\nProvide a detailed clinical answer based on the image.",
            ]

        user_content.append({
            "type": "text",
            "text": "".join(text_parts),
        })

        messages.append({"role": "user", "content": user_content})

        return messages

    # ------------------------------------------------------------------ #
    #  Adapter support (future QLoRA fine-tuning)                          #
    # ------------------------------------------------------------------ #

    def _load_adapter(self, adapter_path: str) -> None:
        """Load a LoRA adapter on top of the base model."""
        from peft import PeftModel

        logger.info(f"  Loading LoRA adapter from: {adapter_path}")
        self._model = PeftModel.from_pretrained(
            self._model, adapter_path
        )
        logger.info("  LoRA adapter loaded successfully")

    @property
    def model(self):
        """Direct access to the underlying model."""
        return self._model

    @property
    def processor(self):
        """Direct access to the processor."""
        return self._processor
