"""
LLaVA-1.5-7B Model Wrapper.

Implements BaseVLM for LLaVA-1.5-7B with 4-bit quantization (QLoRA-ready).
Handles model loading, inference, captioning, and LoRA adapter management.

Supports:
  - Base model inference (no fine-tuning)
  - QLoRA fine-tuned inference (load adapter on top of base)
  - 4-bit quantization via bitsandbytes
"""

import time
from typing import Optional, Dict, Any

import torch
from PIL import Image

from src.generation.base_generator import BaseVLM, VLMOutput
from src.utils.device import get_device, get_vram_usage_gb
from src.utils.logging_utils import setup_logger

logger = setup_logger("models.llava")


class LLaVAModel(BaseVLM):
    """
    LLaVA-1.5-7B wrapper with 4-bit quantization support.

    Usage:
        model = LLaVAModel()
        model.load(config)
        output = model.generate(image, "What does this X-ray show?")
    """

    def __init__(self):
        self._model = None
        self._processor = None
        self._config = None
        self._device = None
        self._model_name = "llava-1.5-7b"
        self._loaded = False

    # ------------------------------------------------------------------ #
    #  BaseVLM interface                                                   #
    # ------------------------------------------------------------------ #

    def load(self, config: dict) -> None:
        """Load LLaVA-1.5-7B with optional 4-bit quantization and LoRA adapter."""
        from transformers import (
            LlavaForConditionalGeneration,
            BitsAndBytesConfig,
        )
        # Use explicit LlavaProcessor — AutoProcessor fails on transformers 5.x
        try:
            from transformers import LlavaProcessor
        except ImportError:
            from transformers import AutoProcessor as LlavaProcessor

        self._config = config
        model_cfg = config["model"]
        model_id = model_cfg["model_id"]

        logger.info(f"Loading LLaVA model: {model_id}")
        logger.info(f"  Quantization: {model_cfg['quantization']}")

        # Build quantization config
        quant_config = None
        quant_enabled = model_cfg["quantization"]["enabled"]

        # Auto-disable quantization if no CUDA GPU available
        if quant_enabled and not torch.cuda.is_available():
            logger.warning("  4-bit quantization requires CUDA GPU — not available")
            logger.warning("  Falling back to CPU float16 mode (slow but functional)")
            quant_enabled = False

        if quant_enabled:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(
                    torch, model_cfg["quantization"]["bnb_4bit_compute_dtype"]
                ),
                bnb_4bit_quant_type=model_cfg["quantization"]["bnb_4bit_quant_type"],
                bnb_4bit_use_double_quant=model_cfg["quantization"][
                    "bnb_4bit_use_double_quant"
                ],
            )
            logger.info("  4-bit quantization config created")

        # Load processor (tokenizer + image processor)
        self._processor = LlavaProcessor.from_pretrained(model_id)
        logger.info("  Processor loaded")

        # Load model
        load_kwargs = {
            "pretrained_model_name_or_path": model_id,
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
        }

        if quant_config:
            load_kwargs["quantization_config"] = quant_config
            load_kwargs["device_map"] = "auto"
        else:
            self._device = get_device(model_cfg.get("device", "auto"))
            if str(self._device) == "cpu":
                # CPU mode: float16 to fit in 16GB RAM (float32 needs ~28GB)
                load_kwargs["torch_dtype"] = torch.float16
                logger.info("  Loading in CPU mode (float16, no quantization)")
            else:
                load_kwargs["device_map"] = "auto"

        self._model = LlavaForConditionalGeneration.from_pretrained(**load_kwargs)

        # Move to device if needed (CPU mode without device_map)
        if "device_map" not in load_kwargs:
            self._model = self._model.to(self._device)

        self._device = self._model.device

        logger.info(f"  Model loaded on device: {self._device}")
        logger.info(f"  Model dtype: {self._model.dtype}")

        # Load LoRA adapter if specified
        adapter_path = model_cfg.get("adapter_path")
        if adapter_path:
            self._load_adapter(adapter_path)

        self._loaded = True
        mem = self.get_memory_footprint()
        logger.info(f"  VRAM allocated: {mem['allocated_gb']} GB")

    def generate(
        self,
        image: Image.Image,
        question: str,
        context: Optional[str] = None,
        max_new_tokens: int = 512,
    ) -> VLMOutput:
        """Generate answer from image + question using LLaVA."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        gen_cfg = self._config["model"]["generation"]

        # Build the conversation prompt
        prompt = self._build_prompt(question, context)

        # Process inputs
        inputs = self._processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )

        # ----------------------------------------------------------
        # Move tensors to the correct device and dtype.
        #
        # CRITICAL: With 4-bit quantized models (bitsandbytes),
        # next(model.parameters()).dtype may return a quantized dtype
        # (e.g. uint8). Casting pixel_values to that dtype corrupts
        # the vision tower, causing bmm shape mismatches like:
        #   "Expected [16, 2916] but got [16, 1]"
        #
        # Strategy:
        #   - pixel_values → float16 (vision tower compute dtype)
        #   - integer tensors (input_ids, attention_mask) → device only
        # ----------------------------------------------------------
        device = self._model.device

        moved_inputs = {}
        for k, v in inputs.items():
            if k == "pixel_values":
                # Always cast pixel_values to float16 — the vision tower
                # expects this regardless of the LM's quantization dtype
                moved_inputs[k] = v.to(device=device, dtype=torch.float16)
            elif v.is_floating_point():
                moved_inputs[k] = v.to(device=device, dtype=torch.float16)
            else:
                moved_inputs[k] = v.to(device=device)
        inputs = moved_inputs

        # ----------------------------------------------------------
        # Shape validation — catch mismatches before they crash inside
        # the model's attention layers with opaque bmm errors.
        # ----------------------------------------------------------
        if "input_ids" in inputs:
            ids_shape = inputs["input_ids"].shape
            if ids_shape[0] != 1:
                logger.warning(
                    f"Unexpected batch size in input_ids: {ids_shape}. "
                    f"Expected batch_size=1."
                )

        if "pixel_values" in inputs:
            pv_shape = inputs["pixel_values"].shape
            logger.info(f"  pixel_values shape: {pv_shape}, dtype: {inputs['pixel_values'].dtype}")
            if pv_shape[0] != 1:
                logger.warning(
                    f"Unexpected batch size in pixel_values: {pv_shape}. "
                    f"Expected batch_size=1."
                )

        if "attention_mask" in inputs and "input_ids" in inputs:
            am_shape = inputs["attention_mask"].shape
            ids_shape = inputs["input_ids"].shape
            if am_shape != ids_shape:
                logger.warning(
                    f"attention_mask shape {am_shape} does not match "
                    f"input_ids shape {ids_shape}."
                )

        input_token_count = inputs["input_ids"].shape[-1]

        # ----------------------------------------------------------
        # Generate — wrapped in try/except for graceful failure
        # ----------------------------------------------------------
        try:
            start_time = time.time()
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=gen_cfg.get("temperature", 0.1),
                    top_p=gen_cfg.get("top_p", 0.9),
                    do_sample=gen_cfg.get("do_sample", False),
                    repetition_penalty=gen_cfg.get("repetition_penalty", 1.1),
                )
            generation_time = time.time() - start_time
        except RuntimeError as e:
            error_msg = str(e)
            # Log detailed diagnostic info for tensor shape errors
            logger.error(f"Model generation failed: {error_msg}")
            logger.error(f"  input_ids shape: {inputs.get('input_ids', 'N/A')}")
            if "pixel_values" in inputs:
                logger.error(f"  pixel_values shape: {inputs['pixel_values'].shape}")
                logger.error(f"  pixel_values dtype: {inputs['pixel_values'].dtype}")
            if "attention_mask" in inputs:
                logger.error(f"  attention_mask shape: {inputs['attention_mask'].shape}")

            return VLMOutput(
                answer=f"[Generation error: {error_msg}]",
                raw_output="",
                generation_time_sec=0.0,
                input_token_count=input_token_count,
                output_token_count=0,
                metadata={"error": error_msg, "model": self._model_name},
            )

        # Decode — only the NEW tokens (skip the input)
        generated_ids = output_ids[0, input_token_count:]
        raw_output = self._processor.decode(generated_ids, skip_special_tokens=True)
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
            question="Describe all clinically significant findings visible in this medical image.",
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
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _build_prompt(self, question: str, context: Optional[str] = None) -> str:
        """
        Build the LLaVA conversation prompt.

        LLaVA-1.5 uses the Vicuna conversation format:
          USER: <image>\n{instruction}\nASSISTANT:

        IMPORTANT: The context is the structured evidence summary from
        the EvidenceAggregator — NOT a full prompt template. We inject
        it directly here with grounding instructions.
        """
        if context:
            # Single clean prompt — no double wrapping
            prompt = (
                f"USER: <image>\n"
                f"You are an expert radiologist. Answer ONLY based on "
                f"the evidence and the image.\n\n"
                f"{context}\n\n"
                f"QUESTION: {question}\n"
                f"If this is a yes/no question, start with YES or NO. "
                f"If the evidence says a finding is absent, reflect that. "
                f"Cite specific evidence.\n"
                f"ASSISTANT:"
            )
        else:
            prompt = f"USER: <image>\n{question}\nASSISTANT:"

        return prompt

    def _load_adapter(self, adapter_path: str) -> None:
        """Load a LoRA adapter on top of the base model."""
        from peft import PeftModel

        logger.info(f"  Loading LoRA adapter from: {adapter_path}")
        self._model = PeftModel.from_pretrained(self._model, adapter_path)
        logger.info("  LoRA adapter loaded successfully")

    # ------------------------------------------------------------------ #
    #  Training support                                                    #
    # ------------------------------------------------------------------ #

    def prepare_for_training(self) -> None:
        """
        Prepare model for QLoRA training.

        Strategy (matches Kaggle training script):
          1. prepare_model_for_kbit_training
          2. Apply LoRA to ALL q/k/v/o_proj modules
          3. Freeze LoRA params in vision_tower and multi_modal_projector
             so only language model LoRA trains
        """
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        lora_cfg = self._config["model"]["lora"]

        # Step 1: Prepare for k-bit training
        self._model = prepare_model_for_kbit_training(self._model)

        # Step 2: Attach LoRA to ALL matching modules
        peft_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"],
            target_modules=lora_cfg["target_modules"],
            bias=lora_cfg["bias"],
            task_type=lora_cfg["task_type"],
        )
        self._model = get_peft_model(self._model, peft_config)

        # Step 3: Freeze non-language-model LoRA params
        frozen = 0
        for name, param in self._model.named_parameters():
            if param.requires_grad and (
                "vision_tower" in name or "multi_modal_projector" in name
            ):
                param.requires_grad = False
                frozen += 1

        trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self._model.parameters())
        logger.info(f"  LoRA attached. Frozen {frozen} non-LM adapter params.")
        logger.info(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")

    def save_adapter(self, output_dir: str) -> None:
        """Save only the LoRA adapter weights (not the full model)."""
        from pathlib import Path

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(output_dir)
        self._processor.save_pretrained(output_dir)
        logger.info(f"  Adapter saved to: {output_dir}")

    @property
    def model(self):
        """Direct access to the underlying model (for training scripts)."""
        return self._model

    @property
    def processor(self):
        """Direct access to the processor (for training scripts)."""
        return self._processor
