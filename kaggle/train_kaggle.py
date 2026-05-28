"""
QLoRA Fine-Tuning: LLaVA-1.5-7B on OpenI Chest X-rays (Kaggle T4).

FINAL PRODUCTION VERSION — 2026-05-15

Paste this ENTIRE script into ONE Kaggle notebook cell and run.
Kernel must have: GPU T4 enabled, Internet ON.

After training, download final_adapter/ from the Output tab.
"""

# ============================================================
# 1. Environment (MUST be before any other imports)
# ============================================================
import os, subprocess, sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ============================================================
# 2. Install dependencies
# ============================================================
def _pip(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

_pip("transformers>=4.40.0")
_pip("accelerate>=0.27.0")
_pip("bitsandbytes>=0.43.0")
_pip("peft>=0.10.0")
_pip("Pillow")

# ============================================================
# 3. Imports
# ============================================================
import csv, re, random, logging, gc, time
from pathlib import Path
from collections import defaultdict

import torch
from PIL import Image
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("kaggle_train")

# ============================================================
# 4. Configuration
# ============================================================
CONFIG = {
    "model_id": "llava-hf/llava-1.5-7b-hf",
    "kaggle_dataset_root": "/kaggle/input/datasets/raddar/chest-xrays-indiana-university",
    "output_dir": "/kaggle/working/llava-medical-vqa",
    "max_samples": None,
    "train_ratio": 0.85,
    "val_ratio": 0.10,
    "seed": 42,
    "num_epochs": 3,
    "lr": 2e-4,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
}

VQA_TEMPLATES = [
    "What are the findings in this chest X-ray?",
    "Describe the abnormalities visible in this radiograph.",
    "What does this chest X-ray show?",
    "Provide a clinical interpretation of this chest radiograph.",
    "What is the radiological impression of this image?",
    "Are there any significant findings in this X-ray?",
    "Summarize the key observations in this chest X-ray.",
    "What clinical conditions can be identified from this radiograph?",
]


# ============================================================
# 5. Path Discovery
# ============================================================
def discover_paths(dataset_root):
    root = Path(dataset_root)
    logger.info(f"Dataset root: {root} (exists: {root.exists()})")

    kaggle_input = Path("/kaggle/input")
    if not root.exists() and kaggle_input.exists():
        logger.warning(f"Root missing. Scanning {kaggle_input} ...")
        for item in sorted(kaggle_input.iterdir()):
            logger.info(f"  [{'DIR ' if item.is_dir() else 'FILE'}] {item.name}")
        root = kaggle_input
    elif root.exists():
        for item in sorted(root.iterdir()):
            logger.info(f"  [{'DIR ' if item.is_dir() else 'FILE'}] {item.name}")

    reports_csv = next(root.rglob("indiana_reports.csv"), None)
    if reports_csv is None:
        raise FileNotFoundError(f"indiana_reports.csv not found under {root}")
    logger.info(f"  Reports: {reports_csv}")

    projections_csv = reports_csv.parent / "indiana_projections.csv"
    if not projections_csv.exists():
        projections_csv = next(root.rglob("indiana_projections.csv"), projections_csv)
    logger.info(f"  Projections: {projections_csv} ({projections_csv.exists()})")

    images_dir = next((d for d in root.rglob("images_normalized") if d.is_dir()), None)
    if images_dir is None:
        first_png = next(root.rglob("*.png"), None)
        images_dir = first_png.parent if first_png else None
    if images_dir is None:
        raise FileNotFoundError(f"No images found under {root}")
    logger.info(f"  Images: {images_dir} ({len(list(images_dir.glob('*.png')))} PNGs)")

    return {"reports_csv": reports_csv, "projections_csv": projections_csv, "images_dir": images_dir}


# ============================================================
# 6. Dataset Loading
# ============================================================
def _load_projections(csv_path):
    uid_map = defaultdict(list)
    if not csv_path.exists():
        return uid_map
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uid = row.get("uid", "").strip()
            fn = row.get("filename", "").strip()
            proj = row.get("projection", "").strip()
            if uid and fn:
                uid_map[uid].append({"filename": fn, "projection": proj})
    logger.info(f"Projections: {len(uid_map)} UIDs")
    return uid_map


def _resolve_image(uid, uid_map, images_dir):
    if uid in uid_map:
        entries = uid_map[uid]
        frontal = [e for e in entries if e["projection"].lower() == "frontal"]
        for e in (frontal or entries):
            p = images_dir / e["filename"]
            if p.exists():
                return str(p)
    hits = sorted(images_dir.glob(f"{uid}_*"))
    return str(hits[0]) if hits else None


def _clean(text):
    if not text or not text.strip():
        return None
    text = re.sub(r"\bXXXX\b", "", text.strip())
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) >= 5 else None


def load_dataset(paths):
    uid_map = _load_projections(paths["projections_csv"])
    rng = random.Random(CONFIG["seed"])
    samples = []
    with open(paths["reports_csv"], "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if CONFIG["max_samples"] and len(samples) >= CONFIG["max_samples"]:
                break
            uid = row.get("uid", "").strip()
            findings = _clean(row.get("findings", ""))
            impression = _clean(row.get("impression", ""))
            if not findings and not impression:
                continue
            img = _resolve_image(uid, uid_map, paths["images_dir"])
            if not img:
                continue
            samples.append({"uid": uid, "findings": findings, "impression": impression, "image_path": img})
    logger.info(f"Loaded {len(samples)} samples")
    rng.shuffle(samples)
    n = len(samples)
    t = int(n * CONFIG["train_ratio"])
    v = int(n * (CONFIG["train_ratio"] + CONFIG["val_ratio"]))
    return samples[:t], samples[t:v], samples[v:]


def build_training_dicts(raw):
    rng = random.Random(CONFIG["seed"])
    out = []
    for s in raw:
        try:
            img = Image.open(s["image_path"]).convert("RGB").resize((336, 336), Image.LANCZOS)
        except Exception:
            continue
        answer = (s.get("impression") or s.get("findings") or "").strip()
        if not answer:
            continue
        out.append({"image": img, "question": rng.choice(VQA_TEMPLATES), "answer": answer})
    return out


# ============================================================
# 7. Collator
# ============================================================
class VQACollator:
    """No truncation — LLaVA expands <image> to ~576 tokens which must not be cut."""

    def __init__(self, processor, max_answer_words=100):
        self.processor = processor
        self.max_answer_words = max_answer_words

    def __call__(self, batch):
        images, texts = [], []
        for s in batch:
            words = s["answer"].split()
            answer = " ".join(words[:self.max_answer_words]) if len(words) > self.max_answer_words else s["answer"]
            images.append(s["image"])
            texts.append(f"USER: <image>\n{s['question']}\nASSISTANT: {answer}")
        enc = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = enc["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        enc["labels"] = labels
        return enc


# ============================================================
# 8. Main
# ============================================================
def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("KAGGLE QLORA TRAINING -- LLaVA-1.5-7B on OpenI")
    logger.info("=" * 60)

    assert torch.cuda.is_available(), "No GPU!"
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}  |  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    logger.info(f"PyTorch {torch.__version__}  |  CUDA {torch.version.cuda}  |  Devices: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    # ---- Data ----
    paths = discover_paths(CONFIG["kaggle_dataset_root"])
    train_raw, val_raw, test_raw = load_dataset(paths)
    train_data = build_training_dicts(train_raw)
    val_data = build_training_dicts(val_raw)
    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_raw)}")
    if not train_data:
        raise RuntimeError("No training samples!")

    # ---- Load model ----
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    free_vram = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9
    logger.info(f"Loading {CONFIG['model_id']} ... (free VRAM: {free_vram:.1f} GB)")
    if free_vram < 12:
        raise RuntimeError(
            f"Only {free_vram:.1f} GB free VRAM — need at least 12 GB.\n"
            f"A previous run left stale tensors on GPU.\n"
            f">>> RESTART THE KERNEL: Runtime -> Restart Session <<<\n"
            f"Then re-run this cell."
        )

    processor = AutoProcessor.from_pretrained(CONFIG["model_id"])
    model = LlavaForConditionalGeneration.from_pretrained(
        CONFIG["model_id"],
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ),
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory={0: "13GiB", "cpu": "24GiB"},
        low_cpu_mem_usage=True,
    )
    gc.collect(); torch.cuda.empty_cache()
    logger.info(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # ---- Prepare for training ----
    model = prepare_model_for_kbit_training(model)

    # ---- Discover actual module names for LoRA ----
    #   We want q/k/v/o_proj ONLY in the language model, NOT in vision_tower.
    #   Strategy:
    #     1. Use simple target_modules ["q_proj", ...] — PEFT matches all modules
    #        whose name ends with these strings.
    #     2. AFTER get_peft_model(), freeze any LoRA params inside vision_tower
    #        and multi_modal_projector so only language model LoRA trains.
    #
    #   This is robust regardless of model nesting or transformers version.

    # Log module structure for debugging
    q_proj_modules = [n for n, _ in model.named_modules() if n.endswith("q_proj")]
    lm_projs = [n for n in q_proj_modules if "vision_tower" not in n and "multi_modal" not in n]
    vt_projs = [n for n in q_proj_modules if "vision_tower" in n]
    logger.info(f"q_proj modules: {len(q_proj_modules)} total ({len(lm_projs)} LM, {len(vt_projs)} vision)")
    if lm_projs:
        logger.info(f"  Sample LM path: {lm_projs[0]}")
    if vt_projs:
        logger.info(f"  Sample VT path: {vt_projs[0]}")

    # Apply LoRA to ALL q/k/v/o_proj modules
    lora_config = LoraConfig(
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Now freeze LoRA adapters in vision_tower and multi_modal_projector.
    # This leaves ONLY language model LoRA params trainable.
    frozen_lora = 0
    for name, param in model.named_parameters():
        if param.requires_grad and ("vision_tower" in name or "multi_modal_projector" in name):
            param.requires_grad = False
            frozen_lora += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"LoRA attached. Frozen {frozen_lora} non-LM adapter params.")
    logger.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")

    # ---- Training ----
    out = CONFIG["output_dir"]
    training_args = TrainingArguments(
        output_dir=out,
        num_train_epochs=CONFIG["num_epochs"],
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=CONFIG["lr"],
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        eval_strategy="steps",
        save_total_limit=2,
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        report_to="none",
        seed=CONFIG["seed"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=VQACollator(processor),
    )

    gc.collect(); torch.cuda.empty_cache()
    logger.info(f"Pre-train VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    logger.info("Starting training ...")

    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        emergency = f"{out}/emergency_adapter"
        os.makedirs(emergency, exist_ok=True)
        model.save_pretrained(emergency)
        processor.save_pretrained(emergency)
        logger.info(f"Emergency save: {emergency}")
        raise

    logger.info(f"Training complete! ({(time.time()-t0)/3600:.1f} hours)")

    # ---- Save ----
    adapter_dir = f"{out}/final_adapter"
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    logger.info(f"Adapter saved: {adapter_dir}")
    logger.info("=" * 60)
    logger.info("Download 'final_adapter/' from the Output tab.")
    logger.info("Local: python scripts/inference.py --adapter checkpoints/llava-medical-vqa/final_adapter")
    logger.info("=" * 60)


# ============================================================
main()
