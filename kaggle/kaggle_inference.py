"""
Phase 1 Evaluation: LLaVA-1.5-7B + LoRA — Medical VQA on OpenI (Kaggle GPU).

FINAL EVALUATION VERSION — Produces professor-ready outputs.

SETUP:
  1. Kaggle notebook → GPU T4 → Internet ON
  2. Add datasets: your adapter + OpenI chest X-rays
  3. Paste this entire script into one cell → Run
  4. Download results from /kaggle/working/results/

OUTPUT FILES:
  results/evaluation_results.json   — structured results
  results/evaluation_results.csv    — spreadsheet-friendly
  results/evaluation_report.md      — markdown report for presentation
  results/console_output.txt        — full console log
"""

# ============================================================
# 1. Environment
# ============================================================
import os, subprocess, sys, io

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def _pip(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

_pip("transformers>=4.40.0")
_pip("accelerate>=0.27.0")
_pip("bitsandbytes>=0.43.0")
_pip("peft>=0.10.0")

# ============================================================
# 2. Imports
# ============================================================
import csv, re, json, time, logging, gc
from pathlib import Path
from datetime import datetime

import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

try:
    from transformers import LlavaProcessor
except ImportError:
    from transformers import AutoProcessor as LlavaProcessor

# ============================================================
# 3. Logging — capture to file + console
# ============================================================
RESULTS_DIR = Path("/kaggle/working/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

log_buffer = io.StringIO()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(log_buffer),
    ],
)
logger = logging.getLogger("eval")

# ============================================================
# 4. Configuration
# ============================================================
MODEL_ID = "llava-hf/llava-1.5-7b-hf"

ADAPTER_PATHS = [
    "/kaggle/input/llava-medical-adapter/final_adapter",
    "/kaggle/input/llava-medical-adapter",
    "/kaggle/working/llava-medical-vqa/final_adapter",
]

DATASET_ROOTS = [
    "/kaggle/input/datasets/raddar/chest-xrays-indiana-university",
    "/kaggle/input/chest-xrays-indiana-university",
]

# Generation settings — tuned for concise, focused radiology answers
#   max_new_tokens=64     → forces short answers (typical radiology report is 1-3 sentences)
#   repetition_penalty=1.5 → penalizes repeated tokens
#   no_repeat_ngram_size=3 → blocks any 3-word phrase from repeating
#   do_sample=False        → greedy decoding (deterministic, reproducible)
GEN_CONFIG = {
    "max_new_tokens": 64,
    "do_sample": False,
    "repetition_penalty": 1.5,
    "no_repeat_ngram_size": 3,
}

# Evaluation questions — short-answer clinical queries
# Phrased to elicit focused, brief radiology responses
EVAL_QUESTIONS = [
    "Briefly describe the key findings in this chest X-ray.",
    "Is there cardiomegaly? Answer briefly.",
    "Are the lungs clear or abnormal? State findings only.",
    "Is there pleural effusion? Answer in one sentence.",
    "What is the radiological impression? Be concise.",
]


# ============================================================
# 5. Path Discovery
# ============================================================
def find_adapter():
    """Locate the LoRA adapter directory."""
    for p in ADAPTER_PATHS:
        path = Path(p)
        if path.exists() and (path / "adapter_config.json").exists():
            return str(path)
        for sub in path.glob("*/adapter_config.json"):
            return str(sub.parent)
    # Fallback scan
    for cfg in Path("/kaggle/input").rglob("adapter_config.json"):
        return str(cfg.parent)
    return None


def find_dataset():
    """Locate OpenI images and reports CSV."""
    for root in DATASET_ROOTS:
        p = Path(root)
        if p.exists():
            for d in p.rglob("images_normalized"):
                if d.is_dir():
                    reports = next(p.rglob("indiana_reports.csv"), None)
                    return {"images_dir": d, "reports_csv": reports}
    for d in Path("/kaggle/input").rglob("images_normalized"):
        if d.is_dir():
            reports = next(Path("/kaggle/input").rglob("indiana_reports.csv"), None)
            return {"images_dir": d, "reports_csv": reports}
    return None


# ============================================================
# 6. Model Loading
# ============================================================
def load_model(adapter_path=None):
    """
    Load LLaVA-1.5-7B in 4-bit quantization + LoRA adapter.

    4-bit NF4 quantization compresses the 7B model from ~14 GB (float16)
    to ~4 GB VRAM, enabling inference on T4/P100 GPUs.
    """
    logger.info(f"Loading processor: {MODEL_ID}")
    processor = LlavaProcessor.from_pretrained(MODEL_ID)

    logger.info("Loading model (4-bit NF4 quantization) ...")
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ),
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    vram = torch.cuda.memory_allocated() / 1e9
    logger.info(f"Base model loaded — VRAM: {vram:.2f} GB")

    if adapter_path:
        logger.info(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        logger.info(f"Adapter loaded — VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    model.eval()
    return model, processor


# ============================================================
# 7. Inference
# ============================================================
def clean_answer(text):
    """Clean generated text: remove repetition, trailing filler, hallucinated extras."""
    text = text.strip()
    # Remove repeated ". . . ." patterns anywhere
    text = re.sub(r'(\. ){2,}', '. ', text)
    text = re.sub(r'(\.\s*){3,}', '.', text)
    # Remove trailing incomplete sentences (cut at last period)
    if '.' in text:
        last_period = text.rfind('.')
        # Keep everything up to and including the last period
        # unless the last period IS the last char
        if last_period < len(text) - 1:
            text = text[:last_period + 1]
    # Remove filler phrases that indicate hallucination
    filler = [
        r'\s*If you need further.*$',
        r'\s*Clinical correlation is recommended.*$',
        r'\s*Recommend followup.*$',
        r'\s*Follow up.*$',
        r'\s*please schedule.*$',
        r'\s*please contact.*$',
    ]
    for pattern in filler:
        text = re.sub(pattern, '.', text, flags=re.IGNORECASE)
    # Final cleanup
    text = re.sub(r'\.{2,}', '.', text)  # ".." → "."
    text = re.sub(r'\s+', ' ', text)      # collapse whitespace
    return text.strip()


def infer(model, processor, image, question, **gen_kwargs):
    """
    Run single-image VQA inference.

    Pipeline internals:
      1. LlavaProcessor encodes the image via CLIPImageProcessor (336×336,
         normalize) and tokenizes the text prompt via LlamaTokenizer
      2. CLIP ViT-L/14 vision tower encodes pixel_values → 576 visual tokens
      3. Multi-modal projector maps visual tokens into LLM embedding space
      4. Llama-2-7B (quantized) + LoRA adapters autoregressively generates
         answer tokens conditioned on [visual_tokens + text_tokens]
      5. Generated token IDs are decoded back to text

    Args:
        image: PIL.Image or file path string
        question: clinical question about the image

    Returns:
        dict with answer, timing, token counts
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")

    # Prompt engineered for concise radiology-style answers
    # The instruction "Answer in 1-2 sentences" constrains verbosity
    prompt = (
        f"USER: <image>\n"
        f"{question} Answer in 1-2 sentences as a radiologist.\n"
        f"ASSISTANT:"
    )
    inputs = processor(text=prompt, images=image, return_tensors="pt")

    # Align dtype: processor outputs float32, model expects float16
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    inputs = {
        k: v.to(device=device, dtype=dtype) if v.is_floating_point()
        else v.to(device=device)
        for k, v in inputs.items()
    }

    config = {**GEN_CONFIG, **gen_kwargs}

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(**inputs, **config)
    elapsed = time.time() - t0

    input_len = inputs["input_ids"].shape[-1]
    generated = output_ids[0, input_len:]
    raw_answer = processor.decode(generated, skip_special_tokens=True).strip()
    answer = clean_answer(raw_answer)

    return {
        "answer": answer,
        "raw_answer": raw_answer,
        "time_sec": round(elapsed, 2),
        "input_tokens": int(input_len),
        "output_tokens": int(len(generated)),
        "question": question,
    }


# ============================================================
# 8. Sample Selection
# ============================================================
def select_eval_samples(dataset_info, n=5):
    """
    Select N diverse samples from OpenI for evaluation.

    Strategy: pick samples with known ground-truth reports,
    preferring a mix of normal and abnormal findings.
    """
    images_dir = dataset_info["images_dir"]
    reports_csv = dataset_info.get("reports_csv")
    samples = []

    if reports_csv and reports_csv.exists():
        with open(reports_csv, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        # Categorize by findings
        normal, abnormal = [], []
        for row in rows:
            uid = row.get("uid", "").strip()
            findings = row.get("findings", "").strip()
            impression = row.get("impression", "").strip()
            report = impression or findings
            if not report or len(report) < 5:
                continue
            report = re.sub(r"\bXXXX\b", "", report).strip()

            hits = sorted(images_dir.glob(f"{uid}_*"))
            if not hits:
                continue

            entry = {"image": str(hits[0]), "report": report, "uid": uid}
            if any(kw in report.lower() for kw in ["normal", "clear", "no acute", "unremarkable"]):
                normal.append(entry)
            else:
                abnormal.append(entry)

        # Mix: try 2 normal + 3 abnormal (or whatever is available)
        n_normal = min(2, len(normal))
        n_abnormal = min(n - n_normal, len(abnormal))
        n_normal = min(n - n_abnormal, len(normal))  # fill remainder

        samples = normal[:n_normal] + abnormal[:n_abnormal]
        # Fill if still short
        remaining = n - len(samples)
        if remaining > 0:
            all_samples = normal + abnormal
            for s in all_samples:
                if s not in samples:
                    samples.append(s)
                    if len(samples) >= n:
                        break
    else:
        pngs = sorted(images_dir.glob("*.png"))[:n]
        samples = [{"image": str(p), "report": None, "uid": p.stem} for p in pngs]

    logger.info(f"Selected {len(samples)} evaluation samples")
    return samples[:n]


# ============================================================
# 9. Batch Evaluation
# ============================================================
def run_evaluation(model, processor, dataset_info, n=5):
    """
    Run full evaluation on N OpenI samples with different questions.

    Each sample gets a unique clinical question to demonstrate
    the model's range of medical VQA capabilities.
    """
    samples = select_eval_samples(dataset_info, n)
    results = []
    total_time = 0

    logger.info("=" * 60)
    logger.info(f"EVALUATION — {len(samples)} samples")
    logger.info("=" * 60)

    for i, sample in enumerate(samples):
        question = EVAL_QUESTIONS[i % len(EVAL_QUESTIONS)]
        img_name = Path(sample["image"]).name

        logger.info(f"\n--- Sample {i+1}/{len(samples)} ---")
        logger.info(f"  UID:   {sample['uid']}")
        logger.info(f"  Image: {img_name}")
        logger.info(f"  Q:     {question}")

        try:
            result = infer(model, processor, sample["image"], question)
            result["sample_id"] = i + 1
            result["uid"] = sample["uid"]
            result["image_path"] = sample["image"]
            result["image_name"] = img_name
            result["ground_truth"] = sample.get("report")
            results.append(result)
            total_time += result["time_sec"]

            logger.info(f"  A:     {result['answer']}")
            if sample.get("report"):
                logger.info(f"  GT:    {sample['report'][:200]}")
            logger.info(f"  Time:  {result['time_sec']}s")

        except Exception as e:
            logger.error(f"  FAILED: {e}")
            results.append({
                "sample_id": i + 1, "uid": sample["uid"],
                "image_path": sample["image"], "image_name": img_name,
                "question": question, "answer": f"ERROR: {e}",
                "time_sec": 0, "ground_truth": sample.get("report"),
                "input_tokens": 0, "output_tokens": 0,
            })

    # Summary
    successful = [r for r in results if not r["answer"].startswith("ERROR")]
    avg_time = total_time / len(successful) if successful else 0
    vram = torch.cuda.memory_allocated() / 1e9

    summary = {
        "total_samples": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "avg_time_sec": round(avg_time, 2),
        "total_time_sec": round(total_time, 2),
        "vram_gb": round(vram, 2),
        "gpu": torch.cuda.get_device_name(0),
        "model": MODEL_ID,
        "adapter": "LoRA (r=16, α=32)",
        "quantization": "4-bit NF4",
        "timestamp": datetime.now().isoformat(),
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"  EVALUATION COMPLETE")
    logger.info(f"  Samples:  {summary['successful']}/{summary['total_samples']}")
    logger.info(f"  Avg time: {summary['avg_time_sec']}s per image")
    logger.info(f"  VRAM:     {summary['vram_gb']} GB")
    logger.info(f"{'='*60}")

    return results, summary


# ============================================================
# 10. Save Results
# ============================================================
def save_results(results, summary, output_dir=None):
    """Save evaluation results in JSON, CSV, and Markdown formats."""
    out = Path(output_dir or RESULTS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- JSON ---
    json_path = out / f"evaluation_results_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)
    logger.info(f"JSON saved: {json_path}")

    # --- CSV ---
    csv_path = out / f"evaluation_results_{ts}.csv"
    fields = ["sample_id", "uid", "image_name", "question", "answer", "ground_truth",
              "time_sec", "input_tokens", "output_tokens"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"CSV saved:  {csv_path}")

    # --- Markdown Report ---
    md_path = out / f"evaluation_report_{ts}.md"
    md = _generate_markdown_report(results, summary)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    logger.info(f"Report:     {md_path}")

    # --- Console Log ---
    log_path = out / f"console_output_{ts}.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log_buffer.getvalue())
    logger.info(f"Log saved:  {log_path}")

    return {"json": str(json_path), "csv": str(csv_path),
            "report": str(md_path), "log": str(log_path)}


def _generate_markdown_report(results, summary):
    """Generate a professional markdown evaluation report."""
    lines = [
        "# Phase 1 — Medical VQA Evaluation Report",
        "",
        f"**Date:** {summary['timestamp']}",
        f"**Model:** {summary['model']}",
        f"**Adapter:** {summary['adapter']}",
        f"**Quantization:** {summary['quantization']}",
        f"**GPU:** {summary['gpu']}",
        f"**VRAM Usage:** {summary['vram_gb']} GB",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Samples Evaluated | {summary['successful']}/{summary['total_samples']} |",
        f"| Avg Inference Time | {summary['avg_time_sec']}s |",
        f"| Total Time | {summary['total_time_sec']}s |",
        f"| GPU | {summary['gpu']} |",
        f"| VRAM | {summary['vram_gb']} GB |",
        "",
        "---",
        "",
        "## Results",
        "",
    ]

    for r in results:
        gt = r.get("ground_truth") or "N/A"
        lines.extend([
            f"### Sample {r['sample_id']} — UID: {r.get('uid', 'N/A')}",
            "",
            f"**Image:** `{r.get('image_name', 'N/A')}`",
            "",
            f"**Question:** {r['question']}",
            "",
            f"**Model Answer:**",
            f"> {r['answer']}",
            "",
            f"**Ground Truth:**",
            f"> {gt[:300]}",
            "",
            f"*Inference: {r['time_sec']}s | Tokens: {r.get('input_tokens', 0)} → {r.get('output_tokens', 0)}*",
            "",
            "---",
            "",
        ])

    lines.extend([
        "## Pipeline Architecture",
        "",
        "```",
        "Chest X-ray (PNG)",
        "  → PIL.Image.open().convert('RGB')",
        "  → LlavaProcessor",
        "      → CLIPImageProcessor: resize 336×336, normalize",
        "      → LlamaTokenizer: tokenize prompt",
        "  → LlavaForConditionalGeneration",
        "      → CLIP ViT-L/14: 576 visual tokens (frozen)",
        "      → Multi-Modal Projector: visual → LLM space",
        "      → Llama-2-7B + LoRA (r=16, α=32): generate answer",
        "  → Decode → Medical Answer",
        "```",
        "",
        "## Notes",
        "",
        "- Model uses 4-bit NF4 quantization (~4 GB VRAM)",
        "- LoRA adapters trained on OpenI chest X-ray dataset",
        "- Greedy decoding with repetition_penalty=1.5",
        "- Phase 1 only — no retrieval/RAG augmentation",
        "",
    ])

    return "\n".join(lines)


# ============================================================
# 11. Interactive Mode
# ============================================================
def interactive(model, processor, images_dir=None):
    """Interactive inference loop."""
    print("\n" + "="*60)
    print("  INTERACTIVE MODE — Type image path + question")
    print("  Type 'quit' to exit")
    print("="*60)

    while True:
        img_input = input("\nImage path or UID (or 'quit'): ").strip()
        if img_input.lower() in ("quit", "exit", "q"):
            break

        img_path = Path(img_input)
        if not img_path.exists() and images_dir:
            hits = sorted(images_dir.glob(f"{img_input}*"))
            if hits:
                img_path = hits[0]

        if not img_path.exists():
            print(f"  Not found: {img_input}")
            continue

        question = input("Question [What does this chest X-ray show?]: ").strip()
        if not question:
            question = "What does this chest X-ray show?"

        try:
            result = infer(model, processor, str(img_path), question)
            print(f"\n{'='*60}")
            print(f"  Image:    {img_path.name}")
            print(f"  Question: {question}")
            print(f"  Answer:   {result['answer']}")
            print(f"  Time:     {result['time_sec']}s")
            print(f"{'='*60}")
        except Exception as e:
            print(f"  Error: {e}")


# ============================================================
# 12. Main
# ============================================================
def main():
    t_start = time.time()

    logger.info("=" * 60)
    logger.info("PHASE 1 EVALUATION — LLaVA-1.5-7B + LoRA on OpenI")
    logger.info("=" * 60)

    # GPU
    assert torch.cuda.is_available(), "No GPU detected!"
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"GPU: {gpu}  |  VRAM: {vram:.1f} GB")
    logger.info(f"PyTorch: {torch.__version__}  |  CUDA: {torch.version.cuda}")

    # Adapter
    adapter_path = find_adapter()
    if adapter_path:
        logger.info(f"Adapter: {adapter_path}")
    else:
        logger.warning("No adapter found — running base model only")

    # Dataset
    dataset_info = find_dataset()
    if dataset_info:
        n_imgs = len(list(dataset_info["images_dir"].glob("*.png")))
        logger.info(f"Dataset: {dataset_info['images_dir']} ({n_imgs} images)")
    else:
        logger.error("OpenI dataset not found!")
        return

    # Load model
    model, processor = load_model(adapter_path)
    gc.collect(); torch.cuda.empty_cache()

    # Run evaluation
    results, summary = run_evaluation(model, processor, dataset_info, n=5)

    # Save outputs
    saved = save_results(results, summary)

    # Print file locations
    logger.info("\n" + "=" * 60)
    logger.info("OUTPUT FILES:")
    for k, v in saved.items():
        logger.info(f"  {k:8s}: {v}")
    logger.info(f"Total runtime: {(time.time()-t_start)/60:.1f} minutes")
    logger.info("=" * 60)
    logger.info("Download results/ from the Kaggle Output tab.")

    # Interactive
    interactive(model, processor, dataset_info["images_dir"])


# ============================================================
main()
