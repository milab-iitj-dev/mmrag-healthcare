# Kaggle Scripts — Self-Contained GPU Validation

These scripts are designed to run **entirely within a Kaggle notebook cell** — no local setup required. Each script installs its own dependencies, finds datasets automatically, and saves structured outputs.

## Prerequisites

- **Kaggle Account** with GPU enabled (T4 or P100)
- **Internet ON** in notebook settings
- **Dataset**: Add "OpenI Chest X-rays Indiana University" to your notebook
- **HuggingFace Token**: Set as a Kaggle secret (`HF_TOKEN`) or replace the placeholder in the script

## Scripts

### `train_kaggle.py` — Phase 1: QLoRA Fine-Tuning

Fine-tunes LLaVA-1.5-7B on OpenI chest X-rays using QLoRA (4-bit quantization + LoRA adapters).

| Setting | Value |
|---------|-------|
| Base Model | `llava-hf/llava-1.5-7b-hf` |
| Quantization | 4-bit NF4 |
| LoRA | r=16, α=32 |
| Epochs | 3 |
| VRAM | ~5 GB |

**Output**: `final_adapter/` directory (download from Kaggle Output tab)

### `kaggle_inference.py` — Phase 1: Evaluation

Evaluates the fine-tuned LLaVA model on 5 OpenI samples with different clinical questions.

**Outputs**:
- `results/evaluation_results_{timestamp}.json`
- `results/evaluation_results_{timestamp}.csv`
- `results/evaluation_report_{timestamp}.md`
- `results/console_output_{timestamp}.txt`

### `kaggle_rag.py` — Phase 2: Full RAG Pipeline Validation

End-to-end validation of the complete RAG pipeline:
1. **Offline**: Index all OpenI images with ColQwen2 multi-vector embeddings
2. **Online**: Query → ColQwen2 retrieval → Context building → LLaVA generation

| Component | Model |
|-----------|-------|
| Retrieval | ColQwen2 (`vidore/colqwen2-v1.0-hf`) |
| Generation | LLaVA-1.5-7B (4-bit + LoRA) |
| Scoring | MaxSim late-interaction |

**Outputs**:
- `results/phase2_results_{timestamp}.json`
- `results/phase2_results_{timestamp}.csv`
- `results/phase2_report_{timestamp}.md`
- `results/console_output_{timestamp}.txt`
- `colqwen2_index/` (saved retrieval index)

### `kaggle_fast_inference.py` — ⚡ Fast RAG Inference (Recommended for Demos)

Loads a **pre-built** ColQwen2 index and runs the full RAG pipeline in **~20 seconds per query** — no need to rebuild the 4-hour index.

**Prerequisites**: Save the `colqwen2_index/` output from a previous `kaggle_rag.py` run as a Kaggle dataset.

**Supports three query modes** (configured via simple variables at the top):
- `"text"` — text-only query
- `"image"` — image-only query
- `"image_text"` — image + text query (best results)

| Feature | Value |
|---------|-------|
| Index loading | ~5 seconds (vs ~4 hours to rebuild) |
| Per-query time | ~20 seconds |
| Batch mode | Yes (multiple queries in one run) |
| Output | JSON + Markdown report |

**Outputs**:
- `fast_inference_results/fast_inference_{timestamp}.json`
- `fast_inference_results/fast_inference_report_{timestamp}.md`

## How to Run

1. Create a new Kaggle notebook
2. Enable GPU (T4 x2 recommended for Phase 2)
3. Turn on Internet access
4. Add the OpenI dataset
5. Set your HuggingFace token as a Kaggle secret named `HF_TOKEN`
6. Paste the entire script into one cell
7. Run and wait for completion
8. Download results from the Output tab

## Validation Results

See [`outputs/kaggle_validation/`](../outputs/kaggle_validation/) for saved outputs from successful Kaggle runs.
