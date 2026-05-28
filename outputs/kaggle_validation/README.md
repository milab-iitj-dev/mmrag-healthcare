# Kaggle Validation Outputs

This directory contains the saved outputs from successful Kaggle GPU runs, serving as evidence that both Phase 1 and Phase 2 pipelines are fully working.

> **Note**: A local GPU is not available for this project. All GPU-dependent validation was performed on Kaggle (Tesla T4).

---

## Phase 1 — Direct VQA (`phase1_kaggle_results/`)

LLaVA-1.5-7B (4-bit NF4 + QLoRA) evaluated on 5 OpenI chest X-ray samples.

| File | Description |
|------|-------------|
| `phase1_evaluation_report.md` | Markdown evaluation report with all results |
| `phase1_evaluation_results.json` | Structured results (JSON) |
| `phase1_evaluation_results.csv` | Spreadsheet-friendly results (CSV) |
| `phase1_console_output.txt` | Full console log from Kaggle run |
| `screenshots/` | Kaggle notebook screenshots showing training & evaluation |

### Key Results

- **GPU**: Tesla T4
- **VRAM**: 4.15 GB
- **Samples**: 5/5 successful
- **Avg Inference Time**: 7.26s per image
- **Model**: LLaVA-1.5-7B + LoRA (r=16, α=32), 4-bit NF4

---

## Phase 2 — RAG Pipeline (`phase2_kaggle_results/`)

ColQwen2 retrieval + LLaVA generation on OpenI dataset.

### Run 1: 50-Sample Index (Quick Validation)

| File | Description |
|------|-------------|
| `phase2_report_50samples.md` | Markdown report with detailed results |
| `phase2_results_50samples.json` | Structured results (JSON) |
| `phase2_results_50samples.csv` | Spreadsheet-friendly results (CSV) |
| `phase2_console_50samples.txt` | Full console log |

- **Documents Indexed**: 50
- **Queries**: 5 (3 text-only + 2 image+text)
- **Embedding Dim**: 128
- **Index Build Time**: 239.27s
- **Avg Total Time**: 7.07s per query

### Run 2: Full 3826-Document Index (Complete Validation)

| File | Description |
|------|-------------|
| `phase2_report_3826docs.md` | Markdown report with detailed results |
| `phase2_results_3826docs.json` | Structured results (JSON) |
| `phase2_results_3826docs.csv` | Spreadsheet-friendly results (CSV) |
| `phase2_console_3826docs.txt` | Full console log |

- **Documents Indexed**: 3826 (entire OpenI dataset)
- **Queries**: 3 (all image+text)
- **Embedding Dim**: 128
- **Index Build Time**: 14734.65s (~4.1 hours)
- **Avg Total Time**: 13.33s per query

---

## Reproducing

To reproduce these results, use the scripts in [`kaggle/`](../../kaggle/):
- Phase 1: `kaggle/kaggle_inference.py`
- Phase 2: `kaggle/kaggle_rag.py`
- Training: `kaggle/train_kaggle.py`
