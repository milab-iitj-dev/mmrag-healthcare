# Healthcare Multimodal RAG

**Multimodal Retrieval-Augmented Generation for Medical Visual Question Answering**

A modular system for medical image understanding that combines vision-language models with retrieval-augmented generation. Built on LLaVA-1.5-7B and ColQwen2 for chest X-ray analysis using the OpenI dataset.

**Lab**: Machine Intelligence Lab · IIT Jodhpur  
**Author**: Gokul  
**License**: Apache 2.0

> **Note**: This project does not currently have access to a local GPU. All GPU-dependent validation was performed on **Kaggle (Tesla T4)**. Both Phase 1 and Phase 2 are confirmed working. See [Kaggle Validation](#kaggle-validation) for details.

---

## Features

### Phase 1 — Direct VQA ✅
- LLaVA-1.5-7B with 4-bit quantization (QLoRA-ready)
- Direct image → question → answer pipeline
- OpenI chest X-ray dataset support
- Local training, inference, and validation scripts
- **Validated on Kaggle**: 5/5 samples, 7.26s avg inference, 4.15 GB VRAM

### Phase 2 — RAG-Augmented VQA ✅
- ColQwen2 multi-vector embeddings with MaxSim retrieval
- Offline index building for the medical knowledge base
- Context-aware generation with retrieved evidence
- Full RAG pipeline: retrieve → context → generate
- **Validated on Kaggle**: 3826 documents indexed, 13.33s avg query time

---

## Project Structure

```
├── src/                          # Modular source code
│   ├── ingestion/                # Data loading (OpenI chest X-rays)
│   ├── embeddings/               # ColQwen2 multi-vector embeddings
│   ├── indexing/                 # Document store and index building
│   ├── retrieval/                # ColQwen2 MaxSim + hybrid retrieval
│   ├── context/                  # Context building from retrieved evidence
│   ├── generation/               # VLM wrappers and RAG generator
│   ├── evaluation/               # Metrics and evaluation runner
│   ├── reranking/                # Cross-encoder reranking
│   ├── verification/             # Self-check utilities
│   └── utils/                    # Device, logging, image utilities
│
├── analysis/                     # Standalone analysis and ablation tools
│   └── ablation/                 # Modality-bias ablation analysis
│       ├── run_ablation.py       # Entry point — orchestrates full ablation
│       ├── baseline_evaluator.py # Question-ignored (pre-fix) evaluation
│       ├── current_evaluator.py  # Dual-index + RRF (post-fix) evaluation
│       ├── sensitivity_analyzer.py # Question sensitivity (Jaccard overlap)
│       ├── report_writer.py      # Observing document + JSON generation
│       └── verify_modules.py     # Quick verification with mock data
│
├── evaluation/                   # Benchmarking and evaluation framework
│   ├── datasets/                 # Test dataset builders (OpenI, VQA-RAD)
│   ├── metrics/                  # Retrieval and generation metrics
│   ├── runners/                  # Benchmark runners
│   └── reporting/                # Result formatting and export
│
├── pipelines/                    # End-to-end orchestrators
│   ├── simple_vqa.py             # Phase 1: Direct VQA
│   ├── rag_vqa.py                # Phase 2: RAG VQA
│   └── offline_indexing.py       # Index builder
│
├── ui/                           # Professional inference UI (Gradio)
│   ├── app.py                    # Gradio Blocks application
│   ├── theme.py                  # Custom dark medical theme
│   └── formatters.py             # Output formatting utilities
│
├── scripts/                      # Training, inference, validation CLI
│   ├── launch_ui.py              # Web UI launcher
│   ├── inference.py              # Single/batch inference
│   ├── train_qlora.py            # QLoRA fine-tuning
│   ├── validate.py               # Validation smoke tests
│   └── ...
│
├── kaggle/                       # Self-contained Kaggle scripts
│   ├── kaggle_inference.py       # Phase 1 evaluation
│   ├── kaggle_rag.py             # Phase 2 full validation
│   └── train_kaggle.py           # QLoRA training on Kaggle
│
├── configs/                      # YAML configuration files
├── docs/                         # Architecture and domain documentation
├── outputs/kaggle_validation/    # Saved Kaggle validation results
├── tests/                        # Test suite
├── LICENSE                       # Apache 2.0
└── CITATION.cff                  # Citation metadata
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/milab-iitj-dev/mmrag-healthcare.git
cd mmrag-healthcare
pip install -e .
```

### Phase 1 — Direct Inference

```bash
# Single image inference
python scripts/inference.py --image path/to/xray.png

# Batch evaluation
python scripts/inference.py --batch-eval --max-samples 5

# Validation smoke test
python scripts/validate.py --quick
```

### Phase 1 — QLoRA Training

```bash
python scripts/train_qlora.py \
    --model-config configs/model_config.yaml \
    --data-config configs/data_config.yaml \
    --training-config configs/training_config.yaml
```

### Phase 2 — RAG Pipeline

```bash
# Step 1: Build the retrieval index (run once)
python -m pipelines.offline_indexing \
    --data-config configs/data_config.yaml \
    --retrieval-config configs/retrieval_config.yaml

# Step 2: Run RAG-augmented inference
python -m pipelines.rag_vqa \
    --query "What does this chest X-ray show?" \
    --query-image path/to/xray.png
```

### Web UI — Interactive Inference

```bash
# Full mode (GPU required)
python scripts/launch_ui.py

# UI preview only (no GPU)
python scripts/launch_ui.py --ui-only

# HPC with public URL
python scripts/launch_ui.py --share
```

Open **http://localhost:7860** — upload a chest X-ray, type a question, click **Analyze**.

### Ablation Analysis — Modality Bias Evaluation

The ablation module compares **baseline** (question-ignored image+text retrieval) vs **current system** (dual-index + RRF + question-aware reranking) to quantify the modality-bias fix.

```bash
# Quick verification with mock data (no GPU required)
python -m analysis.ablation.verify_modules

# Full ablation analysis (requires GPU + built index)
python -m analysis.ablation.run_ablation \
    --retrieval-config configs/retrieval_config.yaml \
    --data-config configs/data_config.yaml \
    --index-dir data/indexes/colqwen2_index \
    --max-queries 50 \
    --output-dir outputs/observations
```

**Output:**
- `outputs/observations/observing_document.md` — Full analysis report with side-by-side comparison tables
- `outputs/observations/ablation_results.json` — Raw metrics for programmatic analysis

The ablation measures:
- **Recall@k, MRR, nDCG@k** for both baseline and current system
- **Question sensitivity** via pairwise Jaccard overlap (same image, different questions)
- **Per-mode breakdown** across hybrid, text-only, and image-only queries

---

## Architecture

```
Query (image + text)
        │
        ▼
┌──────────────────┐
│  ColQwen2 Embed  │  ← Multi-vector patch embeddings
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  MaxSim Retrieve │  ← Late-interaction retrieval
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Context Builder │  ← Format retrieved evidence
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  LLaVA Generate  │  ← Grounded answer generation
└──────────────────┘
```

---

## Kaggle Validation

Since a local GPU is not available, both pipelines were validated on **Kaggle** using Tesla T4 GPUs.

### Phase 1 Results

| Metric | Value |
|--------|-------|
| Samples | 5/5 successful |
| Avg Inference Time | 7.26s |
| VRAM | 4.15 GB |
| GPU | Tesla T4 |
| Model | LLaVA-1.5-7B + LoRA (4-bit NF4) |

### Phase 2 Results

| Metric | Value |
|--------|-------|
| Documents Indexed | 3826 (full OpenI) |
| Queries Evaluated | 3 (image+text) |
| Avg Retrieval Time | 5.8s |
| Avg Generation Time | 7.53s |
| Retrieval Model | ColQwen2 v1.0 |
| Generation Model | LLaVA-1.5-7B (4-bit + LoRA) |

**Full validation results**: See [`outputs/kaggle_validation/`](outputs/kaggle_validation/) for complete reports, CSVs, JSONs, and console logs.

**Kaggle scripts**: See [`kaggle/`](kaggle/) for the self-contained scripts used for validation.

**Detailed documentation**: See [`docs/kaggle_validation.md`](docs/kaggle_validation.md) for comprehensive validation methodology and results.

---

## Requirements

- Python 3.10+
- CUDA GPU (T4 16GB or better)
- ~5.2 GB VRAM for LLaVA (4-bit quantized)
- ~6.5 GB VRAM for ColQwen2

See [requirements.txt](requirements.txt) for full dependencies.

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Citation

If you use this work, please cite using the [CITATION.cff](CITATION.cff) file.
