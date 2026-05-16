# 🏥 Healthcare Multimodal RAG

**A Modular Multimodal Retrieval-Augmented Generation System for Medical Visual Question Answering**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A production-quality, modular pipeline that answers clinical questions about medical images (chest X-rays) by combining vision-language models with retrieval from a medical knowledge base.

> **Current Status:** Phase 1 ✅ — Minimal VQA pipeline (Image → Question → Answer) with LLaVA-1.5-7B 4-bit + QLoRA fine-tuning on OpenI chest X-rays.

---

## Architecture

```
Query Image + Question
        │
        ▼
┌─────────────────────────────────────────────────┐
│  RETRIEVAL STAGE (Phase 3-4)                    │
│  BM25 (text) + CLIP (vision) + ColQwen2 (late)  │
│           └─────── RRF Fusion ─────┘            │
│              Cross-Encoder Reranking            │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│  CONTEXT BUILDING (Phase 3+)                    │
│  Top-k evidence docs → structured prompt        │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│  GENERATION (VLM)                               │
│  LLaVA-1.5-7B (Phase 1) → Qwen2-VL (Phase 5)  │
│  Image + Question + Context → Answer            │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│  GROUNDING & SAFETY (Phase 5)                   │
│  NLI verification + confidence + disclaimers    │
└─────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/Gokul-44/Healthcare-Multimodal-RAG.git
cd Healthcare-Multimodal-RAG

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Prepare Data

Download the [OpenI Chest X-ray dataset](https://openi.nlm.nih.gov/faq#collection) and place it as:

```
data/
└── openi/
    ├── images/              # X-ray PNG files
    │   ├── 1_IM-0001-3001.dcm.png
    │   └── ...
    └── reports/             # CSV report files
        ├── indiana_reports.csv
        └── indiana_projections.csv
```

### 3. Validate Setup (No GPU Required)

```bash
python scripts/validate.py --quick
```

### 4. Run Inference (Requires GPU)

```bash
# Single image
python scripts/inference.py --image data/openi/images/1_IM-0001-3001.dcm.png \
                            --question "What does this chest X-ray show?"

# Batch evaluation (5 samples)
python scripts/inference.py --batch-eval --max-samples 5

# Interactive mode
python scripts/inference.py
```

### 5. Train with QLoRA (Requires CUDA GPU)

```bash
python scripts/train_local.py \
    --model-config configs/model_config.yaml \
    --data-config configs/data_config.yaml \
    --training-config configs/training_config.yaml
```

After training, the adapter is saved to `checkpoints/llava-medical-vqa/final_adapter/`.

### 6. Inference with Trained Adapter

```bash
python scripts/inference.py \
    --adapter checkpoints/llava-medical-vqa/final_adapter \
    --image path/to/xray.png
```

---

## Project Structure

```
Healthcare-Multimodal-RAG/
├── configs/                     # YAML configuration files
│   ├── model_config.yaml        #   Model, quantization, LoRA settings
│   ├── data_config.yaml         #   Dataset paths, preprocessing
│   ├── training_config.yaml     #   QLoRA training hyperparameters
│   ├── retrieval_config.yaml    #   BM25, CLIP, ColQwen2, RRF (Phase 3-4)
│   ├── pipeline_config.yaml     #   Pipeline mode selection
│   └── evaluation_config.yaml   #   Metrics configuration
│
├── src/                         # Core source modules
│   ├── data/                    #   Dataset loaders (OpenI) ✅
│   ├── models/                  #   VLM wrappers (LLaVA, Qwen2-VL) ✅
│   ├── retrieval/               #   Retrievers (BM25, CLIP, ColQwen2, RRF)
│   ├── indexing/                #   Offline knowledge-base builder
│   ├── embeddings/              #   CLIP & ColQwen2 embedders
│   ├── reranking/               #   Cross-encoder reranking
│   ├── context/                 #   Context assembly for prompts
│   ├── generation/              #   RAG generation + grounding
│   └── utils/                   #   Device, logging, image, metrics ✅
│
├── pipelines/                   # End-to-end pipeline orchestrators
│   ├── simple_vqa.py            #   Phase 1: Image → Q → A ✅
│   ├── rag_vqa.py               #   Phase 3+: Retrieval-augmented VQA
│   ├── offline_indexing.py      #   Phase 3-4: Build knowledge base
│   └── evaluation.py            #   Phase 2+: Metrics computation
│
├── scripts/                     # CLI entry points
│   ├── train_local.py           #   QLoRA training ✅
│   ├── inference.py             #   Single/batch/interactive inference ✅
│   ├── validate.py              #   Smoke tests ✅
│   ├── build_index.py           #   Build retrieval index (Phase 3-4)
│   ├── evaluate.py              #   Run evaluation (Phase 2+)
│   ├── run_rag.py               #   RAG inference (Phase 3+)
│   └── download_data.py         #   Dataset download utility
│
├── tests/                       # Test suite
├── docs/                        # Architecture documentation & diagrams
├── data/                        # Raw data (gitignored)
├── checkpoints/                 # Trained adapters (gitignored)
├── outputs/                     # Results & logs (gitignored)
└── experiments/                 # Experiment tracking
```

> ✅ = Implemented | No mark = Placeholder for future phases

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed file descriptions.

---

## Key Technologies

| Component | Technology | Phase |
|-----------|-----------|-------|
| **VLM (current)** | LLaVA-1.5-7B, 4-bit NF4 quantization | Phase 1 ✅ |
| **Fine-tuning** | QLoRA (LoRA r=16, α=32) | Phase 1 ✅ |
| **Dataset** | OpenI Chest X-rays (~3,800 images + reports) | Phase 1 ✅ |
| **Text Retrieval** | BM25 (Okapi) | Phase 3 |
| **Vision Retrieval** | CLIP / BiomedCLIP + FAISS | Phase 4 |
| **Document Retrieval** | ColQwen2 (late-interaction MaxSim) | Phase 4 |
| **Fusion** | Reciprocal Rank Fusion (RRF) | Phase 4 |
| **Reranking** | Cross-encoder (MiniLM) | Phase 4 |
| **VLM (final)** | Qwen2-VL-7B-Instruct | Phase 5 |
| **Grounding** | NLI-based claim verification | Phase 5 |

---

## Phase Roadmap

| Phase | Goal | Status |
|-------|------|--------|
| **1** | Minimal VQA pipeline (Image→Q→A) with LLaVA-1.5-7B 4-bit | ✅ Complete |
| **2** | Evaluation metrics (BLEU, ROUGE, BERTScore) | ⬜ Next |
| **3** | BM25 text retrieval integration | ⬜ Planned |
| **4** | Full multimodal retrieval (ColQwen2 + CLIP + RRF + Reranking) | ⬜ Planned |
| **5** | Qwen2-VL + NLI grounding + safety verification | ⬜ Planned |

---

## Documentation

| Document | Description |
|----------|-------------|
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | Full folder tree with file explanations |
| [STEP_BY_STEP_GUIDE.md](STEP_BY_STEP_GUIDE.md) | Beginner-friendly guide to all techniques |
| [TRACKER.md](TRACKER.md) | Project progress and file inventory |
| [docs/architecture_explained.md](docs/architecture_explained.md) | Architecture deep dive |
| [docs/phase1_implementation_explained.md](docs/phase1_implementation_explained.md) | Phase 1 details |

---

## Hardware Requirements

| Mode | Minimum GPU | VRAM | Notes |
|------|------------|------|-------|
| Inference (4-bit) | NVIDIA T4 | 8 GB | Base model + adapter |
| Training (QLoRA) | NVIDIA T4 | 16 GB | batch=2, grad_accum=4 |
| Inference (CPU) | None | 16 GB RAM | Slow (~60-180s per image) |

---

## Author

**Gokul** — [GitHub](https://github.com/Gokul-44)

---

## License

This project is for educational and research purposes.
