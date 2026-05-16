# Project Structure

This document maps every folder and file in the repository, explains its purpose, and shows how the modules connect.

---

## Repository Tree

```
Healthcare-Multimodal-RAG/
│
├── README.md                              # Project overview, setup, usage
├── PROJECT_STRUCTURE.md                   # ← This file
├── STEP_BY_STEP_GUIDE.md                  # Beginner-friendly technique guide
├── TRACKER.md                             # Progress tracking and file inventory
├── requirements.txt                       # Python dependencies (pip)
├── environment.yml                        # Conda environment specification
├── setup.py                               # Package installation (pip install -e .)
├── .gitignore                             # Git ignore rules
│
├── configs/                               # ── Configuration ──
│   ├── model_config.yaml                  # Model selection, quantization, LoRA
│   ├── data_config.yaml                   # Dataset paths, preprocessing, splits
│   ├── training_config.yaml               # QLoRA training hyperparameters
│   ├── retrieval_config.yaml              # BM25, CLIP, ColQwen2, RRF, reranking
│   ├── pipeline_config.yaml               # Pipeline mode selection and wiring
│   └── evaluation_config.yaml             # Metrics, eval splits, output formats
│
├── src/                                   # ── Core Source Code ──
│   ├── __init__.py
│   │
│   ├── data/                              # Dataset loading & preprocessing
│   │   ├── __init__.py
│   │   ├── base_dataset.py                # Abstract base: BaseDataset + MedicalSample
│   │   ├── openi_dataset.py               # OpenI Chest X-ray loader ✅
│   │   └── preprocessing.py               # Image/text transforms ✅
│   │
│   ├── models/                            # Vision-Language Model wrappers
│   │   ├── __init__.py
│   │   ├── base_vlm.py                    # Abstract base: BaseVLM + VLMOutput ✅
│   │   ├── llava_model.py                 # LLaVA-1.5-7B 4-bit wrapper ✅
│   │   ├── qwen2vl_model.py              # Qwen2-VL wrapper (Phase 5)
│   │   └── model_factory.py               # Config → model instantiation ✅
│   │
│   ├── retrieval/                         # Retrieval methods (Phase 3-4)
│   │   ├── __init__.py
│   │   ├── base_retriever.py              # Abstract base: BaseRetriever + RetrievedDocument
│   │   ├── bm25_retriever.py              # BM25 sparse text retrieval
│   │   ├── clip_retriever.py              # CLIP dense image-text retrieval
│   │   ├── colqwen2_retriever.py          # ColQwen2 late-interaction retrieval
│   │   └── hybrid_retriever.py            # RRF fusion of multiple retrievers
│   │
│   ├── indexing/                          # Offline knowledge-base building (Phase 3-4)
│   │   ├── __init__.py
│   │   ├── index_builder.py               # Full offline indexing pipeline
│   │   └── document_store.py              # Document storage and lookup
│   │
│   ├── embeddings/                        # Embedding models (Phase 4)
│   │   ├── __init__.py
│   │   ├── base_embedder.py               # Abstract base: BaseEmbedder
│   │   ├── clip_embedder.py               # CLIP / BiomedCLIP embeddings
│   │   └── colqwen2_embedder.py           # ColQwen2 multi-vector embeddings
│   │
│   ├── reranking/                         # Reranking (Phase 4-5)
│   │   ├── __init__.py
│   │   ├── base_reranker.py               # Abstract base: BaseReranker
│   │   └── cross_encoder_reranker.py      # Cross-encoder reranking
│   │
│   ├── context/                           # Context assembly (Phase 3+)
│   │   ├── __init__.py
│   │   ├── context_builder.py             # Assembles evidence into prompts
│   │   └── prompt_templates.py            # Prompt templates for all scenarios
│   │
│   ├── generation/                        # Generation & verification (Phase 3-5)
│   │   ├── __init__.py
│   │   ├── rag_generator.py               # RAG-augmented generation engine
│   │   └── grounding.py                   # NLI grounding + safety checks
│   │
│   └── utils/                             # Shared utilities
│       ├── __init__.py
│       ├── device.py                      # GPU detection, VRAM tracking ✅
│       ├── logging_utils.py               # Structured logging ✅
│       ├── image_utils.py                 # Image loading, resizing ✅
│       ├── config_loader.py               # YAML config loading and merging
│       ├── metrics.py                     # Evaluation metrics (BLEU, ROUGE, etc.)
│       └── visualization.py               # Result plotting and visualization
│
├── pipelines/                             # ── End-to-End Pipelines ──
│   ├── __init__.py
│   ├── simple_vqa.py                      # Phase 1: Image → Q → A ✅
│   ├── rag_vqa.py                         # Phase 3+: Retrieval-augmented VQA
│   ├── offline_indexing.py                # Phase 3-4: Build knowledge-base index
│   └── evaluation.py                      # Phase 2+: Full evaluation suite
│
├── scripts/                               # ── CLI Entry Points ──
│   ├── train_local.py                     # QLoRA training on local GPU ✅
│   ├── inference.py                       # Single/batch/interactive inference ✅
│   ├── validate.py                        # Smoke tests and data validation ✅
│   ├── build_index.py                     # Build offline retrieval index (Phase 3-4)
│   ├── evaluate.py                        # Run evaluation metrics (Phase 2+)
│   ├── run_rag.py                         # RAG inference (Phase 3+)
│   └── download_data.py                   # Download and prepare OpenI dataset
│
├── tests/                                 # ── Test Suite ──
│   ├── __init__.py
│   ├── test_data.py                       # Data loading and preprocessing tests
│   ├── test_models.py                     # VLM wrapper tests
│   ├── test_retrieval.py                  # Retrieval module tests
│   ├── test_pipelines.py                  # Pipeline integration tests
│   └── test_utils.py                      # Utility module tests
│
├── docs/                                  # ── Documentation ──
│   ├── architecture_explained.md          # Full architecture deep dive
│   ├── phase1_implementation_explained.md # Phase 1 implementation details
│   ├── arch_left.png                      # Architecture diagram (left half)
│   └── arch_right.png                     # Architecture diagram (right half)
│
├── data/                                  # ── Raw Data (gitignored) ──
│   └── .gitkeep
│
├── checkpoints/                           # ── Trained Adapters (gitignored) ──
│   └── .gitkeep
│
├── outputs/                               # ── Pipeline Outputs (gitignored) ──
│   └── .gitkeep
│
└── experiments/                           # ── Experiment Tracking ──
    ├── configs/.gitkeep
    ├── logs/.gitkeep
    └── results/.gitkeep
```

**Legend:** ✅ = Implemented | No mark = Placeholder for future phases

---

## Module Responsibilities

### `configs/` — Configuration

All settings live in YAML files. No hardcoded values in the codebase.

| File | Controls |
|------|----------|
| `model_config.yaml` | Which VLM to use, quantization bits, LoRA rank, generation params |
| `data_config.yaml` | Dataset paths, image resize, text lengths, train/val/test splits |
| `training_config.yaml` | Learning rate, epochs, batch size, gradient accumulation |
| `retrieval_config.yaml` | Which retrievers to enable, BM25/CLIP/ColQwen2 params, RRF weights |
| `pipeline_config.yaml` | Pipeline mode (simple_vqa / rag_vqa / full_rag) |
| `evaluation_config.yaml` | Which metrics to compute, eval split, output format |

### `src/data/` — Data Loading

Loads the OpenI dataset and converts every sample into a `MedicalSample` dataclass. All pipelines consume `MedicalSample` — they never touch raw files.

### `src/models/` — VLM Wrappers

Wraps each vision-language model behind the `BaseVLM` interface. Swapping LLaVA for Qwen2-VL = changing one line in config.

### `src/retrieval/` — Retrieval Methods

Each retrieval method implements `BaseRetriever`. `HybridRetriever` fuses them with RRF. Pipelines call `retriever.retrieve(query)` without knowing which method runs behind the scenes.

### `src/indexing/` — Knowledge Base Building

`IndexBuilder` runs the offline pipeline: load data → compute embeddings → build indexes → save to disk.

### `src/embeddings/` — Embedding Models

Wraps CLIP and ColQwen2 behind `BaseEmbedder`. Used by both indexing (offline) and retrieval (online).

### `src/reranking/` — Reranking

Cross-encoder rerankers re-order the top-k results from Stage 1 retrieval for higher precision.

### `src/context/` — Context Building

Assembles retrieval results into structured prompts for the VLM.

### `src/generation/` — Generation & Grounding

`RAGGenerator` orchestrates retrieve → build context → generate → verify. `Grounding` uses NLI to verify answers.

### `src/utils/` — Utilities

Config loading, metrics, GPU detection, image I/O, logging, visualization.

---

## Data Flow

```
configs/*.yaml
     │
     ▼
config_loader ──────────────────────────┐
     │                                  │
     ├──→ src/data/ (OpenIDataset)      │
     │         │                        │
     │         ▼                        │
     ├──→ src/indexing/ (offline) ──→ data/indexes/
     │                                  │
     ├──→ src/retrieval/ (online query) │
     │         │                        │
     │         ▼                        │
     ├──→ src/reranking/ ───────────────┤
     │         │                        │
     │         ▼                        │
     ├──→ src/context/ (build prompt)   │
     │         │                        │
     │         ▼                        │
     ├──→ src/models/ (VLM generate)    │
     │         │                        │
     │         ▼                        │
     └──→ src/generation/ (verify)      │
               │                        │
               ▼                        │
         pipelines/ (orchestrate) ◄─────┘
               │
               ▼
         scripts/ (CLI entry points)
```
