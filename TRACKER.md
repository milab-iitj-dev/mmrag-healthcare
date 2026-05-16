# Project Tracker — Healthcare Multimodal RAG

Track what's included, excluded, complete, and pending in this repository.

---

## Current Project State

| Attribute | Value |
|-----------|-------|
| **Phase** | Phase 1 — Simple VQA Pipeline |
| **Status** | ✅ Core pipeline working |
| **VLM** | LLaVA-1.5-7B (4-bit NF4 quantization) |
| **Dataset** | OpenI Chest X-rays (~3,800 samples) |
| **Fine-tuning** | QLoRA (LoRA r=16, α=32) |
| **Repo Version** | 1.0.0 (GitHub release) |

---

## Files Included in Repository

### ✅ Root Files

| File | Status | Notes |
|------|--------|-------|
| `README.md` | ✅ Complete | Full project overview, setup, usage |
| `PROJECT_STRUCTURE.md` | ✅ Complete | Folder tree with explanations |
| `STEP_BY_STEP_GUIDE.md` | ✅ Complete | Beginner-friendly technique guide |
| `TRACKER.md` | ✅ Complete | This tracking file |
| `requirements.txt` | ✅ Complete | All phase dependencies (future commented) |
| `environment.yml` | ✅ Complete | Conda environment spec |
| `setup.py` | ✅ Complete | Package installer with extras |
| `.gitignore` | ✅ Complete | Comprehensive ignore rules |

### ✅ Configuration Files (`configs/`)

| File | Status | Notes |
|------|--------|-------|
| `model_config.yaml` | ✅ Complete | LLaVA-1.5-7B, quantization, LoRA |
| `data_config.yaml` | ✅ Complete | OpenI paths, preprocessing, splits |
| `training_config.yaml` | ✅ Complete | QLoRA hyperparameters |
| `retrieval_config.yaml` | ✅ Complete | BM25, CLIP, ColQwen2, RRF (Phase 3-4) |
| `pipeline_config.yaml` | ✅ Complete | Pipeline mode selection |
| `evaluation_config.yaml` | ✅ Complete | Metrics configuration |

### ✅ Source Code — Implemented (`src/`)

| File | Status | Notes |
|------|--------|-------|
| `src/data/base_dataset.py` | ✅ Implemented | BaseDataset + MedicalSample |
| `src/data/openi_dataset.py` | ✅ Implemented | OpenI loader with CSV parsing |
| `src/data/preprocessing.py` | ✅ Implemented | Image transforms |
| `src/models/base_vlm.py` | ✅ Implemented | BaseVLM + VLMOutput |
| `src/models/llava_model.py` | ✅ Implemented | LLaVA-1.5-7B wrapper |
| `src/models/model_factory.py` | ✅ Implemented | Config → model factory |
| `src/utils/device.py` | ✅ Implemented | GPU detection, VRAM tracking |
| `src/utils/logging_utils.py` | ✅ Implemented | Structured logging |
| `src/utils/image_utils.py` | ✅ Implemented | Image I/O utilities |

### 📋 Source Code — Placeholders (`src/`)

| File | Status | Phase | Notes |
|------|--------|-------|-------|
| `src/models/qwen2vl_model.py` | 📋 Placeholder | 5 | Qwen2-VL wrapper |
| `src/retrieval/base_retriever.py` | 📋 Interface | 3 | BaseRetriever + RetrievedDocument |
| `src/retrieval/bm25_retriever.py` | 📋 Placeholder | 3 | BM25 sparse retrieval |
| `src/retrieval/clip_retriever.py` | 📋 Placeholder | 4 | CLIP dense retrieval |
| `src/retrieval/colqwen2_retriever.py` | 📋 Placeholder | 4 | ColQwen2 late-interaction |
| `src/retrieval/hybrid_retriever.py` | 📋 Placeholder | 4 | RRF fusion |
| `src/indexing/index_builder.py` | 📋 Placeholder | 3-4 | Offline index pipeline |
| `src/indexing/document_store.py` | 📋 Placeholder | 3 | Document storage |
| `src/embeddings/base_embedder.py` | 📋 Interface | 4 | BaseEmbedder |
| `src/embeddings/clip_embedder.py` | 📋 Placeholder | 4 | CLIP embedder |
| `src/embeddings/colqwen2_embedder.py` | 📋 Placeholder | 4 | ColQwen2 embedder |
| `src/reranking/base_reranker.py` | 📋 Interface | 4 | BaseReranker |
| `src/reranking/cross_encoder_reranker.py` | 📋 Placeholder | 4-5 | Cross-encoder |
| `src/context/context_builder.py` | 📋 Placeholder | 3 | Context assembly |
| `src/context/prompt_templates.py` | 📋 Placeholder | 1-3 | Prompt templates |
| `src/generation/rag_generator.py` | 📋 Placeholder | 3+ | RAG generation engine |
| `src/generation/grounding.py` | 📋 Placeholder | 5 | NLI grounding |
| `src/utils/config_loader.py` | 📋 Placeholder | 1 | Config loading utility |
| `src/utils/metrics.py` | 📋 Placeholder | 2 | Evaluation metrics |
| `src/utils/visualization.py` | 📋 Placeholder | 2 | Result visualization |

### ✅ Pipelines (`pipelines/`)

| File | Status | Notes |
|------|--------|-------|
| `pipelines/simple_vqa.py` | ✅ Implemented | Phase 1 end-to-end pipeline |
| `pipelines/rag_vqa.py` | 📋 Placeholder | Phase 3+ RAG pipeline |
| `pipelines/offline_indexing.py` | 📋 Placeholder | Phase 3-4 index building |
| `pipelines/evaluation.py` | 📋 Placeholder | Phase 2+ evaluation suite |

### ✅ Scripts (`scripts/`)

| File | Status | Notes |
|------|--------|-------|
| `scripts/train_local.py` | ✅ Implemented | QLoRA training script |
| `scripts/inference.py` | ✅ Implemented | Full inference (single/batch/interactive) |
| `scripts/validate.py` | ✅ Implemented | Smoke tests |
| `scripts/build_index.py` | 📋 Placeholder | Phase 3-4 |
| `scripts/evaluate.py` | 📋 Placeholder | Phase 2+ |
| `scripts/run_rag.py` | 📋 Placeholder | Phase 3+ |
| `scripts/download_data.py` | 📋 Placeholder | Data download utility |

### ✅ Tests (`tests/`)

| File | Status | Notes |
|------|--------|-------|
| `tests/test_data.py` | 📋 Placeholder | Data module tests |
| `tests/test_models.py` | 📋 Placeholder | Model wrapper tests |
| `tests/test_retrieval.py` | 📋 Placeholder | Retrieval tests |
| `tests/test_pipelines.py` | 📋 Placeholder | Pipeline tests |
| `tests/test_utils.py` | 📋 Placeholder | Utility tests |

### ✅ Documentation (`docs/`)

| File | Status | Notes |
|------|--------|-------|
| `docs/architecture_explained.md` | ✅ Complete | Full architecture deep dive |
| `docs/phase1_implementation_explained.md` | ✅ Complete | Phase 1 details |
| `docs/arch_left.png` | ✅ Complete | Architecture diagram |
| `docs/arch_right.png` | ✅ Complete | Architecture diagram |

---

## Files EXCLUDED from Repository

| File/Folder | Reason |
|-------------|--------|
| `data/openi/` | Raw dataset — too large for GitHub (~2 GB images) |
| `data/indexes/` | Pre-built retrieval indexes (generated, large) |
| `checkpoints/` | Trained model adapters (~20+ MB binary files) |
| `outputs/` | Runtime logs, evaluation results, predictions |
| `experiments/results/` | Experiment output files |
| `experiments/logs/` | Experiment log files |
| `.venv/` | Python virtual environment |
| `__pycache__/` | Python bytecode cache |
| `*.egg-info/` | Package metadata |
| `kaggle/` | Kaggle-specific notebook scripts |
| `scripts/quick_test.py` | Internal test script (redundant with inference.py) |
| `scripts/_test_pipeline.py` | Internal dev test (hardcoded paths) |
| `scripts/setup_adapter.py` | Adapter setup utility (Kaggle workflow specific) |
| `*.bin, *.safetensors` | Model weight files (too large) |
| `*.pkl, *.npy, *.faiss` | Serialized data files (generated) |
| `*.log` | Runtime log files |

---

## Implementation Progress

### Phase 1: Simple VQA ✅ Complete

- [x] Project structure and configs
- [x] OpenI dataset loader with CSV parsing
- [x] LLaVA-1.5-7B wrapper with 4-bit quantization
- [x] Model factory for config-driven model loading
- [x] Simple VQA pipeline (Image → Question → Answer)
- [x] QLoRA training script
- [x] Inference script (single, batch, interactive)
- [x] Validation / smoke test script
- [x] GPU device detection and VRAM tracking
- [x] Structured logging

### Phase 2: Evaluation ⬜ Next

- [ ] Implement `src/utils/metrics.py` (BLEU, ROUGE-L, BERTScore, F1)
- [ ] Implement `pipelines/evaluation.py`
- [ ] Implement `scripts/evaluate.py`
- [ ] Run evaluation on Phase 1 outputs

### Phase 3: BM25 Retrieval ⬜ Planned

- [ ] Implement `src/utils/config_loader.py`
- [ ] Implement `src/indexing/document_store.py`
- [ ] Implement `src/retrieval/bm25_retriever.py`
- [ ] Implement `src/context/context_builder.py`
- [ ] Implement `src/context/prompt_templates.py`
- [ ] Build BM25 index and test retrieval
- [ ] Update RAG pipeline with BM25

### Phase 4: Full Multimodal Retrieval ⬜ Planned

- [ ] Implement CLIP embedder and retriever
- [ ] Implement ColQwen2 embedder and retriever
- [ ] Implement RRF hybrid retriever
- [ ] Implement cross-encoder reranker
- [ ] Build full offline index (BM25 + CLIP + ColQwen2)
- [ ] Update RAG pipeline with full retrieval

### Phase 5: Qwen2-VL + Grounding ⬜ Planned

- [ ] Implement Qwen2-VL model wrapper
- [ ] Implement RAG generation engine
- [ ] Implement NLI grounding and safety checks
- [ ] Final evaluation with all components
