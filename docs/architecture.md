# Architecture

## System Overview

```
Query (image + text)
        │
        ▼
┌──────────────────┐
│  ColQwen2 Embed  │  Multi-vector patch embeddings
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  MaxSim Retrieve │  Late-interaction retrieval
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Context Builder │  Format retrieved evidence
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  LLaVA Generate  │  Grounded answer generation
└──────────────────┘
```

## Module Dependency Graph

```
src/ingestion/     → provides data to → src/indexing/  → builds index for → src/retrieval/
src/retrieval/     → feeds docs to    → src/context/   → builds prompt for → src/generation/
src/generation/    → output goes to   → src/evaluation/
```

## Phase 1: Direct VQA
- **Ingestion**: `OpenIDataset` loads chest X-ray images + radiology reports
- **Generation**: `LLaVAModel` generates answers from image + question
- **Pipeline**: `SimpleVQAPipeline` wires ingestion → generation

## Phase 2: RAG-augmented VQA
- **Offline**: `OfflineIndexingPipeline` → `ColQwen2Embedder` → index on disk
- **Online**: `RAGVQAPipeline` → `ColQwen2Retriever` → `ContextBuilder` → `RAGGenerator`
- **Key Innovation**: MaxSim late-interaction retrieval preserves spatial patch information

## Key Design Decisions

1. **BaseVLM abstract class**: All models implement `load()`, `generate()`, `caption()`, `unload()`
2. **BaseDataset abstract class**: All datasets produce `MedicalSample` instances
3. **Model Factory**: Config-driven model instantiation via `MODEL_REGISTRY`
4. **ColQwen2**: Multi-vector output, separate from single-vector `BaseEmbedder` interface
5. **Context budget**: `ContextBuilder` enforces character limits to prevent prompt overflow
