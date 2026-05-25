# Healthcare Multimodal RAG

**Multimodal Retrieval-Augmented Generation for Medical Visual Question Answering**

A modular system for medical image understanding that combines vision-language models with retrieval-augmented generation. Built on LLaVA-1.5-7B and ColQwen2 for chest X-ray analysis using the OpenI dataset.

**Lab**: Machine Intelligence Lab · IIT Jodhpur  
**Author**: Gokul  
**License**: Apache 2.0

---

## Features

### Phase 1 — Direct VQA
- LLaVA-1.5-7B with 4-bit quantization (QLoRA-ready)
- Direct image → question → answer pipeline
- OpenI chest X-ray dataset support
- Local training, inference, and validation scripts

### Phase 2 — RAG-Augmented VQA
- ColQwen2 multi-vector embeddings with MaxSim retrieval
- Offline index building for the medical knowledge base
- Context-aware generation with retrieved evidence
- Full RAG pipeline: retrieve → context → generate

---

## Project Structure

```
├── src/
│   ├── ingestion/       # Data loading (OpenI chest X-rays)
│   ├── embeddings/      # ColQwen2 multi-vector embeddings
│   ├── indexing/        # Document store and index building
│   ├── retrieval/       # ColQwen2 MaxSim retrieval
│   ├── context/         # Context building from retrieved evidence
│   ├── generation/      # LLaVA VLM wrappers and RAG generator
│   ├── evaluation/      # Metrics and evaluation runner
│   └── utils/           # Device, logging, image utilities
├── pipelines/           # End-to-end orchestrators
├── scripts/             # Training, inference, validation CLI
├── configs/             # YAML configuration files
├── docs/                # Architecture and domain documentation
└── tests/               # Test suite
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/<your-org>/healthcare-mrag.git
cd healthcare-mrag
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
