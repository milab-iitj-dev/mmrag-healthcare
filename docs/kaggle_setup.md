# Kaggle Setup Guide

## GPU Requirements
- **Phase 1 (VQA only)**: T4 16GB sufficient
- **Phase 2 (VQA + Retrieval)**: T4 16GB (sequential model loading)

## VRAM Management
When running both ColQwen2 and LLaVA on a single GPU:
1. Load ColQwen2, build/load index
2. Unload ColQwen2 (`embedder.unload()`)
3. Load LLaVA for generation
4. Peak VRAM stays under 16GB

## Setup Steps

```python
!pip install -q transformers>=4.46.0 accelerate peft bitsandbytes colpali-engine

import sys
sys.path.insert(0, "/kaggle/input/healthcare-mrag/")

from src.generation.model_factory import create_model
from src.ingestion.dicom_loader import OpenIDataset
```
