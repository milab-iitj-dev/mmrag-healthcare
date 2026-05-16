"""
Offline index builder for the medical knowledge base.

Orchestrates the full offline pipeline:
    1. Load dataset (OpenI reports + images)
    2. Generate captions for images (via LLaVA or Qwen2-VL)
    3. Compute embeddings (CLIP, ColQwen2)
    4. Build BM25 inverted index over report text
    5. Build FAISS vector index over embeddings
    6. Save all indexes to disk

This runs ONCE (or on dataset updates), not at query time.

Implementation: Phase 3-4
"""

# TODO: Implement offline index building pipeline
# - build_text_index(): tokenise reports, build BM25 index
# - build_visual_index(): encode images with CLIP/ColQwen2, build FAISS
# - build_all(): orchestrate full offline pipeline
# - save() / load(): persist and restore all indexes
