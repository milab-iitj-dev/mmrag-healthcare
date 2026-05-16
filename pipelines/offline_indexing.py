"""
Offline Indexing Pipeline — Phase 3-4

Runs the complete offline knowledge-base preparation:
    1. Load all dataset samples (OpenI reports + images)
    2. Generate captions for images (using LLaVA / Qwen2-VL)
    3. Build BM25 text index over reports
    4. Compute CLIP embeddings for all images
    5. Compute ColQwen2 embeddings for all document pages
    6. Build FAISS indexes for dense retrieval
    7. Save all indexes and the document store to disk

Run this ONCE before using the RAG pipeline.

Implementation: Phase 3-4
"""

# TODO: Implement offline indexing pipeline
# - class OfflineIndexingPipeline:
#     - __init__(): load configs, embedders, dataset
#     - build_text_index(): BM25 over report text
#     - build_visual_index(): CLIP/ColQwen2 embeddings + FAISS
#     - generate_captions(): caption all images with VLM
#     - run(): orchestrate full offline pipeline
