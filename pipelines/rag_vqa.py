"""
RAG-augmented VQA Pipeline — Phase 3+

End-to-end pipeline:
    1. Load query image + question
    2. Retrieve relevant documents from the knowledge base
    3. (Optional) Rerank retrieved candidates
    4. Build context from top-k evidence
    5. Generate answer using VLM with retrieved context
    6. (Optional) Verify grounding with NLI

This pipeline extends simple_vqa.py by adding retrieval and context.

Implementation: Phase 3+
"""

# TODO: Implement RAG pipeline
# - class RAGPipeline:
#     - __init__(): load model, retriever, reranker, context builder
#     - run_single(): process one (image, question) pair with RAG
#     - run_batch(): process a batch of queries
#     - evaluate(): run on test set and compute metrics
