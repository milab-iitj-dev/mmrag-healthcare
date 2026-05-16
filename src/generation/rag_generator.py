"""
RAG-augmented generation engine.

Orchestrates the full generation pipeline:
    1. Receive question + image
    2. (Phase 3+) Retrieve relevant evidence
    3. Build context from retrieved documents
    4. Call VLM.generate() with assembled prompt
    5. (Phase 5) Verify answer with NLI grounding

This is the main generation entry point for RAG-enabled queries.

Implementation: Phase 3+
"""

# TODO: Implement RAG generation engine
# - generate(): full RAG pipeline (retrieve → build context → generate → verify)
# - generate_simple(): direct VQA without retrieval (Phase 1, already in pipeline)
# - Support configurable retrieval depth and context window
