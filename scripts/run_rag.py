"""
Script: RAG Inference — Phase 3+

Entry point for RAG-augmented inference (retrieval + generation).

Usage:
    python scripts/run_rag.py --image path/to/xray.png --question "What does this show?"
    python scripts/run_rag.py --batch --max-samples 10
    python scripts/run_rag.py --interactive
"""

# TODO: Implement RAG inference script
# - Parse CLI args (image path, question, batch mode, interactive mode)
# - Load model, retriever, knowledge base index
# - Instantiate RAGPipeline
# - Run single query, batch, or interactive loop
# - Display results with retrieved evidence
