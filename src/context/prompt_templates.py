"""
Prompt templates for different generation scenarios.

Stores structured prompt templates for:
    - Zero-shot VQA (Phase 1)
    - RAG-augmented VQA (Phase 3+)
    - Medical captioning
    - Grounded reasoning

Templates use Python format strings so callers can inject
the question, context, and image descriptions.

Implementation: Phase 1 (basic) → Phase 3+ (RAG-augmented)
"""

# TODO: Define prompt templates
# - SIMPLE_VQA_PROMPT: image + question → answer (Phase 1)
# - RAG_VQA_PROMPT: image + question + retrieved context → answer
# - CAPTION_PROMPT: image → clinical caption
# - GROUNDED_PROMPT: image + question + evidence → answer with citations
