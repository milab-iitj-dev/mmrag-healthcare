"""
Prompt templates for different generation scenarios.

Stores structured prompt templates for:
    - Zero-shot VQA (Phase 1)
    - RAG-augmented VQA (Phase 2)
    - Medical captioning
    - Grounded reasoning (future phases)

Templates use Python format strings so callers can inject
the question, context, and image descriptions.

Design notes:
  - LLaVA-1.5 uses the Vicuna conversation format:
      USER: <image>\n{instruction}\nASSISTANT:
  - The <image> token is handled by LLaVA's _build_prompt(),
    so these templates provide only the instruction text.
  - The RAG template injects retrieved evidence before the question.
"""

# ------------------------------------------------------------------ #
#  Phase 1: Simple VQA (no retrieval)                                  #
# ------------------------------------------------------------------ #

SIMPLE_VQA_PROMPT = (
    "{question}"
)
"""
Simple VQA prompt — direct question to VLM with image.

Used by LLaVA's _build_prompt() which wraps this as:
    USER: <image>\n{question}\nASSISTANT:

Placeholders:
    {question} — the clinical question
"""

# ------------------------------------------------------------------ #
#  Phase 2: RAG-augmented VQA                                          #
# ------------------------------------------------------------------ #

RAG_VQA_PROMPT = (
    "You are a medical imaging specialist. Use the following retrieved "
    "clinical evidence from similar cases to help answer the question.\n"
    "\n"
    "{context}\n"
    "\n"
    "Based on the image and the retrieved evidence above, "
    "answer the following question:\n"
    "{question}\n"
    "\n"
    "Provide a detailed, clinically relevant answer. Reference the "
    "retrieved evidence where applicable."
)
"""
RAG-augmented VQA prompt — question + retrieved evidence.

Used when the retrieval pipeline provides context from similar cases.
The context block contains formatted evidence from the top-k retrieved
documents (findings, impressions, metadata).

Placeholders:
    {context}  — formatted retrieved evidence block
    {question} — the clinical question
"""

# ------------------------------------------------------------------ #
#  Medical captioning                                                  #
# ------------------------------------------------------------------ #

CAPTION_PROMPT = (
    "Describe all clinically significant findings visible in this "
    "medical image. Include observations about anatomy, pathology, "
    "and any abnormalities."
)
"""
Medical image captioning prompt.

No placeholders — used directly for generating clinical descriptions
of medical images (e.g., chest X-rays).
"""

# ------------------------------------------------------------------ #
#  RAG captioning (with context)                                       #
# ------------------------------------------------------------------ #

RAG_CAPTION_PROMPT = (
    "You are a medical imaging specialist. Use the following retrieved "
    "clinical evidence from similar cases to help describe this image.\n"
    "\n"
    "{context}\n"
    "\n"
    "Based on the image and the evidence above, describe all clinically "
    "significant findings visible in this medical image."
)
"""
RAG-augmented captioning prompt.

Placeholders:
    {context} — formatted retrieved evidence block
"""

# ------------------------------------------------------------------ #
#  Future: Grounded reasoning (Phase 5)                                #
# ------------------------------------------------------------------ #

# GROUNDED_PROMPT = (
#     "You are a medical imaging specialist. Use the following evidence "
#     "to answer the question. For each claim in your answer, cite the "
#     "evidence source (e.g., [Evidence #1]).\n"
#     "\n"
#     "{context}\n"
#     "\n"
#     "Question: {question}\n"
#     "\n"
#     "Provide a grounded answer with citations."
# )
