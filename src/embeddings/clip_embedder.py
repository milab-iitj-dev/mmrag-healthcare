"""
CLIP / BiomedCLIP embedding model.

Encodes images and text into a shared embedding space using CLIP.
Used for dense retrieval (cosine similarity search) and for building
the FAISS visual index.

Implementation: Phase 4
"""

# TODO: Implement CLIP embedder
# - load(): load CLIP or BiomedCLIP model + processor
# - encode_text(): batch-encode text with CLIP text encoder
# - encode_image(): batch-encode images with CLIP vision encoder
# - Normalise all embeddings to unit length for cosine similarity
