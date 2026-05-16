"""
Context builder for assembling retrieved evidence into VLM prompts.

Takes retrieval results (documents, images, scores) and builds a
structured context string that the VLM can use to generate grounded,
evidence-based answers. Controls how much context to include,
de-duplicates overlapping information, and formats it cleanly.

Implementation: Phase 3-4
"""

# TODO: Implement context builder
# - build_context(): assemble top-k retrieved docs into a prompt section
# - format_evidence(): format a single evidence item (report extract + metadata)
# - truncate_context(): ensure total context fits within VLM token limits
# - Support multimodal context (text evidence + reference images)
