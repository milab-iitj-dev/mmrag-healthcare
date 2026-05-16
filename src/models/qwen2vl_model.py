"""
Qwen2-VL model wrapper.

Wraps the Qwen2-VL-7B-Instruct model as a drop-in replacement for LLaVA.
Implements the same BaseVLM interface, so swapping models is a config change.

Qwen2-VL is the final generation model (Phase 5) — it has stronger
multimodal reasoning and native support for interleaved image-text inputs.

Implementation: Phase 5
"""

# TODO: Implement Qwen2-VL wrapper
# - load(): load Qwen2-VL with quantisation config
# - generate(): format input as Qwen2-VL chat template, generate answer
# - caption(): generate clinical caption for an image
# - Support multi-image inputs (query image + retrieved evidence images)
