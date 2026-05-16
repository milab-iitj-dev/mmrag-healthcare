"""
Unified configuration loader.

Loads and merges YAML config files into a single dictionary.
All scripts and pipelines use this to load their configuration
instead of reading YAML files directly.

Implementation: Phase 1 (basic) → extended in later phases
"""

# TODO: Implement config loader
# - load_config(): load a single YAML file → dict
# - merge_configs(): merge multiple config dicts (later overrides earlier)
# - validate_config(): check required keys are present
# - Support environment variable overrides (e.g., DATA_DIR=/path)
