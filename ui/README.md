# Healthcare MRAG — Inference UI

Professional browser-based inference interface for the Healthcare Multimodal RAG system.

## Quick Start

```bash
# Full mode (requires GPU + pre-built index)
python scripts/launch_ui.py

# UI preview only (no GPU needed)
python scripts/launch_ui.py --ui-only

# With custom paths
python scripts/launch_ui.py \
    --model-config configs/model_config.yaml \
    --retrieval-config configs/retrieval_config.yaml \
    --index-dir data/indexes/colqwen2_index
```

Then open **http://localhost:7860** in your browser.

---

## Usage

1. **Upload** a chest X-ray image (PNG, JPEG, or DICOM-converted)
2. **Type** a clinical question (or leave blank for a default question)
3. Click **Analyze**
4. View the clinical answer, retrieved evidence, and timing

The system automatically detects the query mode:

| You Provide          | Mode Detected   | What Happens                                   |
|----------------------|-----------------|------------------------------------------------|
| Image + Question     | Image + Text    | Full multimodal retrieval and generation       |
| Image only           | Image Only      | Default question applied, full pipeline runs   |
| Question only        | Text Only       | Text retrieval, best retrieved image used       |

---

## HPC / DGX Deployment

```bash
# SSH tunnel mode
python scripts/launch_ui.py --server-name 0.0.0.0 --port 7860

# Then on your local machine:
ssh -L 7860:compute-node:7860 user@hpc-login

# Gradio share mode (easiest)
python scripts/launch_ui.py --share

# SLURM
srun --gres=gpu:1 --mem=32G --time=04:00:00 \
    python scripts/launch_ui.py --share
```

---

## CLI Options

| Option               | Default                            | Description                       |
|----------------------|------------------------------------|-----------------------------------|
| `--model-config`     | `configs/model_config.yaml`        | Model configuration YAML          |
| `--retrieval-config` | `configs/retrieval_config.yaml`    | Retrieval configuration YAML      |
| `--index-dir`        | `data/indexes/colqwen2_index`      | Pre-built ColQwen2 index path     |
| `--top-k`            | `3`                                | Number of cases to retrieve       |
| `--port`             | `7860`                             | Server port                       |
| `--server-name`      | `127.0.0.1`                        | Bind address                      |
| `--share`            | off                                | Create public Gradio URL          |
| `--ui-only`          | off                                | Preview layout without models     |

---

## Architecture

```
Browser ─── Gradio Server (ui/app.py)
                │
                ├── inference_fn()
                │       │
                │       └── pipeline.run_single(query, image)
                │               │
                │               ├── ColQwen2Retriever.retrieve()
                │               ├── ContextBuilder.build_context()
                │               └── LLaVAModel.generate()
                │
                └── formatters.py → HTML output
```

The UI is a **thin wrapper** — all intelligence lives in `src/` and `pipelines/`.

---

## File Structure

```
ui/
├── __init__.py       # Package marker
├── app.py            # Gradio Blocks application
├── theme.py          # Custom dark theme + CSS
├── formatters.py     # RAGOutput → HTML formatters
└── README.md         # This file

scripts/
└── launch_ui.py      # CLI entry point
```
