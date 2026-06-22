#!/bin/bash
# ============================================================
# HPC SLURM Script — Balanced Ablation Analysis (120 queries)
# ============================================================
#
# Runs the full modality-bias ablation analysis:
#   Baseline (question-ignored) vs Current (dual-index + RRF + reranking)
#   40 text_only + 40 image_only + 40 hybrid = 120 balanced queries
#
# Produces:
#   outputs/observations/observing_document.md
#   outputs/observations/ablation_results.json
#
# Usage:
#   sbatch scripts/hpc_ablation_balanced.sh
#   OR (interactive): bash scripts/hpc_ablation_balanced.sh
# ============================================================

#SBATCH --job-name=mrag-ablation-120
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8192
#SBATCH --time=03:00:00
#SBATCH --output=outputs/logs/ablation_balanced_%j.log
#SBATCH --error=outputs/logs/ablation_balanced_%j.err

set -euo pipefail

# ── Configuration ──
PROJECT_DIR="/scratch/data/divyasaxena_rs/Gokul_Faleja_internship"
INDEX_DIR="data/indexes/colqwen2_index"
QUERIES_PER_MODE=40          # 40 text_only + 40 image_only + 40 hybrid = 120
OUTPUT_DIR="outputs/observations"

echo "============================================================"
echo "  Healthcare MRAG — Ablation Analysis (Balanced, 120 queries)"
echo "  $(date)"
echo "  ${QUERIES_PER_MODE} queries × 3 modes = $((QUERIES_PER_MODE * 3)) total"
echo "============================================================"

# ── 1. Navigate to project directory ──
cd "$PROJECT_DIR"
echo "[1/6] Working directory: $(pwd)"

# ── 2. Activate Python environment ──
if [ -d ".venv" ]; then
    source .venv/bin/activate
    export PATH="$VIRTUAL_ENV/bin:$PATH"
    hash -r
    echo "[2/6] Virtual environment activated: $(which python)"
else
    echo "[2/6] WARNING: No .venv found, using system Python"
fi

# ── 3. Verify GPU ──
echo "[3/6] Verifying environment..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
from colpali_engine.models import ColQwen2, ColQwen2Processor
print(f'  colpali-engine: OK')
"

# ── 4. Verify data ──
echo "[4/6] Checking data..."
if [ ! -f "$INDEX_DIR/document_store.json" ]; then
    echo "  ERROR: Index not found at $INDEX_DIR"
    exit 1
fi
if [ ! -f "data/openi/reports/indiana_reports.csv" ]; then
    echo "  ERROR: OpenI reports not found"
    exit 1
fi
echo "  Index and data OK"

# ── 5. Run ablation ──
echo ""
echo "[5/6] Running ABLATION ANALYSIS..."
echo "  Baseline: question-ignored image+text retrieval"
echo "  Current:  dual-index + RRF + question-aware reranking"
echo "  Queries:  $((QUERIES_PER_MODE * 3)) balanced ($QUERIES_PER_MODE per mode)"
echo ""

mkdir -p "$OUTPUT_DIR"
mkdir -p "outputs/logs"

python -m analysis.ablation.run_ablation \
    --retrieval-config configs/retrieval_config.yaml \
    --data-config configs/data_config.yaml \
    --index-dir "$INDEX_DIR" \
    --queries-per-mode "$QUERIES_PER_MODE" \
    --output-dir "$OUTPUT_DIR"

ABLATION_EXIT=$?

if [ $ABLATION_EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: Ablation failed with exit code $ABLATION_EXIT"
    exit $ABLATION_EXIT
fi

# ── 6. Done ──
echo ""
echo "[6/6] COMPLETE"
echo "============================================================"
echo "  Observing document : $OUTPUT_DIR/observing_document.md"
echo "  Raw JSON results   : $OUTPUT_DIR/ablation_results.json"
echo "  Timestamp          : $(date)"
echo "============================================================"
echo ""
echo "  Quick view:"
echo "  cat $OUTPUT_DIR/observing_document.md"
