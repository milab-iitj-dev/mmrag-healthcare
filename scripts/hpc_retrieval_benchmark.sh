#!/bin/bash
# ============================================================
# HPC SLURM Script — OpenI Retrieval Benchmark
# ============================================================
#
# Runs the retrieval benchmark on the OpenI dataset using the
# current dual-index Hybrid Retrieval system (ColQwen2 + RRF).
#
# Produces:
#   outputs/benchmarks/retrieval/retrieval_YYYYMMDD_HHMMSS.json
#   outputs/benchmarks/retrieval/RETRIEVAL_REPORT.md
#
# Usage:
#   sbatch scripts/hpc_retrieval_benchmark.sh
#
#   Or run interactively (after salloc + ssh to GPU node):
#   bash scripts/hpc_retrieval_benchmark.sh
# ============================================================

#SBATCH --job-name=mrag-retrieval-bench
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8192
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/retrieval_benchmark_%j.log
#SBATCH --error=outputs/logs/retrieval_benchmark_%j.err

set -euo pipefail

# ── Configuration ──
PROJECT_DIR="/scratch/data/divyasaxena_rs/Gokul_Faleja_internship"
INDEX_DIR="data/indexes/colqwen2_index"
QUERIES_PER_MODE=40          # 40 text_only + 40 image_only + 40 hybrid = 120 balanced
OUTPUT_DIR="outputs/benchmarks/retrieval"

echo "============================================================"
echo "  Healthcare MRAG — OpenI Retrieval Benchmark (Balanced)"
echo "  $(date)"
echo "  Mode: ${QUERIES_PER_MODE} queries per mode (balanced)"
echo "============================================================"

# ── 1. Navigate to project directory ──
cd "$PROJECT_DIR"
echo "[1/7] Working directory: $(pwd)"

# ── 2. Activate Python environment ──
if [ -d ".venv" ]; then
    source .venv/bin/activate
    export PATH="$VIRTUAL_ENV/bin:$PATH"
    hash -r
    echo "[2/7] Virtual environment activated: $(which python)"
else
    echo "[2/7] WARNING: No .venv found, using system Python"
fi

# ── 3. Verify GPU and dependencies ──
echo "[3/7] Verifying environment..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('  WARNING: No CUDA GPU detected!')

import yaml
print(f'  PyYAML: OK')

from PIL import Image
print(f'  Pillow: OK')

from colpali_engine.models import ColQwen2, ColQwen2Processor
print(f'  colpali-engine: OK')
"

# ── 4. Verify index exists ──
echo "[4/7] Checking index at $INDEX_DIR..."
if [ -f "$INDEX_DIR/document_store.json" ]; then
    DOC_COUNT=$(python -c "import json; d=json.load(open('$INDEX_DIR/document_store.json')); print(len(d.get('documents', d) if isinstance(d, dict) else d))")
    echo "  Index found: $DOC_COUNT documents"
else
    echo "  ERROR: Index not found at $INDEX_DIR"
    echo "  Make sure the ColQwen2 index is built and available."
    exit 1
fi

# ── 5. Verify OpenI data exists ──
echo "[5/7] Checking OpenI data..."
if [ -f "data/openi/reports/indiana_reports.csv" ]; then
    REPORT_LINES=$(wc -l < data/openi/reports/indiana_reports.csv)
    echo "  Reports CSV: $REPORT_LINES lines"
else
    echo "  ERROR: OpenI reports not found at data/openi/reports/indiana_reports.csv"
    exit 1
fi

# ── 6. Run the retrieval benchmark (balanced) ──
echo ""
echo "[6/7] Running BALANCED retrieval benchmark..."
echo "  Index:           $INDEX_DIR"
echo "  Queries/mode:    $QUERIES_PER_MODE"
echo "  Total queries:   $((QUERIES_PER_MODE * 3)) (text_only + image_only + hybrid)"
echo "  Output:          $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"
mkdir -p "outputs/logs"

python -m evaluation.runners.retrieval_benchmark \
    --retrieval-config configs/retrieval_config.yaml \
    --data-config configs/data_config.yaml \
    --index-dir "$INDEX_DIR" \
    --queries-per-mode "$QUERIES_PER_MODE" \
    --output-dir "$OUTPUT_DIR"

BENCH_EXIT=$?

if [ $BENCH_EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: Benchmark failed with exit code $BENCH_EXIT"
    exit $BENCH_EXIT
fi

# ── 7. Generate markdown report ──
echo ""
echo "[7/7] Generating markdown report..."

python scripts/generate_retrieval_report.py \
    --results-dir "$OUTPUT_DIR" \
    --output "$OUTPUT_DIR/RETRIEVAL_REPORT.md"

# ── Done ──
echo ""
echo "============================================================"
echo "  BENCHMARK COMPLETE"
echo "============================================================"
echo "  JSON results:    $OUTPUT_DIR/retrieval_*.json"
echo "  Markdown report: $OUTPUT_DIR/RETRIEVAL_REPORT.md"
echo "  Timestamp:       $(date)"
echo "============================================================"
