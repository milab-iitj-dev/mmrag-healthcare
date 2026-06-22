"""
Generate Retrieval Benchmark Report — clean markdown from JSON results.

Reads the JSON output produced by `retrieval_benchmark.py` and creates
a professional markdown report with:
    - Aggregate metrics (Recall@k, MRR, nDCG@k, Precision@k)
    - Per query mode breakdown (text_only, image_only, hybrid)
    - Per-query diagnostics table
    - System configuration summary

Usage:
    python scripts/generate_retrieval_report.py \
        --json outputs/benchmarks/retrieval/retrieval_YYYYMMDD_HHMMSS.json \
        --output outputs/benchmarks/retrieval/RETRIEVAL_REPORT.md

    # Or auto-find the latest JSON:
    python scripts/generate_retrieval_report.py \
        --results-dir outputs/benchmarks/retrieval \
        --output outputs/benchmarks/retrieval/RETRIEVAL_REPORT.md
"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional


def _fmt(val, decimals=4) -> str:
    """Format a numeric value for tables."""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def _find_latest_json(directory: Path) -> Optional[Path]:
    """Find the most recent retrieval JSON file."""
    if not directory.exists():
        return None
    files = sorted(directory.glob("retrieval_*.json"), reverse=True)
    return files[0] if files else None


def generate_retrieval_report(
    json_path: str,
    output_path: str = "outputs/benchmarks/retrieval/RETRIEVAL_REPORT.md",
) -> str:
    """
    Generate a markdown report from retrieval benchmark JSON results.

    Args:
        json_path:   Path to the retrieval JSON results file.
        output_path: Path to write the markdown report.

    Returns:
        The generated markdown string.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = data.get("metrics", {})
    agg = metrics.get("aggregate", {})
    per_mode = metrics.get("per_mode", {})
    timing = metrics.get("timing", {})
    config = metrics.get("config", {})
    diagnostics = data.get("diagnostics", [])
    timestamp = data.get("timestamp", time.strftime("%Y%m%d_%H%M%S"))

    lines = []

    # ── Title ──
    lines.append("# OpenI Retrieval Benchmark — Results Report")
    lines.append("")
    lines.append("## Healthcare MRAG — Current System Evaluation")
    lines.append("")
    lines.append(f"*Benchmark run: {timestamp}*  ")
    lines.append(f"*Report generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── System Configuration ──
    lines.append("## System Configuration")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| Retriever | {config.get('retriever', 'HybridRetriever')} |")
    lines.append(f"| Index | {config.get('index_dir', 'data/indexes/colqwen2_index')} |")
    lines.append(f"| Query Modes | {', '.join(config.get('query_modes', []))} |")
    lines.append(f"| Top-K Values | {config.get('top_k_values', [1, 3, 5])} |")
    lines.append(f"| Total Queries | {config.get('num_queries', agg.get('num_queries', '?'))} |")
    lines.append(f"| Total Time | {timing.get('total_seconds', '?')}s |")
    lines.append(f"| Avg per Query | {timing.get('avg_seconds_per_query', '?')}s |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Aggregate Results ──
    lines.append("## Aggregate Results")
    lines.append("")
    lines.append("### Recall@k (Hit@k)")
    lines.append("")
    lines.append(
        "Recall@k = 1 if **any** gold-relevant document appears in the top-k "
        "retrieved results, else 0. Averaged across all queries."
    )
    lines.append("")
    lines.append("| Metric | Score |")
    lines.append("|--------|-------|")
    for k in [1, 3, 5]:
        key = f"recall@{k}"
        lines.append(f"| **Recall@{k}** | **{_fmt(agg.get(key))}** |")
    lines.append("")

    lines.append("### Mean Reciprocal Rank (MRR)")
    lines.append("")
    lines.append("| Metric | Score |")
    lines.append("|--------|-------|")
    lines.append(f"| **MRR** | **{_fmt(agg.get('mrr'))}** |")
    lines.append("")

    lines.append("### nDCG@k")
    lines.append("")
    lines.append("| Metric | Score |")
    lines.append("|--------|-------|")
    for k in [3, 5]:
        key = f"ndcg@{k}"
        if key in agg:
            lines.append(f"| **nDCG@{k}** | **{_fmt(agg.get(key))}** |")
    # Also include nDCG@1 if present
    if "ndcg@1" in agg:
        lines.append(f"| **nDCG@1** | **{_fmt(agg.get('ndcg@1'))}** |")
    lines.append("")

    lines.append("### Precision@k")
    lines.append("")
    lines.append("| Metric | Score |")
    lines.append("|--------|-------|")
    for k in [1, 3, 5]:
        key = f"precision@{k}"
        if key in agg:
            lines.append(f"| Precision@{k} | {_fmt(agg.get(key))} |")
    lines.append("")

    # ── Combined summary table ──
    lines.append("### Summary Table")
    lines.append("")
    lines.append("| Metric | Score |")
    lines.append("|--------|-------|")
    for k in [1, 3, 5]:
        lines.append(f"| Recall@{k} | {_fmt(agg.get(f'recall@{k}'))} |")
    lines.append(f"| MRR | {_fmt(agg.get('mrr'))} |")
    for k in [3, 5]:
        key = f"ndcg@{k}"
        if key in agg:
            lines.append(f"| nDCG@{k} | {_fmt(agg.get(key))} |")
    lines.append(f"| Queries | {agg.get('num_queries', '?')} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Per Query Mode ──
    if per_mode:
        lines.append("## Per Query Mode Breakdown")
        lines.append("")
        lines.append(
            "| Mode | Recall@1 | Recall@3 | Recall@5 | MRR "
            "| nDCG@3 | nDCG@5 | Queries |"
        )
        lines.append(
            "|------|----------|----------|----------|-----"
            "|--------|--------|---------|"
        )
        for mode, m in per_mode.items():
            lines.append(
                f"| {mode} "
                f"| {_fmt(m.get('recall@1'))} "
                f"| {_fmt(m.get('recall@3'))} "
                f"| {_fmt(m.get('recall@5'))} "
                f"| {_fmt(m.get('mrr'))} "
                f"| {_fmt(m.get('ndcg@3'))} "
                f"| {_fmt(m.get('ndcg@5'))} "
                f"| {m.get('num_queries', '?')} |"
            )
        lines.append("")
        lines.append("---")
        lines.append("")

    # ── Per-Query Diagnostics (first 20) ──
    if diagnostics:
        lines.append("## Per-Query Diagnostics (sample)")
        lines.append("")
        lines.append(
            f"Showing first {min(20, len(diagnostics))} of "
            f"{len(diagnostics)} queries."
        )
        lines.append("")
        lines.append(
            "| Query | Finding | Mode | Hit@1 | Hit@3 | Hit@5 | RR | Gold |"
        )
        lines.append(
            "|-------|---------|------|-------|-------|-------|----|------|"
        )
        for diag in diagnostics[:20]:
            q_text = diag.get("query_text", "?")
            if len(q_text) > 35:
                q_text = q_text[:32] + "..."
            lines.append(
                f"| {q_text} "
                f"| {diag.get('finding', '?')} "
                f"| {diag.get('query_mode', '?')} "
                f"| {_fmt(diag.get('hit@1'), 0)} "
                f"| {_fmt(diag.get('hit@3'), 0)} "
                f"| {_fmt(diag.get('hit@5'), 0)} "
                f"| {_fmt(diag.get('reciprocal_rank'))} "
                f"| {diag.get('num_gold_docs', '?')} |"
            )
        lines.append("")
        lines.append("---")
        lines.append("")

    # ── Footer ──
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- **Recall@k** uses binary Hit@k: 1 if any gold document appears "
        "in top-k, else 0. This is the standard RAG retrieval metric."
    )
    lines.append(
        "- **MRR** (Mean Reciprocal Rank): average of 1/rank of the first "
        "relevant document across all queries."
    )
    lines.append(
        "- **nDCG@k** uses binary relevance (1 if gold, 0 otherwise) with "
        "logarithmic discount."
    )
    lines.append(
        "- Gold labels are derived from OpenI MeSH terms, Problems fields, "
        "and non-negated text mentions."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "*Report generated by Healthcare MRAG Retrieval Benchmark Suite*"
    )
    lines.append("")

    # ── Write ──
    report_text = "\n".join(lines)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nReport saved to: {out}")
    return report_text


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(
        description="Generate markdown report from retrieval benchmark JSON"
    )
    parser.add_argument(
        "--json",
        default=None,
        help="Path to a specific retrieval JSON results file",
    )
    parser.add_argument(
        "--results-dir",
        default="outputs/benchmarks/retrieval",
        help="Directory to search for the latest retrieval JSON",
    )
    parser.add_argument(
        "--output",
        default="outputs/benchmarks/retrieval/RETRIEVAL_REPORT.md",
        help="Path for the output markdown report",
    )
    args = parser.parse_args()

    # Find JSON path
    if args.json:
        json_path = args.json
    else:
        latest = _find_latest_json(Path(args.results_dir))
        if latest is None:
            print(
                f"ERROR: No retrieval JSON found in {args.results_dir}. "
                f"Run the benchmark first:\n"
                f"  python -m evaluation.runners.retrieval_benchmark"
            )
            return
        json_path = str(latest)
        print(f"Using latest results: {json_path}")

    report = generate_retrieval_report(
        json_path=json_path,
        output_path=args.output,
    )

    # Print preview
    print("\n" + "=" * 60)
    print("REPORT PREVIEW")
    print("=" * 60)
    for line in report.split("\n")[:50]:
        print(line)
    print("...")
    print("=" * 60)


if __name__ == "__main__":
    main()
