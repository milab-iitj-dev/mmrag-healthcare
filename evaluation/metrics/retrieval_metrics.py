"""
Retrieval Metrics — Recall@k, MRR, nDCG.

Evaluates retrieval quality by comparing retrieved document IDs
against gold-standard relevant document IDs.

All metrics operate on ranked lists and are standard IR metrics.
"""

import math
from typing import List, Dict, Any


def recall_at_k(
    retrieved_ids: List[str],
    gold_ids: List[str],
    k: int,
) -> float:
    """
    Recall@k: fraction of gold documents found in top-k retrieved.

    Args:
        retrieved_ids: Ranked list of retrieved doc IDs (best first).
        gold_ids:      Set of relevant document IDs.
        k:             Cutoff rank.

    Returns:
        Recall score in [0.0, 1.0].
    """
    if not gold_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    gold_set = set(gold_ids)
    return len(top_k & gold_set) / len(gold_set)


def reciprocal_rank(
    retrieved_ids: List[str],
    gold_ids: List[str],
) -> float:
    """
    Reciprocal Rank: 1/rank of the first relevant document.

    Returns 0.0 if no relevant document is found.
    """
    gold_set = set(gold_ids)
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in gold_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(
    retrieved_ids: List[str],
    gold_ids: List[str],
    k: int,
) -> float:
    """
    Normalized Discounted Cumulative Gain at k.

    Uses binary relevance: 1 if doc is in gold set, 0 otherwise.
    """
    gold_set = set(gold_ids)

    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        rel = 1.0 if doc_id in gold_set else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1)=0

    # Ideal DCG (all relevant docs at the top)
    ideal_rels = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_rels))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_retrieval_metrics(
    results: List[Dict[str, Any]],
    k_values: List[int] = None,
) -> Dict[str, Any]:
    """
    Compute aggregate retrieval metrics over a list of query results.

    Args:
        results: List of dicts, each with:
            - "retrieved_ids": List[str]
            - "gold_ids": List[str]
            - "query_mode": str (optional, for grouping)
        k_values: List of k values for Recall@k and nDCG@k.

    Returns:
        Dict with aggregate metrics and per-mode breakdowns.
    """
    if k_values is None:
        k_values = [1, 3, 5]

    # Aggregate
    all_recalls = {k: [] for k in k_values}
    all_rr = []
    all_ndcg = {k: [] for k in k_values}

    # Per query mode
    mode_results: Dict[str, Dict] = {}

    for r in results:
        retrieved = r["retrieved_ids"]
        gold = r["gold_ids"]
        mode = r.get("query_mode", "all")

        rr = reciprocal_rank(retrieved, gold)
        all_rr.append(rr)

        for k in k_values:
            rec = recall_at_k(retrieved, gold, k)
            ndcg = ndcg_at_k(retrieved, gold, k)
            all_recalls[k].append(rec)
            all_ndcg[k].append(ndcg)

        # Per-mode accumulation
        if mode not in mode_results:
            mode_results[mode] = {
                "rr": [], **{f"recall@{k}": [] for k in k_values},
                **{f"ndcg@{k}": [] for k in k_values},
            }
        mode_results[mode]["rr"].append(rr)
        for k in k_values:
            mode_results[mode][f"recall@{k}"].append(
                recall_at_k(retrieved, gold, k)
            )
            mode_results[mode][f"ndcg@{k}"].append(
                ndcg_at_k(retrieved, gold, k)
            )

    def _mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    # Aggregate metrics
    aggregate = {
        "mrr": _mean(all_rr),
        "num_queries": len(results),
    }
    for k in k_values:
        aggregate[f"recall@{k}"] = _mean(all_recalls[k])
        aggregate[f"ndcg@{k}"] = _mean(all_ndcg[k])

    # Per-mode metrics
    per_mode = {}
    for mode, data in mode_results.items():
        per_mode[mode] = {
            "mrr": _mean(data["rr"]),
            "num_queries": len(data["rr"]),
        }
        for k in k_values:
            per_mode[mode][f"recall@{k}"] = _mean(data[f"recall@{k}"])
            per_mode[mode][f"ndcg@{k}"] = _mean(data[f"ndcg@{k}"])

    return {
        "aggregate": aggregate,
        "per_mode": per_mode,
    }
