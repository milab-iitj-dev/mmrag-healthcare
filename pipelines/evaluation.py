"""
Evaluation Pipeline — Phase 2+

Runs the full evaluation suite:
    1. Load model + test dataset
    2. Generate predictions on all test samples
    3. Compute generation metrics (BLEU, ROUGE, BERTScore, etc.)
    4. (Phase 3+) Compute retrieval metrics (Recall@k, MRR, NDCG)
    5. Save per-sample results and aggregate summary
    6. Generate visualizations

Implementation: Phase 2+
"""

# TODO: Implement evaluation pipeline
# - class EvaluationPipeline:
#     - __init__(): load model, dataset, metrics config
#     - evaluate_generation(): compute generation metrics
#     - evaluate_retrieval(): compute retrieval metrics (Phase 3+)
#     - save_results(): persist results to outputs/evaluation/
#     - generate_report(): create a markdown summary report
