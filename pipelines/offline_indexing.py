"""
Offline Indexing Pipeline — Phase 2

Runs the complete offline knowledge-base preparation:
    1. Load OpenI dataset (image-report pairs)
    2. Initialize ColQwen2 embedder
    3. Build document store from dataset samples
    4. Encode all images with ColQwen2
    5. Save index to disk

Run this ONCE before using the RAG pipeline.

Usage:
    python -m pipelines.offline_indexing
    python -m pipelines.offline_indexing --max-samples 50
    python -m pipelines.offline_indexing --data-config configs/data_config.yaml
"""

import json
import time
from pathlib import Path
from typing import Optional

from src.ingestion.dicom_loader import OpenIDataset
from src.embeddings.colqwen2_embedder import ColQwen2Embedder
from src.indexing.index_builder import ColQwen2IndexBuilder
from src.utils.logging_utils import setup_logger

logger = setup_logger("pipeline.offline_indexing")


class OfflineIndexingPipeline:
    """
    Phase 2 offline pipeline: OpenI → ColQwen2 index.

    Processes all OpenI image-report pairs, encodes each image with
    ColQwen2, and saves a persistent retrieval index to disk.

    The saved index is later loaded by the RAG VQA pipeline for
    online retrieval.
    """

    def __init__(
        self,
        data_config: dict,
        retrieval_config: dict,
        index_dir: str = "data/indexes/colqwen2_index",
    ):
        """
        Args:
            data_config:      Dataset configuration dict.
            retrieval_config: Retrieval configuration dict.
            index_dir:        Directory to save the built index.
        """
        self.data_config = data_config
        self.retrieval_config = retrieval_config
        self.index_dir = index_dir

    def run(self, max_samples: Optional[int] = None) -> dict:
        """
        Execute the full offline indexing pipeline.

        Args:
            max_samples: Cap on number of samples to index (None = all).

        Returns:
            Summary dict with build statistics.
        """
        total_start = time.time()

        logger.info("=" * 60)
        logger.info("Phase 2: Offline Indexing Pipeline")
        logger.info("=" * 60)

        # Step 1: Load OpenI dataset
        logger.info("\n--- Step 1: Loading OpenI dataset ---")
        dataset = self._load_dataset(max_samples)
        logger.info(f"Dataset loaded: {len(dataset)} samples")
        logger.info(f"Dataset summary: {dataset.summary()}")

        # Step 2: Initialize ColQwen2 embedder
        logger.info("\n--- Step 2: Loading ColQwen2 embedder ---")
        embedder = ColQwen2Embedder()
        embedder.load(self.retrieval_config)
        logger.info("ColQwen2 embedder ready")

        # Step 3: Build index
        logger.info("\n--- Step 3: Building ColQwen2 index ---")
        builder = ColQwen2IndexBuilder(
            embedder=embedder,
            config=self.retrieval_config,
        )

        batch_size = (
            self.retrieval_config
            .get("retrieval", {})
            .get("colqwen2", {})
            .get("batch_size", 4)
        )

        builder.build_from_dataset(
            dataset=dataset,
            max_samples=max_samples,
            batch_size=batch_size,
        )

        # Step 4: Save index to disk
        logger.info("\n--- Step 4: Saving index ---")
        builder.save(self.index_dir)

        # Step 5: Cleanup — free VRAM
        logger.info("\n--- Step 5: Cleanup ---")
        embedder.unload()

        total_time = time.time() - total_start

        # Summary
        summary = builder.summary()
        summary["total_time_seconds"] = round(total_time, 2)
        summary["index_dir"] = self.index_dir

        logger.info("\n" + "=" * 60)
        logger.info("Offline Indexing Complete")
        logger.info(f"  Documents indexed: {summary['num_indexed']}")
        logger.info(f"  Index saved to:    {self.index_dir}")
        logger.info(f"  Total time:        {total_time:.1f}s")
        logger.info("=" * 60)

        return summary

    def _load_dataset(self, max_samples: Optional[int] = None) -> OpenIDataset:
        """Load and return the OpenI dataset."""
        ds_cfg = self.data_config.get("dataset", {})

        dataset = OpenIDataset(
            images_dir=ds_cfg.get("images_dir", "data/openi/images"),
            reports_dir=ds_cfg.get("reports_dir", "data/openi/reports"),
            max_samples=max_samples or ds_cfg.get("max_samples"),
            load_images=False,   # Images loaded on-demand during indexing
        )
        dataset.load()
        return dataset


# ------------------------------------------------------------------ #
#  CLI entry point                                                     #
# ------------------------------------------------------------------ #

def main():
    """Run the offline indexing pipeline from command line."""
    import argparse
    import yaml

    parser = argparse.ArgumentParser(
        description="Phase 2: Offline ColQwen2 Indexing Pipeline"
    )
    parser.add_argument(
        "--data-config",
        default="configs/data_config.yaml",
        help="Path to data config YAML",
    )
    parser.add_argument(
        "--retrieval-config",
        default="configs/retrieval_config.yaml",
        help="Path to retrieval config YAML",
    )
    parser.add_argument(
        "--index-dir",
        default="data/indexes/colqwen2_index",
        help="Directory to save the index",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to index (None = all)",
    )
    args = parser.parse_args()

    # Load configs
    with open(args.data_config) as f:
        data_config = yaml.safe_load(f)
    with open(args.retrieval_config) as f:
        retrieval_config = yaml.safe_load(f)

    # Resolve relative data paths to project root
    from src.utils.config_loader import resolve_data_paths
    data_config = resolve_data_paths(data_config)

    # Run pipeline
    pipeline = OfflineIndexingPipeline(
        data_config=data_config,
        retrieval_config=retrieval_config,
        index_dir=args.index_dir,
    )

    summary = pipeline.run(max_samples=args.max_samples)

    # Save summary
    summary_path = Path(args.index_dir) / "build_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Build summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
