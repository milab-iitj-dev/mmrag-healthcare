"""
Offline index builder for the medical knowledge base.

Orchestrates the full offline indexing pipeline:
    1. Load OpenI dataset (image-report pairs)
    2. Build document store from dataset samples
    3. Encode all document images with ColQwen2
    4. Save multi-vector embeddings + document store to disk

This runs ONCE (or on dataset updates), not at query time.
The saved index is loaded by the ColQwen2Retriever at query time.

Index format on disk:
    data/indexes/colqwen2_index/
    ├── document_store.json        # all document metadata
    ├── embeddings.pt              # list of multi-vector tensors
    ├── doc_ids.json               # ordered list of doc IDs matching embeddings
    └── index_metadata.json        # build info (timestamp, model, count)
"""

import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
from tqdm import tqdm

from src.indexing.document_store import DocumentStore, Document
from src.embeddings.colqwen2_embedder import ColQwen2Embedder
from src.ingestion.base_loader import BaseDataset
from src.utils.logging_utils import setup_logger
from src.utils.image_utils import load_image

logger = setup_logger("indexing.builder")


class ColQwen2IndexBuilder:
    """
    Offline index builder for ColQwen2-based retrieval.

    Takes an OpenI dataset, encodes all images with ColQwen2, and
    saves a persistent index that can be loaded at query time.

    Usage:
        builder = ColQwen2IndexBuilder(embedder, config)
        builder.build_from_dataset(dataset)
        builder.save("data/indexes/colqwen2_index/")
    """

    def __init__(
        self,
        embedder: ColQwen2Embedder,
        config: Optional[dict] = None,
    ):
        """
        Args:
            embedder: Loaded ColQwen2Embedder instance.
            config:   Optional config dict for index builder settings.
        """
        self.embedder = embedder
        self.config = config or {}
        self.document_store = DocumentStore()
        self._embeddings: List[torch.Tensor] = []
        self._doc_ids: List[str] = []
        self._build_metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    #  Build index from dataset                                            #
    # ------------------------------------------------------------------ #

    def build_from_dataset(
        self,
        dataset: BaseDataset,
        max_samples: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        """
        Build the ColQwen2 index from an OpenI dataset.

        Steps:
            1. Iterate dataset samples → build document store
            2. Load images for all documents
            3. Encode images with ColQwen2 in batches
            4. Store embeddings + metadata

        Args:
            dataset:     Loaded OpenI dataset.
            max_samples: Cap on number of samples to index (None = all).
            batch_size:  Override embedder batch size.
        """
        start_time = time.time()
        n_samples = len(dataset)
        if max_samples is not None:
            n_samples = min(n_samples, max_samples)

        logger.info(f"Building ColQwen2 index from {n_samples} samples")

        # Step 1: Build document store and collect images
        images = []
        doc_ids = []
        skipped = 0

        for idx in tqdm(range(n_samples), desc="Loading documents"):
            sample = dataset[idx]

            # Skip samples without usable content
            if not sample.image_path:
                skipped += 1
                continue

            # Try loading the image
            try:
                image = load_image(sample.image_path)
            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"Skipping {sample.sample_id}: {e}")
                skipped += 1
                continue

            # Create document entry
            doc = Document(
                doc_id=sample.sample_id,
                text=sample.report or "",
                image_path=sample.image_path,
                findings=sample.findings,
                impression=sample.impression,
                metadata=sample.metadata,
            )
            self.document_store.add_document(doc)

            images.append(image)
            doc_ids.append(sample.sample_id)

        logger.info(
            f"Document store built: {len(doc_ids)} documents "
            f"({skipped} skipped)"
        )

        # Step 2: Encode all images with ColQwen2
        logger.info("Encoding images with ColQwen2...")
        self._embeddings = self.embedder.encode_images(
            images,
            batch_size=batch_size,
        )
        self._doc_ids = doc_ids

        # Step 3: Store build metadata
        elapsed = time.time() - start_time
        self._build_metadata = {
            "build_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": self.embedder.model_name,
            "num_documents": len(doc_ids),
            "num_skipped": skipped,
            "build_time_seconds": round(elapsed, 2),
            "embedding_shapes": [
                list(emb.shape) for emb in self._embeddings
            ],
        }

        logger.info(
            f"Index built: {len(self._embeddings)} embeddings "
            f"in {elapsed:.1f}s"
        )

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, index_dir: str) -> None:
        """
        Save the complete index to disk.

        Creates:
            index_dir/
            ├── document_store.json
            ├── embeddings.pt
            ├── doc_ids.json
            └── index_metadata.json

        Args:
            index_dir: Directory to save the index in.
        """
        index_path = Path(index_dir)
        index_path.mkdir(parents=True, exist_ok=True)

        # Save document store
        docstore_path = index_path / "document_store.json"
        self.document_store.save(str(docstore_path))

        # Save embeddings as a list of tensors
        embeddings_path = index_path / "embeddings.pt"
        torch.save(self._embeddings, str(embeddings_path))
        logger.info(f"Embeddings saved: {len(self._embeddings)} tensors → {embeddings_path}")

        # Save ordered doc IDs (maps embedding index → doc_id)
        doc_ids_path = index_path / "doc_ids.json"
        with open(doc_ids_path, "w", encoding="utf-8") as f:
            json.dump(self._doc_ids, f, indent=2)
        logger.info(f"Doc IDs saved: {len(self._doc_ids)} → {doc_ids_path}")

        # Save build metadata
        metadata_path = index_path / "index_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._build_metadata, f, indent=2)
        logger.info(f"Index metadata saved → {metadata_path}")

        logger.info(f"Complete index saved to: {index_path}")

    def load(self, index_dir: str) -> None:
        """
        Load a previously saved index from disk.

        Args:
            index_dir: Directory containing the saved index.

        Raises:
            FileNotFoundError: If the index directory or required files are missing.
        """
        index_path = Path(index_dir)
        if not index_path.exists():
            raise FileNotFoundError(f"Index directory not found: {index_path}")

        # Load document store
        docstore_path = index_path / "document_store.json"
        self.document_store.load(str(docstore_path))

        # Load embeddings
        embeddings_path = index_path / "embeddings.pt"
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
        self._embeddings = torch.load(str(embeddings_path), map_location="cpu")
        logger.info(f"Embeddings loaded: {len(self._embeddings)} tensors")

        # Load doc IDs
        doc_ids_path = index_path / "doc_ids.json"
        if not doc_ids_path.exists():
            raise FileNotFoundError(f"Doc IDs file not found: {doc_ids_path}")
        with open(doc_ids_path, "r", encoding="utf-8") as f:
            self._doc_ids = json.load(f)
        logger.info(f"Doc IDs loaded: {len(self._doc_ids)}")

        # Load build metadata
        metadata_path = index_path / "index_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                self._build_metadata = json.load(f)

        logger.info(
            f"Index loaded from {index_path}: "
            f"{len(self._embeddings)} embeddings, "
            f"{len(self.document_store)} documents"
        )

    # ------------------------------------------------------------------ #
    #  Accessors                                                           #
    # ------------------------------------------------------------------ #

    @property
    def embeddings(self) -> List[torch.Tensor]:
        """All stored document embeddings."""
        return self._embeddings

    @property
    def doc_ids(self) -> List[str]:
        """Ordered list of doc IDs (maps index → doc_id)."""
        return self._doc_ids

    @property
    def num_documents(self) -> int:
        """Number of indexed documents."""
        return len(self._doc_ids)

    @property
    def build_metadata(self) -> Dict[str, Any]:
        """Metadata from the last build."""
        return self._build_metadata

    def summary(self) -> Dict[str, Any]:
        """Summary of the current index state."""
        return {
            "num_indexed": len(self._embeddings),
            "num_documents": len(self.document_store),
            "doc_ids_match": len(self._embeddings) == len(self._doc_ids),
            "build_metadata": self._build_metadata,
            "document_store_summary": self.document_store.summary(),
        }
