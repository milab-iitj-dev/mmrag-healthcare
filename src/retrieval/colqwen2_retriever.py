"""
ColQwen2 late-interaction retrieval for multimodal documents.

Uses ColQwen2 (ColPali architecture with Qwen2-VL backbone) for
late-interaction retrieval. Unlike CLIP's single-vector matching,
ColQwen2 produces per-token embeddings and uses MaxSim scoring
for fine-grained matching between query tokens and document patches.

Supports two query modes:
  - Text-only query: encode query text → search saved image embeddings
  - Image + text query: encode query image+text → search saved image embeddings

The retriever loads a pre-built index (from ColQwen2IndexBuilder) and
performs online similarity search at query time.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
from PIL import Image

from src.retrieval.base_retriever import BaseRetriever, RetrievedDocument
from src.embeddings.colqwen2_embedder import ColQwen2Embedder
from src.indexing.document_store import DocumentStore
from src.utils.logging_utils import setup_logger
from src.utils.image_utils import load_image

logger = setup_logger("retrieval.colqwen2")


class ColQwen2Retriever(BaseRetriever):
    """
    ColQwen2 late-interaction retriever.

    Loads a pre-built ColQwen2 index and performs MaxSim-based
    retrieval for user queries.

    Usage:
        retriever = ColQwen2Retriever(embedder)
        retriever.load_index("data/indexes/colqwen2_index/")
        results = retriever.retrieve("What does this chest X-ray show?", top_k=3)
    """

    def __init__(self, embedder: ColQwen2Embedder):
        """
        Args:
            embedder: Loaded ColQwen2Embedder instance (shared with indexing).
        """
        self.embedder = embedder
        self.document_store = DocumentStore()
        self._embeddings: List[torch.Tensor] = []
        self._doc_ids: List[str] = []
        self._index_loaded = False

    # ------------------------------------------------------------------ #
    #  BaseRetriever: index()                                              #
    # ------------------------------------------------------------------ #

    def index(self, documents: List[Dict[str, Any]]) -> None:
        """
        Build the retrieval index from a list of document dicts.

        This is an alternative to loading a pre-built index. It encodes
        all document images with ColQwen2 and stores the embeddings.

        Args:
            documents: List of dicts, each with at least:
                - 'doc_id': str
                - 'image_path': str (path to the image file)
                - 'text': str (report text, optional)
                - 'findings': str (optional)
                - 'impression': str (optional)
                - 'metadata': dict (optional)
        """
        from src.indexing.document_store import Document

        logger.info(f"Indexing {len(documents)} documents with ColQwen2")

        images = []
        doc_ids = []

        for doc_data in documents:
            doc_id = doc_data["doc_id"]
            image_path = doc_data.get("image_path", "")

            # Add to document store
            doc = Document(
                doc_id=doc_id,
                text=doc_data.get("text", ""),
                image_path=image_path,
                findings=doc_data.get("findings"),
                impression=doc_data.get("impression"),
                metadata=doc_data.get("metadata", {}),
            )
            self.document_store.add_document(doc)

            # Load image for encoding
            if image_path:
                try:
                    image = load_image(image_path)
                    images.append(image)
                    doc_ids.append(doc_id)
                except (FileNotFoundError, ValueError) as e:
                    logger.warning(f"Skipping {doc_id}: {e}")

        # Encode all images
        if images:
            self._embeddings = self.embedder.encode_images(images)
            self._doc_ids = doc_ids
            self._index_loaded = True
            logger.info(f"Indexed {len(self._embeddings)} documents")
        else:
            logger.warning("No images found to index")

    # ------------------------------------------------------------------ #
    #  BaseRetriever: retrieve()                                           #
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        query: str,
        query_image: Optional[Image.Image] = None,
        top_k: int = 3,
    ) -> List[RetrievedDocument]:
        """
        Retrieve the top-k most relevant documents for a query.

        Supports two query modes:
          1. Text-only: query string → ColQwen2 text embedding → MaxSim search
          2. Image + text: query image + text → ColQwen2 joint embedding → MaxSim search

        Args:
            query:       Text query string.
            query_image: Optional query image (for multimodal retrieval).
            top_k:       Number of documents to return.

        Returns:
            List of RetrievedDocument sorted by relevance (highest first).
        """
        if not self._index_loaded:
            raise RuntimeError(
                "No index loaded. Call load_index() or index() first."
            )

        if not self._embeddings:
            logger.warning("Index is empty, no documents to retrieve from")
            return []

        # Encode the query
        if query_image is not None:
            # Mode 2: Image + text query
            logger.info(f"Retrieval mode: image + text query")
            query_embeddings = self.embedder.encode_image_queries(
                images=[query_image],
                queries=[query],
            )
        else:
            # Mode 1: Text-only query
            logger.info(f"Retrieval mode: text-only query")
            query_embeddings = self.embedder.encode_queries([query])

        # Compute MaxSim scores against all indexed documents
        scores = self.embedder.score(
            query_embeddings=query_embeddings,
            doc_embeddings=self._embeddings,
        )

        # scores shape: [1, n_docs] — squeeze to [n_docs]
        scores = scores.squeeze(0)

        # Get top-k indices
        k = min(top_k, len(self._doc_ids))
        top_scores, top_indices = torch.topk(scores, k=k)

        # Build RetrievedDocument results
        results = []
        for rank, (score, idx) in enumerate(zip(top_scores.tolist(), top_indices.tolist())):
            doc_id = self._doc_ids[idx]
            doc = self.document_store.get_document(doc_id)

            if doc is None:
                logger.warning(f"Document {doc_id} not found in store")
                continue

            # Try loading the image for the retrieved document
            retrieved_image = None
            if doc.image_path:
                try:
                    retrieved_image = load_image(doc.image_path)
                except (FileNotFoundError, ValueError):
                    pass

            result = RetrievedDocument(
                doc_id=doc_id,
                score=score,
                text=doc.text,
                image=retrieved_image,
                image_path=doc.image_path,
                source="colqwen2",
                metadata={
                    "rank": rank + 1,
                    "findings": doc.findings,
                    "impression": doc.impression,
                    **doc.metadata,
                },
            )
            results.append(result)

        logger.info(
            f"Retrieved {len(results)} documents "
            f"(scores: {[f'{r.score:.4f}' for r in results]})"
        )
        return results

    # ------------------------------------------------------------------ #
    #  BaseRetriever: save/load index                                      #
    # ------------------------------------------------------------------ #

    def save_index(self, path: str) -> None:
        """
        Save the retrieval index to disk.

        Args:
            path: Directory path to save the index.
        """
        index_path = Path(path)
        index_path.mkdir(parents=True, exist_ok=True)

        # Save document store
        self.document_store.save(str(index_path / "document_store.json"))

        # Save embeddings
        torch.save(self._embeddings, str(index_path / "embeddings.pt"))

        # Save doc IDs
        with open(index_path / "doc_ids.json", "w", encoding="utf-8") as f:
            json.dump(self._doc_ids, f, indent=2)

        logger.info(f"Index saved to {index_path}")

    def load_index(self, path: str) -> None:
        """
        Load a previously saved index from disk.

        This loads the document store, embeddings, and doc ID mapping
        created by ColQwen2IndexBuilder or save_index().

        Args:
            path: Directory path containing the saved index.

        Raises:
            FileNotFoundError: If required index files are missing.
        """
        index_path = Path(path)
        if not index_path.exists():
            raise FileNotFoundError(f"Index directory not found: {index_path}")

        # Load document store
        docstore_path = index_path / "document_store.json"
        if not docstore_path.exists():
            raise FileNotFoundError(f"Document store not found: {docstore_path}")
        self.document_store.load(str(docstore_path))

        # Load embeddings
        embeddings_path = index_path / "embeddings.pt"
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")
        self._embeddings = torch.load(
            str(embeddings_path),
            map_location="cpu",
        )

        # Load doc IDs
        doc_ids_path = index_path / "doc_ids.json"
        if not doc_ids_path.exists():
            raise FileNotFoundError(f"Doc IDs not found: {doc_ids_path}")
        with open(doc_ids_path, "r", encoding="utf-8") as f:
            self._doc_ids = json.load(f)

        self._index_loaded = True

        logger.info(
            f"Index loaded: {len(self._embeddings)} embeddings, "
            f"{len(self.document_store)} documents"
        )

        # Validate consistency
        if len(self._embeddings) != len(self._doc_ids):
            logger.warning(
                f"Mismatch: {len(self._embeddings)} embeddings "
                f"vs {len(self._doc_ids)} doc IDs"
            )

    # ------------------------------------------------------------------ #
    #  Info                                                                #
    # ------------------------------------------------------------------ #

    @property
    def is_index_loaded(self) -> bool:
        """Whether an index has been loaded or built."""
        return self._index_loaded

    @property
    def num_indexed(self) -> int:
        """Number of indexed documents."""
        return len(self._embeddings)

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the retriever state."""
        return {
            "retriever": "ColQwen2Retriever",
            "index_loaded": self._index_loaded,
            "num_indexed": len(self._embeddings),
            "num_documents": len(self.document_store),
            "embedder_loaded": self.embedder.is_loaded,
        }
