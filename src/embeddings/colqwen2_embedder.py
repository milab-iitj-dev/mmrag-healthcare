"""
ColQwen2 multi-vector embedding model.

Produces per-token embeddings for late-interaction retrieval (MaxSim).
Unlike CLIP's single-vector approach, ColQwen2 retains spatial information
by generating one embedding per image patch / text token.

Uses the HuggingFace-native ColQwen2ForRetrieval + ColQwen2Processor
from transformers>=4.46.0 for stable, dependency-light usage.

Key design decisions:
  - Does NOT extend BaseEmbedder because ColQwen2 produces multi-vector
    torch.Tensor outputs (shape [n_tokens, embed_dim]), not single-vector
    np.ndarray outputs. Keeping a separate interface is more honest than
    forcing incompatible shapes through the same API.
  - Supports both image encoding (for offline indexing) and text/image
    query encoding (for online retrieval).
  - Batch processing with configurable batch size for VRAM management.
"""

import torch
from typing import List, Optional, Dict, Any
from PIL import Image

from src.utils.logging_utils import setup_logger

logger = setup_logger("embeddings.colqwen2")


def _extract_embeddings(outputs, context=""):
    """
    Safely extract embedding tensor from ColQwen2 model outputs.

    Handles version differences across transformers and colpali-engine:
      - HF native:      outputs.embeddings
      - colpali-engine:  outputs.reps or raw tensor
      - fallback:        outputs.last_hidden_state
    """
    if isinstance(outputs, torch.Tensor):
        return outputs
    if hasattr(outputs, "embeddings") and outputs.embeddings is not None:
        return outputs.embeddings
    if hasattr(outputs, "reps") and outputs.reps is not None:
        return outputs.reps
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state
    if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        if isinstance(outputs[0], torch.Tensor):
            return outputs[0]
    raise ValueError(
        f"Unknown ColQwen2 output format in {context}: "
        f"type={type(outputs)}"
    )


class ColQwen2Embedder:
    """
    ColQwen2 multi-vector encoder for document retrieval.

    Produces per-token embeddings for late-interaction (MaxSim) scoring.
    Used in two places:
      1. Offline indexing: encode OpenI images → stored embeddings
      2. Online retrieval: encode user queries → query embeddings

    Usage:
        embedder = ColQwen2Embedder()
        embedder.load(config)

        # Offline: encode document images
        doc_embeddings = embedder.encode_images([pil_image_1, pil_image_2])

        # Online: encode text query
        query_embeddings = embedder.encode_queries(["What does this X-ray show?"])

        # Online: encode image + text query
        query_embeddings = embedder.encode_image_queries(
            images=[query_image],
            queries=["What abnormalities are visible?"]
        )

        # Score
        scores = embedder.score(query_embeddings, doc_embeddings)
    """

    def __init__(self):
        self._model = None
        self._processor = None
        self._device = None
        self._loaded = False
        self._model_name = "colqwen2"

    # ------------------------------------------------------------------ #
    #  Model loading                                                       #
    # ------------------------------------------------------------------ #

    def load(self, config: dict) -> None:
        """
        Load ColQwen2 model and processor.

        Args:
            config: Config dict. Expected structure:
                config["retrieval"]["colqwen2"]["model_name"] = "vidore/colqwen2-v1.0-hf"
                config["retrieval"]["colqwen2"]["batch_size"] = 4
        """
        from transformers import ColQwen2ForRetrieval, ColQwen2Processor

        # HuggingFace authentication for gated models
        # Set HF_TOKEN environment variable or use `huggingface-cli login`
        try:
            import os
            from huggingface_hub import login
            hf_token = os.environ.get("HF_TOKEN", os.environ.get("HUGGINGFACE_TOKEN"))
            if hf_token:
                login(token=hf_token)
                logger.info("  HuggingFace authentication successful")
            else:
                logger.info("  No HF_TOKEN set — using cached credentials or public models")
        except Exception as e:
            logger.warning(f"  HuggingFace login skipped: {e}")

        colqwen2_cfg = config.get("retrieval", {}).get("colqwen2", {})
        model_id = colqwen2_cfg.get("model_name", "vidore/colqwen2-v1.0-hf")
        self._batch_size = colqwen2_cfg.get("batch_size", 4)

        logger.info(f"Loading ColQwen2 model: {model_id}")

        # Load processor
        self._processor = ColQwen2Processor.from_pretrained(model_id)
        logger.info("  Processor loaded")

        # Load model in bfloat16 for memory efficiency
        self._model = ColQwen2ForRetrieval.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()

        self._device = self._model.device
        self._loaded = True

        logger.info(f"  Model loaded on device: {self._device}")
        logger.info(f"  Model dtype: {self._model.dtype}")
        logger.info(f"  Batch size: {self._batch_size}")

    # ------------------------------------------------------------------ #
    #  Document image encoding (offline indexing)                           #
    # ------------------------------------------------------------------ #

    def encode_images(
        self,
        images: List[Image.Image],
        batch_size: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """
        Encode document images into multi-vector embeddings.

        Each image produces a tensor of shape [n_patches, embed_dim],
        where n_patches depends on the image resolution.

        Args:
            images:     List of PIL images (document pages / X-rays).
            batch_size: Override default batch size if needed.

        Returns:
            List of torch.Tensor, one per image. Each tensor has shape
            [n_patches, embed_dim] and is stored on CPU for persistence.
        """
        self._check_loaded()
        batch_size = batch_size or self._batch_size
        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            logger.info(
                f"  Encoding image batch {i // batch_size + 1}"
                f"/{(len(images) + batch_size - 1) // batch_size}"
                f" ({len(batch_images)} images)"
            )

            # Process images through ColQwen2 processor
            inputs = self._processor(
                images=batch_images,
                return_tensors="pt",
            ).to(self._model.device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            # Extract embeddings tensor from model output object
            embeddings = _extract_embeddings(outputs, context="encode_images")
            for j in range(embeddings.shape[0]):
                all_embeddings.append(embeddings[j].cpu())

        logger.info(f"  Encoded {len(all_embeddings)} images total")
        return all_embeddings

    # ------------------------------------------------------------------ #
    #  Text query encoding (online retrieval — text only)                   #
    # ------------------------------------------------------------------ #

    def encode_queries(
        self,
        queries: List[str],
        batch_size: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """
        Encode text queries into multi-vector embeddings.

        Each query produces a tensor of shape [n_tokens, embed_dim].

        Args:
            queries:    List of query strings.
            batch_size: Override default batch size if needed.

        Returns:
            List of torch.Tensor, one per query. Each tensor has shape
            [n_tokens, embed_dim] and is stored on CPU.
        """
        self._check_loaded()
        batch_size = batch_size or self._batch_size
        all_embeddings = []

        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]

            inputs = self._processor(
                text=batch_queries,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self._model.device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            embeddings = _extract_embeddings(outputs, context="encode_queries")
            for j in range(embeddings.shape[0]):
                all_embeddings.append(embeddings[j].cpu())

        return all_embeddings

    # ------------------------------------------------------------------ #
    #  Image + text query encoding (online retrieval — multimodal)          #
    # ------------------------------------------------------------------ #

    def encode_image_queries(
        self,
        images: List[Image.Image],
        queries: List[str],
        batch_size: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """
        Encode image + text queries into multi-vector embeddings.

        For multimodal queries where the user provides both an image
        and a text question. The processor combines both modalities
        into a joint representation.

        Args:
            images:     List of query images (one per query).
            queries:    List of query strings (same length as images).
            batch_size: Override default batch size if needed.

        Returns:
            List of torch.Tensor, one per query.

        Note:
            If the model/processor doesn't natively support joint
            image+text query encoding, we fall back to encoding the
            image alone (since ColQwen2 is primarily a visual retriever).
        """
        self._check_loaded()
        batch_size = batch_size or self._batch_size

        if len(images) != len(queries):
            raise ValueError(
                f"images and queries must have the same length, "
                f"got {len(images)} images and {len(queries)} queries"
            )

        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_queries = queries[i:i + batch_size]

            # ColQwen2 processor supports images + text together
            # The text is used as a query prefix/context with the image
            try:
                inputs = self._processor(
                    images=batch_images,
                    text=batch_queries,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self._model.device)
            except Exception:
                # Fallback: if joint encoding fails, encode image only
                logger.warning(
                    "Joint image+text query encoding not supported, "
                    "falling back to image-only encoding"
                )
                inputs = self._processor(
                    images=batch_images,
                    return_tensors="pt",
                ).to(self._model.device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            embeddings = _extract_embeddings(outputs, context="encode_image_queries")
            for j in range(embeddings.shape[0]):
                all_embeddings.append(embeddings[j].cpu())

        return all_embeddings

    # ------------------------------------------------------------------ #
    #  Scoring (MaxSim late interaction)                                    #
    # ------------------------------------------------------------------ #

    def score(
        self,
        query_embeddings: List[torch.Tensor],
        doc_embeddings: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute MaxSim late-interaction scores between queries and documents.

        MaxSim (ColBERT-style):
          For each query token, find its max cosine similarity to any
          document token, then sum across all query tokens.

          score(q, d) = sum_i( max_j( cos_sim(q_i, d_j) ) )

        This preserves the multi-vector retrieval behavior of ColQwen2.

        Args:
            query_embeddings: List of query tensors [n_tokens, embed_dim].
            doc_embeddings:   List of document tensors [n_patches, embed_dim].

        Returns:
            torch.Tensor of shape [n_queries, n_docs] with similarity scores.
        """
        self._check_loaded()

        n_queries = len(query_embeddings)
        n_docs = len(doc_embeddings)
        scores = torch.zeros(n_queries, n_docs)

        for qi in range(n_queries):
            q = query_embeddings[qi].to(self._model.device).float()
            # L2 normalize query tokens
            q = torch.nn.functional.normalize(q, p=2, dim=-1)

            for di in range(n_docs):
                d = doc_embeddings[di].to(self._model.device).float()
                # L2 normalize document tokens
                d = torch.nn.functional.normalize(d, p=2, dim=-1)

                # [n_q_tokens, n_d_tokens] cosine similarity matrix
                sim_matrix = torch.matmul(q, d.transpose(0, 1))

                # MaxSim: for each query token, take max similarity across doc tokens
                max_sim_per_token = sim_matrix.max(dim=-1).values  # [n_q_tokens]

                # Sum over query tokens to get final score
                scores[qi, di] = max_sim_per_token.sum().item()

        logger.info(
            f"  MaxSim scoring: {n_queries} queries x {n_docs} docs, "
            f"scores range [{scores.min():.2f}, {scores.max():.2f}]"
        )

        return scores

    # ------------------------------------------------------------------ #
    #  Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _pad_and_stack(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """
        Pad a list of variable-length tensors and stack into a batch.

        Each tensor has shape [seq_len, embed_dim]. We pad seq_len to
        the maximum across all tensors, using zeros.

        Returns:
            Batched tensor of shape [batch_size, max_seq_len, embed_dim].
        """
        max_len = max(t.shape[0] for t in tensors)
        embed_dim = tensors[0].shape[1]

        padded = torch.zeros(len(tensors), max_len, embed_dim, dtype=tensors[0].dtype)
        for i, t in enumerate(tensors):
            padded[i, :t.shape[0], :] = t

        return padded

    def _check_loaded(self) -> None:
        """Raise if model not loaded."""
        if not self._loaded:
            raise RuntimeError(
                "ColQwen2 model not loaded. Call load(config) first."
            )

    @property
    def is_loaded(self) -> bool:
        """Whether the model is loaded and ready for inference."""
        return self._loaded

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self):
        """The device the model is loaded on."""
        return self._device

    def unload(self) -> None:
        """
        Unload the model to free VRAM.

        Useful when switching between ColQwen2 and LLaVA
        on memory-constrained GPUs.
        """
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._loaded = False

        # Force garbage collection and CUDA cache clearing
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("ColQwen2 model unloaded, VRAM freed")
