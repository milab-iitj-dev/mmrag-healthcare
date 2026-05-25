"""
RAG-augmented generation engine.

Orchestrates the full retrieval-augmented generation pipeline:
    1. Receive question (+ optional query image)
    2. Retrieve relevant evidence from ColQwen2 index
    3. Build context from top-k retrieved documents
    4. Call LLaVA with assembled prompt + image
    5. Return answer with full provenance

This is the main generation entry point for RAG-enabled queries.
It wires together the retriever, context builder, and VLM into
a single clean interface.
"""

import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from PIL import Image

from src.generation.base_generator import BaseVLM, VLMOutput
from src.retrieval.colqwen2_retriever import ColQwen2Retriever
from src.retrieval.base_retriever import RetrievedDocument
from src.context.context_builder import ContextBuilder
from src.utils.logging_utils import setup_logger

logger = setup_logger("generation.rag")


@dataclass
class RAGOutput:
    """
    Full output from the RAG generation pipeline.

    Contains the generated answer plus all intermediate results
    for traceability and evaluation.
    """
    answer: str                                       # final generated answer
    retrieved_docs: List[RetrievedDocument] = field(default_factory=list)
    context_text: str = ""                            # assembled context string
    query: str = ""                                   # original query
    vlm_output: Optional[VLMOutput] = None            # raw VLM output
    retrieval_time_sec: float = 0.0                   # time spent on retrieval
    generation_time_sec: float = 0.0                  # time spent on generation
    total_time_sec: float = 0.0                       # total pipeline time
    metadata: Dict[str, Any] = field(default_factory=dict)


class RAGGenerator:
    """
    RAG-augmented generation engine.

    Wires together:
      - ColQwen2Retriever: find relevant cases
      - ContextBuilder: format evidence for VLM
      - LLaVA (BaseVLM): generate grounded answer

    Usage:
        generator = RAGGenerator(vlm, retriever, context_builder)
        output = generator.generate("What does this chest X-ray show?")
        print(output.answer)
        print(output.retrieved_docs)  # provenance
    """

    def __init__(
        self,
        vlm: BaseVLM,
        retriever: ColQwen2Retriever,
        context_builder: Optional[ContextBuilder] = None,
        top_k: int = 3,
    ):
        """
        Args:
            vlm:             Loaded VLM (LLaVA) for answer generation.
            retriever:       Loaded ColQwen2Retriever with index.
            context_builder: ContextBuilder for formatting evidence.
                             If None, creates one with defaults.
            top_k:           Number of documents to retrieve.
        """
        self.vlm = vlm
        self.retriever = retriever
        self.context_builder = context_builder or ContextBuilder()
        self.top_k = top_k

    # ------------------------------------------------------------------ #
    #  Full RAG pipeline                                                   #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        query: str,
        query_image: Optional[Image.Image] = None,
        top_k: Optional[int] = None,
        max_new_tokens: int = 512,
    ) -> RAGOutput:
        """
        Run the full RAG pipeline: retrieve → build context → generate.

        Supports two query modes:
          1. Text-only query: retrieves similar cases, uses the best
             retrieved image as visual input for LLaVA.
          2. Image + text query: retrieves similar cases, uses the
             query image as visual input for LLaVA.

        Args:
            query:          The clinical question.
            query_image:    Optional query image (e.g., patient's X-ray).
            top_k:          Override default top_k for this query.
            max_new_tokens: Max tokens for LLaVA generation.

        Returns:
            RAGOutput with answer, retrieved docs, context, and timing.
        """
        total_start = time.time()
        k = top_k or self.top_k

        logger.info(f"RAG pipeline: query='{query[:80]}...' top_k={k}")

        # Step 1: Retrieve relevant documents
        retrieval_start = time.time()
        retrieved_docs = self.retriever.retrieve(
            query=query,
            query_image=query_image,
            top_k=k,
        )
        retrieval_time = time.time() - retrieval_start
        logger.info(f"  Retrieval: {len(retrieved_docs)} docs in {retrieval_time:.2f}s")

        # Step 2: Build context from retrieved evidence
        context_text = self.context_builder.build_context(retrieved_docs)
        logger.info(f"  Context: {len(context_text)} chars")

        # Step 3: Select the image for LLaVA
        # If user provided a query image, use that.
        # Otherwise, use the best retrieved image.
        if query_image is not None:
            llava_image = query_image
            logger.info("  Image: using query image")
        else:
            llava_image = self.context_builder.get_best_image(retrieved_docs)
            if llava_image is not None:
                logger.info("  Image: using best retrieved image")
            else:
                logger.warning("  No image available for LLaVA")

        # Step 4: Generate answer with LLaVA
        generation_start = time.time()

        if llava_image is None:
            # Cannot run LLaVA without an image
            logger.error("  Cannot generate: no image available")
            return RAGOutput(
                answer="[Error: No image available for visual question answering]",
                retrieved_docs=retrieved_docs,
                context_text=context_text,
                query=query,
                retrieval_time_sec=round(retrieval_time, 2),
                total_time_sec=round(time.time() - total_start, 2),
                metadata={"error": "no_image"},
            )

        vlm_output = self.vlm.generate(
            image=llava_image,
            question=query,
            context=context_text,
            max_new_tokens=max_new_tokens,
        )
        generation_time = time.time() - generation_start

        total_time = time.time() - total_start

        logger.info(f"  Generation: {generation_time:.2f}s")
        logger.info(f"  Total: {total_time:.2f}s")
        logger.info(f"  Answer: {vlm_output.answer[:200]}...")

        return RAGOutput(
            answer=vlm_output.answer,
            retrieved_docs=retrieved_docs,
            context_text=context_text,
            query=query,
            vlm_output=vlm_output,
            retrieval_time_sec=round(retrieval_time, 2),
            generation_time_sec=round(generation_time, 2),
            total_time_sec=round(total_time, 2),
            metadata={
                "top_k": k,
                "num_retrieved": len(retrieved_docs),
                "context_length": len(context_text),
                "query_has_image": query_image is not None,
                "image_source": "query" if query_image else "retrieved",
            },
        )

    # ------------------------------------------------------------------ #
    #  Simple generation (no retrieval, backward compat)                   #
    # ------------------------------------------------------------------ #

    def generate_simple(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 512,
    ) -> VLMOutput:
        """
        Direct VQA without retrieval (Phase 1 compatibility).

        Bypasses the retrieval pipeline and generates directly
        from the image and question.

        Args:
            image:          Input image.
            question:       The question.
            max_new_tokens: Max tokens to generate.

        Returns:
            VLMOutput from the VLM.
        """
        return self.vlm.generate(
            image=image,
            question=question,
            max_new_tokens=max_new_tokens,
        )
