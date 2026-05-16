# Step-by-Step Guide — Healthcare Multimodal RAG

A beginner-friendly, detailed guide to understanding and building the entire Healthcare Multimodal RAG system. This document explains every technique used in the project, what order to implement things, and how all the pieces fit together.

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [The Big Picture](#2-the-big-picture)
3. [Phase 1: Simple VQA Pipeline](#3-phase-1-simple-vqa-pipeline)
   - [OpenI Dataset Ingestion](#31-openi-dataset-ingestion)
   - [LLaVA-1.5-7B — The Vision-Language Model](#32-llava-15-7b--the-vision-language-model)
   - [4-bit Quantization](#33-4-bit-quantization)
   - [QLoRA Fine-tuning](#34-qlora-fine-tuning)
   - [LoRA — Low-Rank Adaptation](#35-lora--low-rank-adaptation)
4. [Phase 2: Evaluation](#4-phase-2-evaluation)
5. [Phase 3: BM25 Text Retrieval](#5-phase-3-bm25-text-retrieval)
6. [Phase 4: Full Multimodal Retrieval](#6-phase-4-full-multimodal-retrieval)
   - [CLIP Retrieval](#61-clip-retrieval)
   - [ColQwen2 — Late-Interaction Retrieval](#62-colqwen2--late-interaction-retrieval)
   - [RRF — Reciprocal Rank Fusion](#63-rrf--reciprocal-rank-fusion)
   - [Cross-Encoder Reranking](#64-cross-encoder-reranking)
   - [Context Building](#65-context-building)
7. [Phase 5: Advanced Generation & Safety](#7-phase-5-advanced-generation--safety)
   - [Qwen2-VL — Advanced Vision-Language Model](#71-qwen2-vl--advanced-vision-language-model)
   - [Multimodal Reasoning](#72-multimodal-reasoning)
   - [NLI Grounding](#73-nli-grounding)
8. [Step-by-Step Implementation Order](#8-step-by-step-implementation-order)
9. [Glossary](#9-glossary)

---

## 1. What Is This Project?

Imagine you're a doctor looking at a chest X-ray. You want to ask a question like:

> *"What abnormalities are visible in this X-ray?"*

This project builds an AI system that can:

1. **Look** at the X-ray image (computer vision)
2. **Search** a database of past medical reports for similar cases (retrieval)
3. **Read** the most relevant past reports (context building)
4. **Answer** your question using both the image and the retrieved evidence (generation)
5. **Verify** its answer against the evidence so it doesn't hallucinate (grounding)

This is called **Multimodal RAG** (Retrieval-Augmented Generation):

- **Multimodal** = works with multiple types of data (images + text)
- **RAG** = retrieves relevant information before generating an answer
- **Healthcare** = specialized for medical images and reports

---

## 2. The Big Picture

The system has two main phases of operation:

### Offline Phase (runs once)
You prepare the knowledge base by processing all your medical data:
```
All X-rays + Reports  →  Compute Embeddings  →  Build Indexes  →  Save to Disk
```

### Online Phase (runs per query)
When a user asks a question:
```
User Image + Question
       │
       ▼
   RETRIEVE similar cases from the knowledge base
       │
       ▼
   RERANK to find the most relevant evidence
       │
       ▼
   BUILD CONTEXT from the best evidence
       │
       ▼
   GENERATE answer using VLM (image + question + context)
       │
       ▼
   VERIFY the answer is grounded in evidence
       │
       ▼
   Final Answer (with confidence and citations)
```

We build this incrementally across 5 phases, starting simple and adding complexity.

---

## 3. Phase 1: Simple VQA Pipeline

**Goal:** Get a working Image → Question → Answer pipeline with zero retrieval.

This is the foundation. A user gives an X-ray image and a question, and the model answers directly.

### 3.1 OpenI Dataset Ingestion

**What is OpenI?**

OpenI (Open Access Biomedical Image Search Engine) is a public dataset from the National Library of Medicine. It contains:

- **~3,800 chest X-ray images** (frontal and lateral views)
- **~3,800 radiology reports** with structured sections:
  - **Findings**: what the radiologist observes
  - **Impression**: the radiologist's conclusion/diagnosis

**How we use it:**

1. **Load** the CSV files (`indiana_reports.csv` and `indiana_projections.csv`)
2. **Match** each report to its corresponding X-ray image(s)
3. **Prefer frontal (PA) views** — these are the standard diagnostic view
4. **Convert** each matched pair into a `MedicalSample` with:
   - Image (loaded as PIL Image)
   - Report text (findings + impression)
   - A VQA question (e.g., "Describe the findings in this chest X-ray")
   - The answer (the impression/findings from the report)

**Where in the code:** `src/data/openi_dataset.py`

### 3.2 LLaVA-1.5-7B — The Vision-Language Model

**What is LLaVA?**

LLaVA (Large Language and Vision Assistant) is a model that can look at images AND understand text at the same time. Think of it as ChatGPT with eyes.

**How it works internally:**

```
┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│  X-ray Image │────→│  CLIP Vision  │────→│              │
│              │     │   Encoder     │     │              │
└──────────────┘     └───────────────┘     │   LLaMA-2    │
                            │              │   Language   │──→  Answer
                     visual tokens         │    Model     │
                            │              │              │
┌──────────────┐     ┌──────▼────────┐     │              │
│  "What does  │────→│   Projection  │────→│              │
│  this show?" │     │     Layer     │     │              │
└──────────────┘     └───────────────┘     └──────────────┘
```

1. The **CLIP vision encoder** converts the image into a sequence of "visual tokens" — numbers that represent what the model sees
2. A **projection layer** maps these visual tokens into the same space as text tokens
3. The **LLaMA-2 language model** processes both the visual tokens and the text question together
4. It generates an answer word by word

**LLaVA-1.5-7B** means:
- **1.5** = version 1.5 (improved training recipe)
- **7B** = 7 billion parameters

**Where in the code:** `src/models/llava_model.py`

### 3.3 4-bit Quantization

**The problem:** LLaVA-1.5-7B normally needs ~14 GB of GPU memory (VRAM). Most consumer GPUs (like a T4 with 16 GB) can barely fit it, leaving no room for actual computation.

**The solution:** **Quantization** = representing the model's numbers with fewer bits.

- **Normal (FP16):** Each parameter uses 16 bits → 7B × 2 bytes = ~14 GB
- **4-bit (NF4):** Each parameter uses 4 bits → 7B × 0.5 bytes = ~3.5 GB

**NF4 (NormalFloat4)** is a special 4-bit format designed by the bitsandbytes team. It's optimized for the bell-curve distribution of neural network weights, so it loses very little accuracy compared to naive 4-bit rounding.

**Double quantization** goes further: it quantizes the quantization constants themselves, saving another ~0.4 GB.

**Result:** The model fits in ~4-5 GB of VRAM, leaving room for training on a T4!

**Where in the code:** Configured in `configs/model_config.yaml` under `quantization`

### 3.4 QLoRA Fine-tuning

**What is QLoRA?**

QLoRA = **Quantized Low-Rank Adaptation**. It combines 4-bit quantization (Q) with LoRA fine-tuning.

**The idea:**
1. Load the full 7B model in 4-bit (frozen — never changes)
2. Attach tiny trainable LoRA adapters (only ~1-4 million parameters)
3. Train ONLY the LoRA adapters on your medical data
4. At inference time, the adapter output is added to the frozen model's output

**Why this matters for us:**
- We can fine-tune a 7B model on a **single GPU with 16 GB VRAM** (e.g., NVIDIA T4)
- Training takes only **1-3 hours** instead of days
- The trained adapter is only **~20 MB** (vs. 14 GB for the full model)
- We get a model specialized for medical VQA without needing massive compute

**Where in the code:** `scripts/train_local.py`

### 3.5 LoRA — Low-Rank Adaptation

**What is LoRA?**

LoRA is the core technique inside QLoRA. Here's the intuition:

Normal fine-tuning updates a weight matrix **W** (size d×d, e.g., 4096×4096 = 16.7M parameters per layer).

LoRA says: the change to W during fine-tuning is **low-rank** — it can be approximated by two much smaller matrices:

```
ΔW = A × B

Where:
  W is 4096 × 4096  (16.7M params)  ← frozen
  A is 4096 × 16    (65K params)    ← trainable
  B is 16 × 4096    (65K params)    ← trainable
  r = 16             (the "rank")
```

Instead of training 16.7M parameters, we train only 130K — a **128× reduction**!

**Key hyperparameters:**
- **r (rank):** How many dimensions the adaptation uses. Higher = more capacity. We use r=16.
- **lora_alpha:** A scaling factor (we use 32). The effective scaling is `alpha/r = 32/16 = 2`.
- **target_modules:** Which layers get LoRA adapters. We target `q_proj`, `k_proj`, `v_proj`, `o_proj` (the attention layers).

**Where in the code:** Configured in `configs/model_config.yaml` under `lora`

---

## 4. Phase 2: Evaluation

**Goal:** Measure how good the model's answers are.

### Generation Metrics

| Metric | What it measures | How it works |
|--------|-----------------|--------------|
| **BLEU** | Word overlap | Counts matching n-grams (1-2-3-4 word phrases) between prediction and reference |
| **ROUGE-L** | Longest common subsequence | Finds the longest sequence of words appearing in both prediction and reference |
| **METEOR** | Semantic overlap | Like BLEU but accounts for synonyms and word stems |
| **BERTScore** | Semantic similarity | Uses BERT embeddings to compare meaning, not just words |
| **F1** | Token overlap | Precision × Recall of individual tokens |
| **Exact Match** | Perfect accuracy | 1 if prediction exactly matches reference, 0 otherwise |

**Where in the code:** `src/utils/metrics.py`, `pipelines/evaluation.py`

---

## 5. Phase 3: BM25 Text Retrieval

**Goal:** Before answering, search past reports for similar cases.

### What is BM25?

BM25 (Best Matching 25) is a classic information retrieval algorithm. Think of it as a smart keyword search:

1. **Indexing (offline):** Break every report into words. Build an inverted index (word → list of documents containing it).
2. **Querying (online):** Break the question into words. For each word, look up which documents contain it. Score each document based on:
   - **Term Frequency (TF):** How often the query word appears in the document (with saturation — diminishing returns)
   - **Inverse Document Frequency (IDF):** Rare words matter more than common words
   - **Document Length:** Shorter documents with the same term count score higher

**Example:**
```
Query: "cardiomegaly bilateral pleural effusion"

BM25 finds reports that mention these terms frequently,
especially reports that mention the rare term "cardiomegaly"
(high IDF) rather than the common word "bilateral" (low IDF).
```

**Why BM25 first?** It's fast, simple, and surprisingly effective for text-heavy queries. It's our baseline retrieval method.

**Where in the code:** `src/retrieval/bm25_retriever.py`

---

## 6. Phase 4: Full Multimodal Retrieval

**Goal:** Search by both text AND image similarity, then fuse the results.

### 6.1 CLIP Retrieval

**What is CLIP?**

CLIP (Contrastive Language-Image Pre-training) is a model trained by OpenAI to understand the relationship between images and text. It can encode both images and text into the same vector space.

**How we use it for retrieval:**

```
Offline: Encode all X-ray images → store vectors in FAISS index
Online:  Encode query text → find nearest neighbor vectors → return matches
```

Two modes:
- **Text → Image:** "Find X-rays showing cardiomegaly" → finds matching images
- **Image → Image:** Show an X-ray → finds visually similar X-rays

**BiomedCLIP** is a variant trained specifically on medical images — better for our use case.

**Where in the code:** `src/embeddings/clip_embedder.py`, `src/retrieval/clip_retriever.py`

### 6.2 ColQwen2 — Late-Interaction Retrieval

**What is ColQwen2?**

ColQwen2 uses the ColPali architecture with a Qwen2-VL backbone. Unlike CLIP (which produces ONE vector per image), ColQwen2 produces **one vector per image patch**.

**Why this matters:**

```
CLIP:      Entire X-ray → 1 vector (512 dims)     → coarse matching
ColQwen2:  Entire X-ray → 1024 vectors (128 dims)  → fine-grained matching
```

ColQwen2 uses **MaxSim scoring**: for each query token, find its maximum similarity to any document patch, then sum. This enables **fine-grained matching** — it can match specific regions of an X-ray to specific words in the query.

**Example:** If you ask about "left lower lobe opacity", ColQwen2 can specifically match the image patches in the left lower lobe region.

**Where in the code:** `src/embeddings/colqwen2_embedder.py`, `src/retrieval/colqwen2_retriever.py`

### 6.3 RRF — Reciprocal Rank Fusion

**The problem:** We now have 3 ranked lists of documents (from BM25, CLIP, ColQwen2). How do we combine them?

**The solution: RRF** merges multiple ranked lists by focusing on **rank position** rather than raw scores.

**Formula:**
```
RRF_score(doc) = Σ  1 / (k + rank_i(doc))
                 i

Where:
  k = 60 (constant, prevents top-ranked docs from dominating too much)
  rank_i(doc) = position of doc in retriever i's result list
```

**Example:**
```
BM25 ranks document X at position 2   → score = 1/(60+2)  = 0.0161
CLIP ranks document X at position 5   → score = 1/(60+5)  = 0.0154
ColQwen2 ranks document X at position 1 → score = 1/(60+1) = 0.0164

RRF score for X = 0.0161 + 0.0154 + 0.0164 = 0.0479
```

**Why RRF?**
- Score-agnostic: works even though BM25, CLIP, and ColQwen2 produce scores on completely different scales
- Simple: no learning required
- Proven: consistently outperforms individual retrievers

**Where in the code:** `src/retrieval/hybrid_retriever.py`

### 6.4 Cross-Encoder Reranking

**The problem:** RRF gives us ~20 candidates. Some are good, some are noise. We need to pick the best 5.

**The solution:** A **cross-encoder** that jointly reads the query and each candidate document together.

**Difference from bi-encoders (CLIP, ColQwen2):**

```
Bi-encoder:     Encode query and doc SEPARATELY → compare vectors
                Fast (can pre-compute doc vectors) but less accurate

Cross-encoder:  Encode query and doc TOGETHER → produce single score
                Slow (must process each pair) but more accurate
```

The cross-encoder sees the full interaction between query and document, catching subtle relevance signals that bi-encoders miss.

**Where in the code:** `src/reranking/cross_encoder_reranker.py`

### 6.5 Context Building

After retrieval and reranking, we have the top-5 most relevant past cases. The **context builder** assembles them into a structured prompt:

```
Based on the following similar medical cases:

[Case 1] Score: 0.92
Findings: The cardiac silhouette is enlarged. There is a left-sided
pleural effusion. The lungs are otherwise clear.
Impression: Cardiomegaly with left pleural effusion.

[Case 2] Score: 0.88
Findings: ...

Given the above evidence and the provided X-ray image, please answer:
What abnormalities are visible in this chest X-ray?
```

The VLM now has both the image AND relevant past cases to inform its answer.

**Where in the code:** `src/context/context_builder.py`, `src/context/prompt_templates.py`

---

## 7. Phase 5: Advanced Generation & Safety

### 7.1 Qwen2-VL — Advanced Vision-Language Model

**Why switch from LLaVA to Qwen2-VL?**

Qwen2-VL-7B-Instruct has several advantages for RAG:
- **Native multi-image support:** Can process the query X-ray AND reference images from retrieval simultaneously
- **Better instruction following:** More reliably follows complex prompts with evidence
- **Stronger reasoning:** Better at synthesizing information from multiple sources
- **Dynamic resolution:** Can handle different image sizes without resizing

**Where in the code:** `src/models/qwen2vl_model.py`

### 7.2 Multimodal Reasoning

With the full pipeline, the VLM performs **multimodal reasoning**:

1. **Visual analysis:** Looks at the query X-ray to identify visual patterns
2. **Evidence reading:** Reads the retrieved report excerpts for context
3. **Cross-referencing:** Compares what it sees in the image with what similar past cases showed
4. **Synthesis:** Generates an answer that integrates both visual and textual evidence
5. **Citation:** References which past cases support its conclusions

This is significantly more reliable than Phase 1's direct VQA, because the model's answer is grounded in real evidence.

### 7.3 NLI Grounding

**The problem:** Even with retrieved context, LLMs can hallucinate (make up facts).

**The solution: NLI (Natural Language Inference) verification.**

NLI models classify the relationship between two sentences:
- **Entailment:** The evidence supports the claim ✅
- **Contradiction:** The evidence contradicts the claim ❌
- **Neutral:** The evidence doesn't address the claim ⚠️

**How we use it:**

```
For each claim in the generated answer:
  1. Extract the claim (e.g., "There is cardiomegaly")
  2. Check it against each piece of retrieved evidence
  3. If ENTAILED → keep the claim (grounded)
  4. If CONTRADICTED → remove or flag the claim
  5. If NEUTRAL → flag as uncertain, add disclaimer
```

**Result:** The final answer includes:
- A confidence score (% of claims that are grounded)
- Citations for supported claims
- Disclaimers for uncertain claims

**Where in the code:** `src/generation/grounding.py`

---

## 8. Step-by-Step Implementation Order

Here's exactly what to implement, in order:

### Phase 1: Simple VQA (Current ✅)
- [x] Set up project structure
- [x] Implement OpenI dataset loader (`src/data/openi_dataset.py`)
- [x] Implement LLaVA wrapper (`src/models/llava_model.py`)
- [x] Implement simple VQA pipeline (`pipelines/simple_vqa.py`)
- [x] Implement QLoRA training script (`scripts/train_local.py`)
- [x] Implement inference script (`scripts/inference.py`)
- [x] Implement validation / smoke test script (`scripts/validate.py`)

### Phase 2: Evaluation
- [ ] Implement `src/utils/metrics.py` (BLEU, ROUGE-L, BERTScore, F1)
- [ ] Implement `pipelines/evaluation.py`
- [ ] Implement `scripts/evaluate.py`
- [ ] Run evaluation on Phase 1 outputs → get baseline scores

### Phase 3: BM25 Retrieval
- [ ] Implement `src/utils/config_loader.py`
- [ ] Implement `src/indexing/document_store.py`
- [ ] Implement `src/retrieval/bm25_retriever.py`
- [ ] Implement `src/context/prompt_templates.py` (RAG prompt)
- [ ] Implement `src/context/context_builder.py`
- [ ] Implement `src/indexing/index_builder.py` (BM25 part only)
- [ ] Implement `scripts/build_index.py` (BM25 only)
- [ ] Test: build BM25 index → retrieve → check relevance
- [ ] Update `pipelines/rag_vqa.py` with BM25-only RAG
- [ ] Evaluate: does BM25 retrieval improve answer quality?

### Phase 4: Full Multimodal Retrieval
- [ ] Implement `src/embeddings/clip_embedder.py`
- [ ] Implement `src/retrieval/clip_retriever.py`
- [ ] Implement `src/embeddings/colqwen2_embedder.py`
- [ ] Implement `src/retrieval/colqwen2_retriever.py`
- [ ] Implement `src/retrieval/hybrid_retriever.py` (RRF fusion)
- [ ] Implement `src/reranking/cross_encoder_reranker.py`
- [ ] Update `src/indexing/index_builder.py` (CLIP + ColQwen2)
- [ ] Update `scripts/build_index.py` (all methods)
- [ ] Update `pipelines/rag_vqa.py` with full retrieval
- [ ] Evaluate: compare BM25-only vs hybrid retrieval

### Phase 5: Qwen2-VL + Grounding
- [ ] Implement `src/models/qwen2vl_model.py`
- [ ] Update `src/models/model_factory.py` with Qwen2-VL
- [ ] Implement `src/generation/rag_generator.py`
- [ ] Implement `src/generation/grounding.py` (NLI verification)
- [ ] Update `pipelines/rag_vqa.py` with grounding
- [ ] Final evaluation: full pipeline with all components

---

## 9. Glossary

| Term | Definition |
|------|-----------|
| **VLM** | Vision-Language Model — a model that understands both images and text |
| **RAG** | Retrieval-Augmented Generation — search relevant docs before generating |
| **VQA** | Visual Question Answering — answering questions about images |
| **QLoRA** | Quantized LoRA — fine-tuning a quantized model with LoRA adapters |
| **LoRA** | Low-Rank Adaptation — efficient fine-tuning using low-rank matrices |
| **NF4** | NormalFloat4 — 4-bit quantization format optimized for neural network weights |
| **BM25** | Best Matching 25 — classic keyword-based text retrieval algorithm |
| **CLIP** | Contrastive Language-Image Pre-training — model for image-text similarity |
| **ColQwen2** | ColPali architecture with Qwen2-VL — late-interaction multimodal retrieval |
| **RRF** | Reciprocal Rank Fusion — method to combine multiple ranked lists |
| **FAISS** | Facebook AI Similarity Search — fast nearest neighbor search library |
| **NLI** | Natural Language Inference — classifying if evidence supports a claim |
| **MaxSim** | Maximum Similarity — late-interaction scoring used by ColPali/ColQwen2 |
| **Cross-encoder** | Model that jointly encodes a query-document pair for accurate scoring |
| **Bi-encoder** | Model that encodes query and document separately for fast retrieval |
| **Adapter** | Small trainable module attached to a frozen model (e.g., LoRA weights) |
| **Grounding** | Verifying that generated text is supported by source evidence |
