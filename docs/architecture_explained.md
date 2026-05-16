# Healthcare Multimodal RAG — Complete Architecture Explained

> Based on the exact blocks shown in **Architecture.png**

![Architecture Diagram](C:/Users/DELL/.gemini/antigravity/brain/e55d7fc4-0ec8-4c4c-95f9-be6e9dfb3d8f/Architecture.png)

---

## 1. Overview — What Is This System?

This system is a **Healthcare Multimodal Retrieval-Augmented Generation (MRAG)** pipeline.

> A doctor uploads a chest X-ray and asks a question → the system searches medical knowledge → generates a **grounded, cited answer with confidence**.

**Why is this needed?** Regular AI models hallucinate — they invent medical facts. Our system solves this by *searching verified evidence before answering*, and citing sources so doctors can verify.

The architecture has **three major sections** visible in the diagram:

| Section | Color in Diagram | When It Runs | Purpose |
|---|---|---|---|
| **Offline Pipeline — Index Building** | Left section (yellow border) | Once, before use | Build searchable indexes from medical documents |
| **Online Pipeline — Retrieval** | Center section (green border) | Every query | Search indexes for relevant evidence |
| **Generation — Qwen2-VL** | Right section (purple border) | Every query | Generate grounded answer from evidence + image |

---

## 2. Offline Pipeline — Index Building

> **Runs once.** Converts raw medical documents into searchable indexes.

### Block 1: Healthcare Data Sources

```
What:   Raw medical documents — PDFs, textbooks, radiology reports, clinical guidelines
Input:  Nothing (this is the starting point)
Output: Raw files to be processed
Why:    The system needs a knowledge base to search through
```

**Example:** A radiology textbook PDF, OpenI reports CSV, clinical practice guidelines.

---

### Block 2: PDF / Page Processing — Layout + Text Extraction

```
What:   Extracts text and understands page layout from documents
Input:  Raw PDFs, images, reports
Output: Structured text with layout information (headings, tables, figures)
Why:    Medical documents have complex layouts — tables, diagrams, multi-column text.
        Simple text extraction misses structure. Layout-aware processing preserves it.
```

**How it works:** Uses document parsing libraries (like PyMuPDF, Unstructured) to extract text while remembering where each piece was on the page — was it a heading? A table cell? A figure caption?

---

### Block 3: Text Extraction — Medical NLP Processing

```
What:   Cleans and structures extracted text using medical NLP
Input:  Raw extracted text with layout info
Output: Clean, chunked text ready for indexing
Why:    Raw text has noise — headers, footers, page numbers, formatting artifacts.
        Medical NLP also identifies entities: diseases, anatomy, medications.
```

**What happens here:**
- Remove noise (page numbers, headers)
- Split into chunks (~200-500 words each, respecting section boundaries)
- Identify medical entities (e.g., "pneumonia", "left lower lobe", "consolidation")
- Tag each chunk with metadata (source document, section, quality score)

---

### Block 4: ColQwen2 Index — Layout-Aware Multi-Vector

```
What:   Creates a dense vector index that understands document layout
Input:  Processed text chunks with layout information
Output: Multi-vector index stored on disk
Why:    Traditional embedding creates ONE vector per chunk. ColQwen2 creates
        MULTIPLE vectors per page, preserving spatial layout information.
```

**What is ColQwen2?**

ColQwen2 is a **multi-vector retrieval model** built on Qwen2-VL. Unlike traditional embeddings that collapse an entire page into one vector, ColQwen2:

1. Treats each document page as an image
2. Creates a grid of vectors that preserve spatial layout
3. During search, it matches the query against ALL vectors on a page
4. This means it can find information in tables, figures, and complex layouts that flat text search misses

**Why chosen:** Medical documents are layout-heavy (radiology report tables, anatomical diagrams). ColQwen2 understands *where* on the page information appears, not just *what* it says.

---

### Block 5: BM25 Index — Exact Keyword Match

```
What:   Creates a traditional keyword search index
Input:  Processed text chunks
Output: Inverted index (word → list of documents containing it)
Why:    Fast, reliable, catches exact medical terms that neural search might miss
```

**What is BM25?**

BM25 (Best Matching 25) is the algorithm behind search engines like Elasticsearch. It works by:

1. Building a dictionary: for each word, record which documents contain it
2. When searching, score each document by how many query words it contains
3. Weight rare words higher than common words (TF-IDF principle)

**Example:**
- Query: "pleural effusion right hemithorax"
- BM25 finds all chunks containing "pleural", "effusion", "hemithorax"
- Chunks with all three words score highest

**Why used alongside neural search:** Neural models understand meaning but sometimes miss exact medical terminology. BM25 catches the precise term "cardiomegaly" even if the neural model would match "enlarged heart" instead.

---

### Block 6: CLIP Image Index — Visual Similarity

```
What:   Creates a visual search index from medical images
Input:  Medical images (X-rays, CT slices, etc.)
Output: Vector index where each image is a 512/768-dim vector
Why:    Enables finding visually similar past cases
```

**What is CLIP?**

CLIP (Contrastive Language-Image Pre-training) by OpenAI maps both images and text into the **same vector space**. This means:
- Two similar X-rays → nearby vectors
- An X-ray showing pneumonia → near the text "lung consolidation"

**How it helps:** If a doctor uploads an X-ray showing an unusual pattern, CLIP finds past cases that look similar. Those past cases have radiologist reports attached — instant relevant evidence.

---

### Block 7: SQLite Metadata Store — Source · Section · Quality

```
What:   Stores metadata about every indexed chunk
Input:  Metadata from all indexing steps
Output: Queryable database with source, section, quality info
Why:    When the system retrieves evidence, it needs to cite WHERE it came from
```

**What it stores for each chunk:**
- `source`: which document (e.g., "Harrison's Radiology Ch.5")
- `section`: which section (e.g., "Pneumonia — Imaging Findings")
- `quality`: reliability score (textbook > blog post)
- `chunk_id`: links to the vector indexes

---

### Checkpoint: "Indexes Ready"

Once all three indexes (ColQwen2, BM25, CLIP) and the metadata store are built, the offline pipeline is complete. The system is ready to answer queries.

---

## 3. Online Pipeline — Retrieval

> **Runs every time a user asks a question.** Searches all indexes and finds the best evidence.

### Block 8: User Query + Optional Image

```
What:   The starting point of every query
Input:  A text question + optionally a medical image
Output: Query text and image passed to downstream blocks
```

**Example:**
```
Question: "Is there pleural effusion in this chest X-ray?"
Image:    chest_xray_patient_42.png
```

---

### Block 9: Query Understanding — NER + Intent Classification

```
What:   Analyzes the query to understand what the user is asking
Input:  Raw query text
Output: Structured query with medical entities and intent
Why:    A smarter query leads to better retrieval results
```

**Two sub-tasks:**

1. **NER (Named Entity Recognition):** Extracts medical terms
   - "Is there **pleural effusion** in the **right lung**?" → entities: `pleural effusion`, `right lung`

2. **Intent Classification:** Determines what type of answer is needed
   - "Is there...?" → yes/no finding question
   - "Describe the..." → descriptive report
   - "What is the differential diagnosis?" → differential reasoning

**Why this matters:** Instead of searching for the entire raw question, the system searches for the extracted medical entities — much more precise.

---

### Block 10: ColQwen2 Retrieval — Layout-Aware · Top-20 Pages

```
What:   Searches the ColQwen2 index for relevant document pages
Input:  Processed query from Query Understanding
Output: Top 20 most relevant document pages
Why:    Finds evidence in complex layouts (tables, figures) that text search misses
```

**How it works at query time:**
1. Encode the query into multiple vectors
2. Compare against all page vectors in the index
3. Score each page by how well ANY of its vectors match the query
4. Return top 20 pages

---

### Block 11: BM25 Retrieval — Exact Medical Keywords

```
What:   Searches the BM25 index for keyword matches
Input:  Medical keywords from Query Understanding
Output: Top-K text chunks matching the keywords
Why:    Fast, precise, catches exact medical terminology
```

---

### Block 12: CLIP Retrieval — Conditioned on Image · Top-10

```
What:   If the user provided an image, finds visually similar past cases
Input:  User's medical image (the query image)
Output: Top 10 visually similar images with their reports
Why:    Similar-looking X-rays often have similar findings
```

**Important:** This block only activates when the user provides an image. For text-only queries, it's skipped.

---

### Block 13: RRF Fusion — Score Fusion + Page-ID Alignment

```
What:   Combines results from all three retrieval systems into one ranked list
Input:  Three separate ranked lists (ColQwen2, BM25, CLIP)
Output: One unified ranked list
Why:    Each retrieval system has strengths; combining them is better than any one alone
```

**What is RRF (Reciprocal Rank Fusion)?**

A simple formula that merges multiple ranked lists:

```
For each document d:
  RRF_score(d) = Σ  1 / (k + rank_i(d))
                 i∈{ColQwen2, BM25, CLIP}

where k = 60 (constant), rank_i = position in list i
```

**Example:**
| Document | ColQwen2 Rank | BM25 Rank | CLIP Rank | RRF Score |
|---|---|---|---|---|
| Doc A | 1 | 5 | 3 | 1/61 + 1/65 + 1/63 = 0.0475 |
| Doc B | 10 | 1 | - | 1/70 + 1/61 = 0.0307 |
| Doc C | 3 | 2 | 1 | 1/63 + 1/62 + 1/61 = 0.0483 ← **wins** |

Doc C wins because it appears near the top in ALL three lists.

**Page-ID Alignment:** When ColQwen2 retrieves a page and BM25 retrieves a chunk from that same page, RRF aligns them by page ID so they boost each other's score.

**Why RRF?** It's parameter-free, robust, and works even when different systems use different score scales.

---

### Block 14: Reranking — Cross-Encoder Precision

```
What:   Re-scores the top candidates using a more powerful model
Input:  Top candidates from RRF fusion
Output: Final reranked evidence list
Why:    Initial retrieval is fast but rough; reranking is slow but precise
```

**How cross-encoder reranking works:**

Initial retrieval (BM25, ColQwen2) encodes query and document *separately* then compares vectors. This is fast but approximate.

A cross-encoder reads the query and document *together* — it sees the full context of how they relate:

```
Input to cross-encoder:  [query] [SEP] [document chunk]
Output:                  relevance score (0.0 to 1.0)
```

This is 10-100× slower than vector search, so we only apply it to the already-filtered top candidates (e.g., top 20 → rerank → top 5).

---

### Block 15: Context Builder — Evidence Assembly + Token Budget

```
What:   Assembles the final evidence context for the generator
Input:  Top reranked evidence chunks + metadata from SQLite
Output: Formatted evidence string within the token budget
Why:    The generator has a limited context window; we must fit the best evidence in it
```

**What it does:**
1. Takes the top-K reranked evidence chunks
2. Fetches metadata from SQLite (source, section, quality)
3. Formats each piece with citation markers
4. Fits within the token budget (e.g., 2048 tokens for evidence)
5. Prioritizes higher-quality, higher-ranked evidence

**Output format:**
```
EVIDENCE:
[1] Source: Harrison's Radiology, Ch.5 — Pleural Disease
    "Pleural effusion appears as blunting of the costophrenic angle
     on upright chest radiograph..."

[2] Source: OpenI Report #3421
    "Small bilateral pleural effusions with associated bibasilar
     atelectasis."

[3] Source: ACR Guidelines — Chest Imaging
    "Pleural effusion is best confirmed on lateral decubitus view..."
```

---

## 4. Generation — Qwen2-VL

> **The brain of the system.** Takes evidence + image + question and generates a grounded answer.

The diagram shows **two parallel paths** that merge at the **LLM Decoder (Fusion Point)**.

---

### Text Path (top row)

### Block 16: Retrieved Evidence → Tokenizer (Qwen2-VL)

```
What:   Converts the formatted evidence text into token IDs
Input:  Evidence string from Context Builder + user question
Output: Token ID sequence
Why:    Neural networks work with numbers, not text
```

The tokenizer splits text into subword tokens:
```
"Pleural effusion" → [1234, 5678]  (2 tokens)
"costophrenic angle" → [8901, 2345, 6789]  (3 tokens)
```

### Block 17: Text Embeddings — Token to 4096-dim

```
What:   Converts token IDs into dense vector representations
Input:  Token ID sequence
Output: Sequence of 4096-dimensional vectors (one per token)
Why:    The LLM decoder operates on continuous vectors, not discrete IDs
```

Each token ID is looked up in an embedding table → a 4096-dimensional vector that captures the token's meaning.

---

### Image Path (bottom row)

### Block 18: User Image → Vision Tower (Frozen ViT · Patch Embeddings)

```
What:   Encodes the medical image into patch-level features
Input:  Raw medical image (e.g., chest X-ray PNG)
Output: Grid of patch embeddings
Why:    Converts visual information into a format the LLM can understand
```

**How it works:**
1. Resize image to model's expected resolution
2. Split into non-overlapping patches (e.g., 14×14 pixel patches)
3. Each patch is encoded by the Vision Transformer (ViT)
4. Output: a sequence of patch embedding vectors

**Frozen** means the vision tower's weights are NOT updated during fine-tuning — its visual understanding is already strong from pre-training.

### Block 19: Vision Projector — MLP · 256 Visual Tokens

```
What:   Projects visual features into the LLM's embedding space
Input:  Patch embeddings from Vision Tower
Output: 256 visual tokens (each 4096-dim, matching text embedding size)
Why:    Vision Tower outputs are in "vision space"; the LLM needs them in "language space"
```

The projector is a Multi-Layer Perceptron (MLP) that:
- Takes vision embeddings (e.g., 1024-dim from ViT)
- Maps them to 4096-dim (matching text embeddings)
- Optionally compresses: 576 patches → 256 visual tokens (via pooling/selection)

After this step, visual tokens and text tokens are in the **same vector space** and can be concatenated.

---

### Fusion and Generation

### Block 20: LLM Decoder — Fusion Point

```
What:   Merges visual tokens and text tokens into one unified sequence
Input:  256 visual tokens + text embedding tokens
Output: Fused sequence ready for autoregressive generation
Why:    This is where the model "sees" both the image and the evidence simultaneously
```

The fusion is simple concatenation:
```
[256 visual tokens] + [evidence text tokens] + [question tokens]
= one long sequence that the LLM processes with self-attention
```

Every token can attend to every other token — so the model can cross-reference what it sees in the image with what the evidence says.

### Block 21: Multimodal Reasoning — Joint Visual-Text Attention

```
What:   The LLM generates answer tokens using joint attention over image + text
Input:  Fused visual-text sequence
Output: Generated answer tokens
Why:    This is the actual "thinking" step — the model reasons across modalities
```

**How joint visual-text attention works:**

At each generation step, the model:
1. Looks at ALL visual tokens (what does the image show?)
2. Looks at ALL evidence tokens (what does the literature say?)
3. Looks at ALL previously generated tokens (what have I said so far?)
4. Predicts the next most likely token

This cross-modal attention is what makes the answer **grounded** — the model doesn't just generate from memory; it actively references the visual features and retrieved evidence.

### Block 22: Grounded Answer — Generation · Cited Sources · Confidence

```
What:   Post-processes the generated text to add citations and confidence
Input:  Raw generated answer
Output: Grounded answer with source citations and confidence score
Why:    Doctors need to verify AI answers — citations make this possible
```

**Three outputs:**
1. **Answer text:** The clinical response
2. **Cited sources:** Which evidence pieces the answer is based on
3. **Confidence:** How certain the model is (HIGH/MEDIUM/LOW)

**Example output:**
```
Answer:     "Small bilateral pleural effusions are present, with associated
             bibasilar atelectasis [1][2]."
Sources:    [1] OpenI Report #3421, [2] Harrison's Radiology Ch.5
Confidence: HIGH (0.91)
```

### Block 23: Final Output

The complete response delivered to the user — answer + citations + confidence.

---

## 5. Key Concepts Explained

### LLaVA-Med vs. Qwen2-VL

The diagram shows **Qwen2-VL** as the generation model (top-right label). In our Phase 1 implementation, we used **LLaVA-1.5-7B** as a stepping stone:

| Aspect | LLaVA-1.5 (Phase 1) | Qwen2-VL (Final Architecture) |
|---|---|---|
| Vision encoder | CLIP ViT-L/14 | ViT (native) |
| LLM backbone | Llama-2-7B | Qwen2-7B |
| Visual tokens | 576 | 256 (more efficient) |
| Embedding dim | 4096 | 4096 |
| Status | Implemented + tested | Planned for future phases |

**LLaVA-Med** is a medical-domain variant. Our QLoRA fine-tuning on OpenI gives similar medical specialization.

### MedRAG

A medical RAG framework that provides:
- Curated medical corpora (PubMed, textbooks)
- Domain-specific chunking (respects medical document structure)
- Medical-tuned embeddings
- **Where in our architecture:** Influences the Offline Pipeline design — how we chunk, index, and store medical documents.

### MMed-RAG (Multimodal Medical RAG)

Extends RAG to handle images alongside text:
- Retrieves both text chunks AND similar medical images
- Uses vision-language models for generation
- **Where in our architecture:** This IS our architecture — the three retrieval paths (ColQwen2 + BM25 + CLIP) feeding into a multimodal generator is the MMed-RAG paradigm.

### RAG-Anything

The concept of handling ANY input modality (text, images, tables, PDFs). Our system handles:
- Chest X-ray images (visual)
- Clinical questions (text)
- Radiology reports (text evidence)
- Similar past cases (visual evidence)

---

## 6. Complete Example Walkthrough

### Scenario

```
Doctor uploads:  chest_xray_042.png
Doctor asks:     "Is there pleural effusion?"
```

### Step 1 — Offline (already done)
```
3,955 OpenI reports + 7,470 X-rays → indexed into:
  - ColQwen2 index (layout-aware page vectors)
  - BM25 index (keyword inverted index)
  - CLIP index (image vectors)
  - SQLite metadata store
```

### Step 2 — Query Understanding
```
Input:  "Is there pleural effusion?"
NER:    entities = ["pleural effusion"]
Intent: "yes/no finding question"
Output: structured query with medical focus
```

### Step 3 — Three Parallel Retrievals
```
ColQwen2: searches document pages → top 20 pages about pleural effusion
BM25:     searches "pleural" + "effusion" → top 20 keyword-matched chunks
CLIP:     encodes chest_xray_042.png → finds 10 visually similar past X-rays
```

### Step 4 — RRF Fusion
```
ColQwen2 results + BM25 results + CLIP results
  → RRF formula → combined ranked list
  → page-ID alignment merges overlapping results
```

### Step 5 — Reranking
```
Top 20 from RRF → cross-encoder reads each (query, document) pair
  → re-scores for precise relevance
  → top 5 evidence pieces selected
```

### Step 6 — Context Building
```
Top 5 evidence + metadata from SQLite:

[1] "Pleural effusion appears as blunting of costophrenic angles..."
    — Harrison's Radiology Ch.5
[2] "Small right pleural effusion with meniscus sign..."  
    — OpenI Report #3421 (similar X-ray)
[3] "Bilateral effusions associated with heart failure..."
    — ACR Guidelines
```

### Step 7 — Text Path
```
Evidence text → Qwen2-VL Tokenizer → token IDs → Text Embeddings [N, 4096]
```

### Step 8 — Image Path
```
chest_xray_042.png → Vision Tower (frozen ViT) → patch embeddings
  → Vision Projector (MLP) → 256 visual tokens [256, 4096]
```

### Step 9 — Fusion + Generation
```
[256 visual tokens] + [evidence tokens] + [question tokens]
  → LLM Decoder (fusion point)
  → Joint visual-text attention
  → Generates: "Small right pleural effusion is present,
     with blunting of the right costophrenic angle [1][2]."
```

### Step 10 — Final Output
```
Answer:     "Small right pleural effusion is present, with blunting
             of the right costophrenic angle."
Sources:    [1] Harrison's Radiology Ch.5
            [2] OpenI Report #3421
Confidence: HIGH (0.93)
```

---

## 7. Why Each Technique Was Chosen

| Component | Choice | Why This Over Alternatives |
|---|---|---|
| Page understanding | **ColQwen2** | Preserves layout — tables, figures, multi-column. Traditional embeddings flatten everything. |
| Keyword search | **BM25** | Zero GPU cost, catches exact medical terms, complement to neural search |
| Image search | **CLIP** | Maps images and text to same space, enables cross-modal retrieval |
| Fusion | **RRF** | Parameter-free, robust, works across different score scales |
| Reranking | **Cross-encoder** | Reads query+doc together for precise relevance, worth the cost on top-K |
| Generation | **Qwen2-VL** | State-of-the-art open VLM, native multimodal, 4096-dim embeddings |
| Fine-tuning | **QLoRA** | Train 7B model on single T4 GPU, <1% params, 76 MB adapter |
| Quantization | **NF4 4-bit** | 14 GB → 4 GB VRAM, minimal quality loss |
| Metadata | **SQLite** | Lightweight, no server needed, perfect for citation tracking |

---

## 8. Summary in Simple Language

> **The system works like a very diligent medical resident:**
>
> 1. A doctor shows it an X-ray and asks a question
> 2. Before answering, it searches through textbooks, past reports, and similar X-rays
> 3. It reads the most relevant evidence carefully
> 4. It looks at the actual X-ray image, cross-referencing with the evidence
> 5. It writes a brief, precise answer — citing which sources it used
> 6. It tells the doctor how confident it is
>
> The key difference from a regular AI chatbot: **it never answers from memory alone** — it always checks the evidence first. This is what makes it safe for healthcare.
