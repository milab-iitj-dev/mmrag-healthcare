"""
Phase 2 Validation: ColQwen2 Retrieval + LLaVA Generation — Kaggle GPU.
FINAL HARDENED VERSION — Production-stable for Kaggle T4/P100.

COMPLETE END-TO-END VALIDATION of the Phase 2 RAG pipeline on OpenI.

SETUP:
  1. Kaggle notebook -> GPU T4 x2 or P100 -> Internet ON
  2. Add dataset: "OpenI Chest X-rays Indiana University"
  3. (Optional) Add your Phase 1 LoRA adapter dataset
  4. Paste this entire script into one cell -> Run
  5. Download results from /kaggle/working/results/

PIPELINE:
  OFFLINE:  OpenI image-report pairs -> ColQwen2 encoding -> saved index
  ONLINE:   User query -> ColQwen2 retrieval -> top 3 -> context -> LLaVA -> answer

OUTPUT FILES:
  results/phase2_results_{timestamp}.json   - structured results
  results/phase2_results_{timestamp}.csv    - spreadsheet-friendly
  results/phase2_report_{timestamp}.md      - markdown report
  results/console_output_{timestamp}.txt    - full console log
  colqwen2_index/                           - saved retrieval index
"""

# ============================================================
# 1. Environment Setup
# ============================================================
import os, subprocess, sys, io

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def _pip(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

print("=" * 60)
print("PHASE 2 VALIDATION - Installing dependencies...")
print("=" * 60)

_pip("transformers>=4.46.0")
_pip("accelerate>=0.27.0")
_pip("bitsandbytes>=0.43.0")
_pip("peft>=0.10.0")
_pip("colpali-engine>=0.3.0")
_pip("tqdm")

# ── HuggingFace Authentication ──
# Priority: Kaggle secret > env var > hardcoded fallback
_hf_token = None
try:
    from kaggle_secrets import UserSecretsClient
    _hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF auth: using Kaggle secret")
except Exception:
    pass

if not _hf_token:
    _hf_token = os.environ.get("HF_TOKEN", os.environ.get("HUGGINGFACE_TOKEN"))
    if _hf_token:
        print("HF auth: using environment variable")

if not _hf_token:
    _hf_token = "YOUR_HF_TOKEN_HERE"  # Replace with your HuggingFace token
    print("HF auth: using fallback token (replace YOUR_HF_TOKEN_HERE with your actual token)")

try:
    from huggingface_hub import login
    login(token=_hf_token)
    print("HuggingFace authentication successful")
except Exception as e:
    print(f"HuggingFace login note: {e}")

# ============================================================
# 2. Imports
# ============================================================
import csv, re, json, time, gc, logging, traceback
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import torch
from PIL import Image
from tqdm import tqdm

# ============================================================
# 3. Logging & Output Structure
# ============================================================
BASE_DIR = Path("/kaggle/working")
INDEX_DIR = BASE_DIR / "colqwen2_index"
RESULTS_DIR = BASE_DIR / "results"
EVAL_DIR = BASE_DIR / "evaluation"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"
DOCS_DIR = BASE_DIR / "docs"

for d in [INDEX_DIR, RESULTS_DIR, EVAL_DIR, REPORTS_DIR, LOGS_DIR, DOCS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

log_buffer = io.StringIO()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(log_buffer),
    ],
)
logger = logging.getLogger("phase2")

# ============================================================
# 4. Configuration
# ============================================================

# --- LLaVA ---
LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
LLAVA_GEN_CONFIG = {
    "max_new_tokens": 128,
    "do_sample": False,
    "repetition_penalty": 1.3,
    "no_repeat_ngram_size": 3,
}

# --- ColQwen2 ---
COLQWEN2_MODEL_ID = "vidore/colqwen2-v1.0-hf"
COLQWEN2_BATCH_SIZE = 2          # conservative for T4 (16 GB VRAM)

# --- Retrieval ---
TOP_K = 3
MAX_INDEX_SAMPLES = None         # None = all, set to 50 for quick test

# --- Context budget ---
MAX_CONTEXT_CHARS = 3000         # hard cap on context fed to LLaVA
MAX_EVIDENCE_CHARS = 800         # per-evidence block cap

# --- LLaVA adapter paths ---
ADAPTER_PATHS = [
    "/kaggle/input/llava-medical-adapter/final_adapter",
    "/kaggle/input/llava-medical-adapter",
    "/kaggle/working/llava-medical-vqa/final_adapter",
]

# --- OpenI dataset paths ---
DATASET_ROOTS = [
    "/kaggle/input/datasets/raddar/chest-xrays-indiana-university",
    "/kaggle/input/chest-xrays-indiana-university",
    "/kaggle/input/openi-chest-xray",
]

# --- Evaluation queries ---
EVAL_QUERIES = [
    {"query": "What are the key findings in this chest X-ray?",
     "type": "image_text"},
    {"query": "Is there cardiomegaly or any cardiac abnormality?",
     "type": "image_text"},
    {"query": "Are there signs of pleural effusion or pneumonia?",
     "type": "image_text"},
]


# ============================================================
# 5. VRAM Utilities
# ============================================================
def vram_gb():
    """Current GPU memory allocated in GB."""
    if torch.cuda.is_available():
        return round(torch.cuda.memory_allocated() / 1e9, 2)
    return 0.0


def vram_free_gb():
    """Free GPU memory in GB."""
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        used = torch.cuda.memory_allocated() / 1e9
        return round(total - used, 2)
    return 0.0


def flush_vram():
    """Force garbage collection and CUDA cache clearing."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ============================================================
# 6. Path Discovery
# ============================================================
def find_adapter():
    """Locate the LoRA adapter directory."""
    for p in ADAPTER_PATHS:
        path = Path(p)
        if path.exists() and (path / "adapter_config.json").exists():
            return str(path)
        for sub in path.glob("*/adapter_config.json"):
            return str(sub.parent)
    for cfg in Path("/kaggle/input").rglob("adapter_config.json"):
        return str(cfg.parent)
    return None


def find_dataset():
    """Locate OpenI images and reports CSV on Kaggle."""
    for root in DATASET_ROOTS:
        p = Path(root)
        if not p.exists():
            continue
        for img_dir_name in ["images_normalized", "images"]:
            for d in p.rglob(img_dir_name):
                if d.is_dir() and any(d.glob("*.png")):
                    reports_csv = next(p.rglob("indiana_reports.csv"), None)
                    projections_csv = next(p.rglob("indiana_projections.csv"), None)
                    return {
                        "images_dir": d,
                        "reports_csv": reports_csv,
                        "projections_csv": projections_csv,
                    }

    # Fallback: scan all of /kaggle/input
    for d in Path("/kaggle/input").rglob("images_normalized"):
        if d.is_dir():
            reports = next(Path("/kaggle/input").rglob("indiana_reports.csv"), None)
            projections = next(Path("/kaggle/input").rglob("indiana_projections.csv"), None)
            return {"images_dir": d, "reports_csv": reports, "projections_csv": projections}

    return None


# ============================================================
# 7. Data Loading (self-contained, no src/ dependency)
# ============================================================
@dataclass
class MedicalCase:
    """One OpenI image-report pair = one retrieval unit."""
    case_id: str
    image_path: str
    findings: Optional[str] = None
    impression: Optional[str] = None
    report: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


def load_openi_cases(dataset_info, max_samples=None):
    """Load OpenI cases from CSV files."""
    images_dir = dataset_info["images_dir"]
    reports_csv = dataset_info.get("reports_csv")
    projections_csv = dataset_info.get("projections_csv")

    if not reports_csv or not reports_csv.exists():
        logger.error("Reports CSV not found")
        return []

    # UID -> image filename mapping
    uid_to_images = {}
    if projections_csv and projections_csv.exists():
        with open(projections_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                uid = row.get("uid", "").strip()
                filename = row.get("filename", "").strip()
                projection = row.get("projection", "").strip().lower()
                if uid and filename:
                    uid_to_images.setdefault(uid, []).append(
                        {"filename": filename, "projection": projection}
                    )
        logger.info(f"Loaded projections for {len(uid_to_images)} UIDs")

    # Parse reports
    cases = []
    skipped_text, skipped_image = 0, 0

    with open(reports_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if max_samples and len(cases) >= max_samples:
                break

            uid = row.get("uid", "").strip()
            findings = _clean_text(row.get("findings", ""))
            impression = _clean_text(row.get("impression", ""))

            if not findings and not impression:
                skipped_text += 1
                continue

            image_path = _resolve_image(uid, uid_to_images, images_dir)
            if image_path is None:
                skipped_image += 1
                continue

            # Validate image can actually be opened
            try:
                img = Image.open(image_path)
                img.verify()  # verify but don't fully load
            except Exception:
                skipped_image += 1
                continue

            report = ""
            if findings:
                report += f"FINDINGS: {findings} "
            if impression:
                report += f"IMPRESSION: {impression}"

            metadata = {}
            mesh = row.get("MeSH", "").strip()
            if mesh:
                metadata["mesh_terms"] = [t.strip() for t in mesh.split(";") if t.strip()]
            problems = row.get("Problems", "").strip()
            if problems:
                metadata["problems"] = [p.strip() for p in problems.split(";") if p.strip()]

            cases.append(MedicalCase(
                case_id=uid,
                image_path=str(image_path),
                findings=findings,
                impression=impression,
                report=report.strip(),
                metadata=metadata,
            ))

    logger.info(
        f"Loaded {len(cases)} cases "
        f"(skipped: {skipped_text} no text, {skipped_image} no image)"
    )
    return cases


def _clean_text(text):
    if not text or not text.strip():
        return None
    text = text.strip().replace("XXXX", "").replace("xxxx", "")
    text = " ".join(text.split())
    text = text.strip(" .,;:")
    return text if len(text) >= 5 else None


def _resolve_image(uid, uid_to_images, images_dir):
    """Resolve UID to image path, preferring frontal projection."""
    if uid in uid_to_images:
        entries = uid_to_images[uid]
        frontal = [e for e in entries if e["projection"] == "frontal"]
        candidates = frontal if frontal else entries
        for entry in candidates:
            path = images_dir / entry["filename"]
            if path.exists():
                return path
    matches = sorted(images_dir.glob(f"{uid}_*"))
    return matches[0] if matches else None


# ============================================================
# 8. Safe Embedding Extraction
# ============================================================
def _extract_embeddings(outputs, context=""):
    """
    Safely extract the embedding tensor from ColQwen2 model outputs.

    HuggingFace output objects vary across versions:
      - transformers native: outputs.embeddings
      - colpali-engine:      raw tensor or outputs.reps
      - fallback:            outputs.last_hidden_state

    Args:
        outputs: Model forward pass output (object or tensor).
        context: Description string for error messages.

    Returns:
        torch.Tensor of shape [batch, seq_len, embed_dim].
    """
    # Case 1: already a raw tensor (some versions return this)
    if isinstance(outputs, torch.Tensor):
        logger.debug(f"  [{context}] Output is raw tensor: {outputs.shape}")
        return outputs

    # Case 2: HF-native ColQwen2ForRetrievalOutput
    if hasattr(outputs, "embeddings") and outputs.embeddings is not None:
        emb = outputs.embeddings
        logger.debug(f"  [{context}] Extracted .embeddings: {emb.shape}")
        return emb

    # Case 3: colpali-engine style
    if hasattr(outputs, "reps") and outputs.reps is not None:
        emb = outputs.reps
        logger.debug(f"  [{context}] Extracted .reps: {emb.shape}")
        return emb

    # Case 4: generic transformer output
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        emb = outputs.last_hidden_state
        logger.debug(f"  [{context}] Extracted .last_hidden_state: {emb.shape}")
        return emb

    # Case 5: tuple output (some older wrappers)
    if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        if isinstance(outputs[0], torch.Tensor):
            emb = outputs[0]
            logger.debug(f"  [{context}] Extracted from tuple[0]: {emb.shape}")
            return emb

    raise ValueError(
        f"Unknown ColQwen2 output format in {context}: "
        f"type={type(outputs)}, "
        f"attrs={[a for a in dir(outputs) if not a.startswith('_')]}"
    )


# ============================================================
# 9. ColQwen2 Embedder (self-contained, hardened)
# ============================================================
class ColQwen2Embedder:
    """
    ColQwen2 multi-vector encoder for document retrieval.

    Hardened for Kaggle execution with:
      - Safe embedding extraction (multi-version compatible)
      - VRAM monitoring
      - Tensor shape validation
      - Graceful error handling
    """

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self._loaded = False

    def load(self, model_id=COLQWEN2_MODEL_ID):
        """Load ColQwen2 model with VRAM monitoring."""
        from transformers import ColQwen2ForRetrieval, ColQwen2Processor

        logger.info(f"Loading ColQwen2: {model_id}")
        logger.info(f"  VRAM before load: {vram_gb()} GB")

        self.processor = ColQwen2Processor.from_pretrained(model_id)
        logger.info("  Processor loaded")

        self.model = ColQwen2ForRetrieval.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()

        self.device = self.model.device
        self._loaded = True

        logger.info(f"  Model loaded on {self.device}")
        logger.info(f"  VRAM after load: {vram_gb()} GB (free: {vram_free_gb()} GB)")

    def encode_images(self, images, batch_size=COLQWEN2_BATCH_SIZE):
        """Encode document images -> list of multi-vector tensors."""
        all_embeddings = []
        n_batches = (len(images) + batch_size - 1) // batch_size

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_num = i // batch_size + 1
            logger.info(
                f"  Encoding image batch {batch_num}/{n_batches} "
                f"({len(batch)} images) | VRAM: {vram_gb()} GB"
            )

            # Use process_images if available (ColQwen2-specific)
            try:
                inputs = self.processor(images=batch, return_tensors="pt")
            except Exception as e:
                logger.error(f"  Processor failed on batch {batch_num}: {e}")
                continue

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            embeddings = _extract_embeddings(outputs, context=f"encode_images batch {batch_num}")

            # Validate shape: expect [batch_size, n_patches, embed_dim]
            if embeddings.dim() != 3:
                logger.warning(
                    f"  Unexpected embedding dim: {embeddings.dim()} "
                    f"(shape: {embeddings.shape}), expected 3D"
                )

            for j in range(embeddings.shape[0]):
                emb = embeddings[j].cpu().float()  # store as float32 for stability
                all_embeddings.append(emb)

            # Free batch VRAM
            del inputs, outputs, embeddings
            flush_vram()

        logger.info(f"  Encoded {len(all_embeddings)} images total")

        # Validate all embeddings have consistent embed_dim
        if all_embeddings:
            dims = set(e.shape[-1] for e in all_embeddings)
            if len(dims) > 1:
                logger.warning(f"  Inconsistent embed dims: {dims}")
            else:
                logger.info(f"  Embedding dim: {dims.pop()}")

        return all_embeddings

    def encode_queries(self, queries, batch_size=COLQWEN2_BATCH_SIZE):
        """Encode text queries -> list of multi-vector tensors."""
        all_embeddings = []

        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]

            try:
                inputs = self.processor(
                    text=batch, return_tensors="pt",
                    padding=True, truncation=True,
                )
            except Exception as e:
                logger.error(f"  Query processor failed: {e}")
                continue

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            embeddings = _extract_embeddings(outputs, context="encode_queries")

            for j in range(embeddings.shape[0]):
                all_embeddings.append(embeddings[j].cpu().float())

            del inputs, outputs, embeddings
            flush_vram()

        if not all_embeddings:
            logger.error("  No query embeddings produced!")

        return all_embeddings

    def encode_image_queries(self, images, queries, batch_size=COLQWEN2_BATCH_SIZE):
        """Encode image+text queries -> list of multi-vector tensors."""
        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i + batch_size]
            batch_qs = queries[i:i + batch_size]

            try:
                inputs = self.processor(
                    images=batch_imgs, text=batch_qs,
                    return_tensors="pt", padding=True, truncation=True,
                )
            except Exception:
                logger.warning("Joint image+text encoding failed, falling back to image-only")
                try:
                    inputs = self.processor(
                        images=batch_imgs, return_tensors="pt",
                    )
                except Exception as e:
                    logger.error(f"  Image-only fallback also failed: {e}")
                    continue

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            embeddings = _extract_embeddings(outputs, context="encode_image_queries")

            for j in range(embeddings.shape[0]):
                all_embeddings.append(embeddings[j].cpu().float())

            del inputs, outputs, embeddings
            flush_vram()

        return all_embeddings

    def score(self, query_embs, doc_embs):
        """
        Compute MaxSim late-interaction scores [n_queries, n_docs].

        MaxSim (ColBERT-style):
          score(q, d) = sum_i( max_j( cos_sim(q_i, d_j) ) )

        For each query token, find max cosine similarity to any doc token,
        then sum across all query tokens.
        """
        if not query_embs or not doc_embs:
            logger.warning("Empty embeddings passed to score()")
            return torch.zeros(max(len(query_embs), 1), max(len(doc_embs), 1))

        n_queries = len(query_embs)
        n_docs = len(doc_embs)

        logger.info(f"  MaxSim scoring: {n_queries} queries x {n_docs} docs")
        scores = torch.zeros(n_queries, n_docs)

        for qi in range(n_queries):
            q = query_embs[qi].to(self.device).float()
            q = torch.nn.functional.normalize(q, p=2, dim=-1)

            for di in range(n_docs):
                d = doc_embs[di].to(self.device).float()
                d = torch.nn.functional.normalize(d, p=2, dim=-1)

                # [n_q_tokens, n_d_tokens] cosine similarity
                sim_matrix = torch.matmul(q, d.transpose(0, 1))

                # MaxSim: per query token, take max over doc tokens, then sum
                max_sim = sim_matrix.max(dim=-1).values
                scores[qi, di] = max_sim.sum().item()

            # Log progress for large indexes
            if n_docs > 20 and (qi + 1) % max(1, n_queries // 5) == 0:
                logger.info(f"    Query {qi+1}/{n_queries} scored")

        logger.info(
            f"  Scores range: [{scores.min():.2f}, {scores.max():.2f}]"
        )

        flush_vram()
        return scores

    def unload(self):
        """Free VRAM completely."""
        logger.info(f"  Unloading ColQwen2 | VRAM before: {vram_gb()} GB")
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._loaded = False
        flush_vram()
        logger.info(f"  ColQwen2 unloaded | VRAM after: {vram_gb()} GB")


def _pad_and_stack(tensors):
    """Pad variable-length tensors and stack into [B, max_len, dim]."""
    max_len = max(t.shape[0] for t in tensors)
    dim = tensors[0].shape[1]
    padded = torch.zeros(len(tensors), max_len, dim, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        padded[i, :t.shape[0], :] = t
    return padded


# ============================================================
# 10. Offline Indexing — Chunked, Streaming, Resumable
# ============================================================
CHUNK_SIZE = 100          # images per chunk saved to disk
INDEX_BATCH_SIZE = 1      # images per ColQwen2 forward pass (safest)

def _chunk_dir(index_dir):
    """Return the path to the temporary chunk directory."""
    p = Path(index_dir) / "_chunks"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _completed_chunks(index_dir):
    """Return set of chunk indices that have already been saved."""
    chunk_d = _chunk_dir(index_dir)
    done = set()
    for f in chunk_d.glob("chunk_*.pt"):
        try:
            idx = int(f.stem.split("_")[1])
            done.add(idx)
        except (ValueError, IndexError):
            pass
    return done


def _validate_case_image(case):
    """
    Open, validate, and RETURN a PIL image for a single case.
    Returns (image, True) on success, (None, False) on failure.
    Image is opened fresh and will be closed after encoding.
    """
    try:
        img = Image.open(case.image_path).convert("RGB")
        if img.size[0] < 10 or img.size[1] < 10:
            return None, False
        return img, True
    except Exception:
        return None, False


def build_colqwen2_index(cases, embedder, index_dir):
    """
    OFFLINE STAGE: Build ColQwen2 index from OpenI cases.

    MEMORY-SAFE IMPLEMENTATION:
      - Processes images in chunks of CHUNK_SIZE (default 100)
      - Saves each chunk of embeddings to disk immediately
      - Clears memory after every chunk
      - Supports RESUME: skips already-completed chunks
      - Merges all chunks into final index at the end
      - Uses batch_size=1 for maximum GPU stability

    This allows full 3826-image indexing on a Kaggle T4 (16 GB).
    """
    logger.info("=" * 60)
    logger.info("OFFLINE STAGE: Building ColQwen2 Index (Chunked/Resumable)")
    logger.info("=" * 60)
    t0 = time.time()

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    chunk_d = _chunk_dir(index_dir)

    # --- Step 1: Pre-validate all cases (lightweight, no image loading) ---
    logger.info("Step 1: Validating case image paths...")
    valid_cases = []
    for case in cases:
        p = Path(case.image_path)
        if p.exists() and p.stat().st_size > 100:
            valid_cases.append(case)
    logger.info(f"  {len(valid_cases)} / {len(cases)} cases have valid image paths")

    if not valid_cases:
        logger.error("No valid cases found! Cannot build index.")
        return [], [], {}

    # --- Step 2: Determine which chunks are already done (resume support) ---
    total_chunks = (len(valid_cases) + CHUNK_SIZE - 1) // CHUNK_SIZE
    done_chunks = _completed_chunks(index_dir)
    logger.info(f"  Total chunks needed: {total_chunks}  (CHUNK_SIZE={CHUNK_SIZE})")
    if done_chunks:
        logger.info(f"  Resuming: {len(done_chunks)} chunks already completed")

    # --- Step 3: Process each chunk ---
    skipped_images = 0

    for chunk_idx in range(total_chunks):
        if chunk_idx in done_chunks:
            logger.info(
                f"  Chunk {chunk_idx + 1}/{total_chunks}: ALREADY DONE (skipping)"
            )
            continue

        start = chunk_idx * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, len(valid_cases))
        chunk_cases = valid_cases[start:end]

        logger.info(
            f"\n  ── Chunk {chunk_idx + 1}/{total_chunks} "
            f"(cases {start + 1}..{end}) ──"
        )
        logger.info(f"  VRAM: {vram_gb()} GB used, {vram_free_gb()} GB free")

        chunk_embeddings = []
        chunk_ids = []

        for ci, case in enumerate(chunk_cases):
            # Load image fresh for this one case
            img, ok = _validate_case_image(case)
            if not ok:
                skipped_images += 1
                continue

            try:
                # Process single image through ColQwen2
                inputs = embedder.processor(
                    images=[img], return_tensors="pt",
                )
                inputs = {k: v.to(embedder.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = embedder.model(**inputs)

                emb_tensor = _extract_embeddings(
                    outputs, context=f"chunk{chunk_idx}_img{ci}"
                )

                # Store on CPU as float32
                for j in range(emb_tensor.shape[0]):
                    chunk_embeddings.append(emb_tensor[j].cpu().float())
                    chunk_ids.append(case.case_id)

            except Exception as e:
                logger.warning(
                    f"    Failed {case.case_id}: {e}"
                )
                skipped_images += 1

            finally:
                # AGGRESSIVE CLEANUP — release everything immediately
                del img
                try:
                    del inputs, outputs, emb_tensor
                except NameError:
                    pass
                if (ci + 1) % 10 == 0 or ci == len(chunk_cases) - 1:
                    flush_vram()

            # Progress logging every 20 images
            if (ci + 1) % 20 == 0 or ci == len(chunk_cases) - 1:
                logger.info(
                    f"    {ci + 1}/{len(chunk_cases)} images encoded "
                    f"| VRAM: {vram_gb()} GB"
                )

        # --- Save chunk to disk immediately ---
        if chunk_embeddings:
            chunk_path = chunk_d / f"chunk_{chunk_idx}.pt"
            torch.save(chunk_embeddings, str(chunk_path))

            ids_path = chunk_d / f"chunk_{chunk_idx}_ids.json"
            with open(ids_path, "w") as f:
                json.dump(chunk_ids, f)

            logger.info(
                f"  ✓ Chunk {chunk_idx + 1} saved: "
                f"{len(chunk_embeddings)} embeddings → {chunk_path.name}"
            )
        else:
            logger.warning(f"  Chunk {chunk_idx + 1}: no embeddings produced")

        # --- Free chunk memory ---
        del chunk_embeddings, chunk_ids, chunk_cases
        flush_vram()
        gc.collect()

    # --- Step 4: Merge all chunks into final index ---
    logger.info("\n" + "=" * 60)
    logger.info("Merging chunks into final index...")
    logger.info("=" * 60)

    all_embeddings = []
    all_doc_ids = []

    for chunk_idx in range(total_chunks):
        chunk_path = chunk_d / f"chunk_{chunk_idx}.pt"
        ids_path = chunk_d / f"chunk_{chunk_idx}_ids.json"

        if not chunk_path.exists():
            logger.warning(f"  Chunk {chunk_idx} missing, skipping")
            continue

        chunk_embs = torch.load(str(chunk_path), map_location="cpu")
        with open(ids_path, "r") as f:
            chunk_ids = json.load(f)

        all_embeddings.extend(chunk_embs)
        all_doc_ids.extend(chunk_ids)

        del chunk_embs, chunk_ids
        gc.collect()

        logger.info(
            f"  Merged chunk {chunk_idx + 1}/{total_chunks} "
            f"(total so far: {len(all_embeddings)})"
        )

    logger.info(f"  Total embeddings merged: {len(all_embeddings)}")

    # --- Step 5: Save final consolidated index ---
    logger.info("Saving final consolidated index...")

    torch.save(all_embeddings, str(index_dir / "embeddings.pt"))

    with open(index_dir / "doc_ids.json", "w") as f:
        json.dump(all_doc_ids, f, indent=2)

    # Build case lookup for document store
    case_lookup = {c.case_id: c for c in valid_cases}
    doc_store = {}
    final_cases = []
    for doc_id in all_doc_ids:
        case = case_lookup.get(doc_id)
        if case:
            doc_store[doc_id] = {
                "case_id": case.case_id,
                "image_path": case.image_path,
                "findings": case.findings,
                "impression": case.impression,
                "report": case.report,
                "metadata": case.metadata,
            }
            final_cases.append(case)

    with open(index_dir / "document_store.json", "w") as f:
        json.dump({"version": "2.0", "num_documents": len(doc_store),
                    "documents": doc_store}, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    build_meta = {
        "build_timestamp": datetime.now().isoformat(),
        "model": COLQWEN2_MODEL_ID,
        "num_indexed": len(all_embeddings),
        "num_skipped": skipped_images + (len(cases) - len(valid_cases)),
        "build_time_seconds": round(elapsed, 2),
        "embedding_dim": all_embeddings[0].shape[-1] if all_embeddings else 0,
        "avg_seq_len": round(
            sum(e.shape[0] for e in all_embeddings) / max(len(all_embeddings), 1), 1
        ),
        "chunk_size": CHUNK_SIZE,
        "total_chunks": total_chunks,
    }
    with open(index_dir / "index_metadata.json", "w") as f:
        json.dump(build_meta, f, indent=2)

    logger.info(f"\nIndex saved to {index_dir}")
    logger.info(f"  Documents indexed: {len(all_embeddings)}")
    logger.info(f"  Skipped (invalid): {build_meta['num_skipped']}")
    logger.info(f"  Embedding dim: {build_meta['embedding_dim']}")
    logger.info(f"  Avg seq length: {build_meta['avg_seq_len']}")
    logger.info(f"  Chunks used: {total_chunks} x {CHUNK_SIZE}")
    logger.info(f"  Build time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    return all_embeddings, final_cases, build_meta


# ============================================================
# 11. Online Retrieval
# ============================================================
def retrieve_top_k(
    query, embedder, embeddings, cases,
    query_image=None, top_k=TOP_K,
):
    """
    ONLINE STAGE: Retrieve top-k cases for a query.

    Supports text-only and image+text queries.
    Includes retrieval score validation.
    """
    if not embeddings:
        logger.warning("Empty index, cannot retrieve")
        return []

    if not cases:
        logger.warning("No cases in index, cannot retrieve")
        return []

    # Encode query
    if query_image is not None:
        logger.info("Retrieval mode: image + text")
        query_embs = embedder.encode_image_queries(
            images=[query_image], queries=[query],
        )
    else:
        logger.info("Retrieval mode: text-only")
        query_embs = embedder.encode_queries([query])

    # Validate query embedding was produced
    if not query_embs:
        logger.error("Failed to encode query! Returning empty results.")
        return []

    # Compute MaxSim scores
    scores = embedder.score(query_embs, embeddings)

    # Safe dimension handling: flatten to 1D
    if scores.dim() == 0:
        logger.warning("Scalar score returned, wrapping")
        scores = scores.unsqueeze(0)
    elif scores.dim() == 2:
        scores = scores[0]  # [1, n_docs] -> [n_docs]
    scores = scores.reshape(-1)  # ensure 1D

    # Sanity check
    if scores.shape[0] != len(cases):
        logger.warning(
            f"Score count ({scores.shape[0]}) != case count ({len(cases)}). "
            f"Using min."
        )
        n = min(scores.shape[0], len(cases))
        scores = scores[:n]
        cases = cases[:n]

    # Top-k
    k = min(top_k, len(cases))
    top_scores, top_indices = torch.topk(scores, k=k)

    results = []
    for rank, (score_val, idx) in enumerate(zip(top_scores.tolist(), top_indices.tolist())):
        case = cases[idx]
        results.append({
            "rank": rank + 1,
            "case_id": case.case_id,
            "score": round(float(score_val), 4),
            "image_path": case.image_path,
            "findings": case.findings,
            "impression": case.impression,
            "report": case.report,
            "metadata": case.metadata,
        })

    logger.info(
        f"Retrieved {len(results)} cases: "
        f"{[(r['case_id'], r['score']) for r in results]}"
    )
    return results


# ============================================================
# 12. Context Builder (token-budget-aware)
# ============================================================
def build_context(retrieved_results):
    """
    Build structured, token-budget-aware context from retrieved cases.

    Ensures the total context stays within MAX_CONTEXT_CHARS to avoid
    overwhelming LLaVA's context window.
    """
    if not retrieved_results:
        return ""

    parts = ["=== Retrieved Medical Evidence ===\n"]
    total_chars = len(parts[0])

    for r in retrieved_results:
        block_parts = [f"--- Evidence #{r['rank']} ---"]
        block_parts.append(f"Case ID: {r['case_id']}")
        block_parts.append(f"Relevance: {r['score']}")

        if r.get("findings"):
            findings = r["findings"][:MAX_EVIDENCE_CHARS]
            block_parts.append(f"Findings: {findings}")

        if r.get("impression"):
            impression = r["impression"][:MAX_EVIDENCE_CHARS]
            block_parts.append(f"Impression: {impression}")

        # Only add full report if no findings/impression
        if not r.get("findings") and not r.get("impression") and r.get("report"):
            report = r["report"][:MAX_EVIDENCE_CHARS]
            block_parts.append(f"Report: {report}")

        mesh = r.get("metadata", {}).get("mesh_terms")
        if mesh:
            block_parts.append(f"MeSH: {'; '.join(mesh[:5])}")

        block_parts.append("")
        block = "\n".join(block_parts)

        # Token budget check: stop adding if we'd exceed the limit
        if total_chars + len(block) > MAX_CONTEXT_CHARS:
            logger.info(
                f"  Context budget reached at evidence #{r['rank']} "
                f"({total_chars} chars)"
            )
            break

        parts.append(block)
        total_chars += len(block)

    parts.append("=== End of Retrieved Evidence ===")

    context = "\n".join(parts)

    # Hard truncation safety net
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n[...truncated for token budget]"

    logger.info(f"  Context: {len(context)} chars, ~{len(context.split())} words")
    return context


# ============================================================
# 13. LLaVA Generation
# ============================================================
def load_llava(adapter_path=None):
    """Load LLaVA-1.5-7B with 4-bit quantization + optional LoRA adapter."""
    from transformers import LlavaForConditionalGeneration, BitsAndBytesConfig
    try:
        from transformers import LlavaProcessor
    except ImportError:
        from transformers import AutoProcessor as LlavaProcessor

    logger.info(f"Loading LLaVA: {LLAVA_MODEL_ID}")
    logger.info(f"  VRAM before: {vram_gb()} GB")

    processor = LlavaProcessor.from_pretrained(LLAVA_MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        LLAVA_MODEL_ID,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ),
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    logger.info(f"  LLaVA base loaded | VRAM: {vram_gb()} GB")

    if adapter_path:
        from peft import PeftModel
        logger.info(f"  Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        logger.info(f"  Adapter loaded | VRAM: {vram_gb()} GB")

    model.eval()
    return model, processor


def generate_answer(model, processor, image, question, context=""):
    """Generate answer with LLaVA, using retrieved context."""
    if isinstance(image, (str, Path)):
        try:
            image = Image.open(image).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return {"answer": f"[Image load error: {e}]", "time_sec": 0,
                    "input_tokens": 0, "output_tokens": 0}

    # Build prompt with token-budget-safe context
    if context:
        prompt = (
            f"USER: <image>\n"
            f"You are a medical imaging specialist. Use the following retrieved "
            f"clinical evidence from similar cases to help answer the question.\n"
            f"\n{context}\n\n"
            f"Based on the image and the retrieved evidence above, "
            f"answer the following question:\n"
            f"{question}\n"
            f"Provide a clinically relevant answer.\n"
            f"ASSISTANT:"
        )
    else:
        prompt = f"USER: <image>\n{question}\nASSISTANT:"

    try:
        inputs = processor(text=prompt, images=image, return_tensors="pt")
    except Exception as e:
        logger.error(f"LLaVA processor failed: {e}")
        return {"answer": f"[Processor error: {e}]", "time_sec": 0,
                "input_tokens": 0, "output_tokens": 0}

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    inputs = {
        k: v.to(device=device, dtype=dtype) if v.is_floating_point()
        else v.to(device=device)
        for k, v in inputs.items()
    }

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(**inputs, **LLAVA_GEN_CONFIG)
    elapsed = time.time() - t0

    input_len = inputs["input_ids"].shape[-1]
    generated = output_ids[0, input_len:]
    answer = processor.decode(generated, skip_special_tokens=True).strip()

    # Clean repetition
    answer = re.sub(r'(\. ){2,}', '. ', answer)
    answer = re.sub(r'(\.){3,}', '.', answer)
    answer = re.sub(r'\s+', ' ', answer).strip()

    return {
        "answer": answer,
        "time_sec": round(elapsed, 2),
        "input_tokens": int(input_len),
        "output_tokens": int(len(generated)),
    }


# ============================================================
# 14. End-to-End RAG Pipeline
# ============================================================
def run_rag_query(
    query, embedder, embeddings, cases,
    llava_model, llava_processor,
    query_image=None, top_k=TOP_K,
):
    """
    Full RAG pipeline for one query:
      query -> ColQwen2 retrieval -> context -> LLaVA -> answer
    """
    t_total = time.time()

    # Step 1: Retrieve
    t_ret = time.time()
    retrieved = retrieve_top_k(
        query, embedder, embeddings, cases,
        query_image=query_image, top_k=top_k,
    )
    retrieval_time = time.time() - t_ret

    # Step 2: Build context
    context = build_context(retrieved)

    # Step 3: Select image for LLaVA
    if query_image is not None:
        llava_image = query_image
        image_source = "query"
    elif retrieved:
        try:
            llava_image = Image.open(retrieved[0]["image_path"]).convert("RGB")
            image_source = f"retrieved (case {retrieved[0]['case_id']})"
        except Exception as e:
            logger.error(f"Failed to load retrieved image: {e}")
            llava_image = None
            image_source = "none (load failed)"
    else:
        llava_image = None
        image_source = "none (empty retrieval)"

    # Step 4: Generate answer
    t_gen = time.time()
    if llava_image is not None:
        gen_result = generate_answer(
            llava_model, llava_processor,
            llava_image, query, context,
        )
    else:
        gen_result = {
            "answer": "[No image available for generation]",
            "time_sec": 0, "input_tokens": 0, "output_tokens": 0,
        }
    generation_time = time.time() - t_gen

    total_time = time.time() - t_total

    return {
        "query": query,
        "query_type": "image_text" if query_image else "text_only",
        "answer": gen_result["answer"],
        "retrieved": retrieved,
        "context_length": len(context),
        "image_source": image_source,
        "retrieval_time_sec": round(retrieval_time, 2),
        "generation_time_sec": round(generation_time, 2),
        "total_time_sec": round(total_time, 2),
        "input_tokens": gen_result["input_tokens"],
        "output_tokens": gen_result["output_tokens"],
    }


# ============================================================
# 15. Full Evaluation
# ============================================================
def run_full_evaluation(
    embedder, embeddings, cases,
    llava_model, llava_processor,
    dataset_info,
):
    """Run all evaluation queries (text-only and image+text)."""
    logger.info("=" * 60)
    logger.info("PHASE 2 EVALUATION - RAG Pipeline")
    logger.info("=" * 60)

    results = []

    for i, q in enumerate(EVAL_QUERIES):
        query = q["query"]
        query_type = q["type"]

        logger.info(f"\n--- Query {i+1}/{len(EVAL_QUERIES)} ---")
        logger.info(f"  Type:  {query_type}")
        logger.info(f"  Query: {query}")
        logger.info(f"  VRAM:  {vram_gb()} GB")

        # For image+text queries, pick a test image
        query_image = None
        query_image_path = None
        if query_type == "image_text" and cases:
            test_idx = min(i * 3, len(cases) - 1)
            test_case = cases[test_idx]
            try:
                query_image = Image.open(test_case.image_path).convert("RGB")
                query_image_path = test_case.image_path
                logger.info(f"  Query image: {Path(test_case.image_path).name}")
            except Exception as e:
                logger.warning(f"  Failed to load query image: {e}")
                query_image = None

        try:
            result = run_rag_query(
                query, embedder, embeddings, cases,
                llava_model, llava_processor,
                query_image=query_image,
            )
            result["query_id"] = i + 1
            result["query_image_path"] = query_image_path
            results.append(result)

            logger.info(f"  Answer: {result['answer'][:200]}")
            logger.info(f"  Retrieved: {[r['case_id'] for r in result['retrieved']]}")
            logger.info(f"  Scores: {[r['score'] for r in result['retrieved']]}")
            logger.info(
                f"  Time: {result['total_time_sec']}s "
                f"(ret={result['retrieval_time_sec']}s, "
                f"gen={result['generation_time_sec']}s)"
            )

        except Exception as e:
            logger.error(f"  FAILED: {e}")
            traceback.print_exc()
            results.append({
                "query_id": i + 1, "query": query,
                "query_type": query_type, "answer": f"ERROR: {e}",
                "retrieved": [], "total_time_sec": 0,
            })

    return results


# ============================================================
# 16. Save Results
# ============================================================
def save_results(results, build_meta):
    """Save evaluation results in JSON, CSV, and Markdown."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"

    successful = [r for r in results if not str(r.get("answer", "")).startswith("ERROR")]
    summary = {
        "phase": "Phase 2 - ColQwen2 Retrieval + LLaVA Generation",
        "total_queries": len(results),
        "successful": len(successful),
        "text_only": sum(1 for r in results if r.get("query_type") == "text_only"),
        "image_text": sum(1 for r in results if r.get("query_type") == "image_text"),
        "avg_retrieval_time": round(
            sum(r.get("retrieval_time_sec", 0) for r in successful) / max(len(successful), 1), 2
        ),
        "avg_generation_time": round(
            sum(r.get("generation_time_sec", 0) for r in successful) / max(len(successful), 1), 2
        ),
        "avg_total_time": round(
            sum(r.get("total_time_sec", 0) for r in successful) / max(len(successful), 1), 2
        ),
        "top_k": TOP_K,
        "index_size": build_meta.get("num_indexed", 0),
        "colqwen2_model": COLQWEN2_MODEL_ID,
        "llava_model": LLAVA_MODEL_ID,
        "vram_gb": vram_gb(),
        "gpu": gpu,
        "timestamp": datetime.now().isoformat(),
    }

    # --- JSON ---
    json_path = EVAL_DIR / f"phase2_results_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "build_meta": build_meta, "results": results},
                  f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"JSON: {json_path}")

    # --- CSV ---
    csv_path = EVAL_DIR / f"phase2_results_{ts}.csv"
    fields = ["query_id", "query", "query_type", "answer",
              "retrieved_case_ids", "retrieved_scores",
              "retrieval_time_sec", "generation_time_sec", "total_time_sec"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = dict(r)
            row["retrieved_case_ids"] = "; ".join(str(d["case_id"]) for d in r.get("retrieved", []))
            row["retrieved_scores"] = "; ".join(str(d["score"]) for d in r.get("retrieved", []))
            writer.writerow(row)
    logger.info(f"CSV:  {csv_path}")

    # --- Markdown ---
    md_path = REPORTS_DIR / f"phase2_report_{ts}.md"
    md = _generate_report(results, summary, build_meta)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    logger.info(f"Report: {md_path}")

    # --- Console Log ---
    log_path = LOGS_DIR / f"console_output_{ts}.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log_buffer.getvalue())
    logger.info(f"Log:  {log_path}")

    return {"json": str(json_path), "csv": str(csv_path),
            "report": str(md_path), "log": str(log_path)}


def _generate_report(results, summary, build_meta):
    """Generate professional markdown report."""
    lines = [
        "# Phase 2 - ColQwen2 Retrieval + LLaVA Generation Report",
        "",
        f"**Date:** {summary['timestamp']}",
        f"**GPU:** {summary['gpu']} | **VRAM:** {summary['vram_gb']} GB",
        "",
        "## Pipeline Architecture",
        "",
        "```",
        "OpenI Image-Report Pairs",
        "  -> ColQwen2 Encoding (multi-vector embeddings)",
        "  -> Saved Retrieval Index",
        "",
        "User Query (text or image+text)",
        "  -> ColQwen2 Query Embedding",
        "  -> MaxSim Similarity Search",
        "  -> Top 3 Retrieved Cases",
        "  -> Context Builder (evidence formatting)",
        "  -> LLaVA-1.5-7B (4-bit) + LoRA",
        "  -> Grounded Medical Answer",
        "```",
        "",
        "---",
        "",
        "## Index Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Documents Indexed | {build_meta.get('num_indexed', 'N/A')} |",
        f"| Skipped | {build_meta.get('num_skipped', 'N/A')} |",
        f"| Embedding Dim | {build_meta.get('embedding_dim', 'N/A')} |",
        f"| Index Build Time | {build_meta.get('build_time_seconds', 'N/A')}s |",
        f"| ColQwen2 Model | `{COLQWEN2_MODEL_ID}` |",
        "",
        "---",
        "",
        "## Evaluation Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Queries | {summary['total_queries']} |",
        f"| Successful | {summary['successful']} |",
        f"| Text-Only Queries | {summary['text_only']} |",
        f"| Image+Text Queries | {summary['image_text']} |",
        f"| Avg Retrieval Time | {summary['avg_retrieval_time']}s |",
        f"| Avg Generation Time | {summary['avg_generation_time']}s |",
        f"| Avg Total Time | {summary['avg_total_time']}s |",
        "",
        "---",
        "",
        "## Detailed Results",
        "",
    ]

    for r in results:
        ids = [str(d["case_id"]) for d in r.get("retrieved", [])]
        scores = [str(d["score"]) for d in r.get("retrieved", [])]

        lines.extend([
            f"### Query {r.get('query_id', '?')} ({r.get('query_type', '?')})",
            "",
            f"**Query:** {r['query']}",
            "",
            f"**Answer:**",
            f"> {r.get('answer', 'N/A')}",
            "",
            f"**Retrieved Cases:** {', '.join(ids)}",
            f"**Scores:** {', '.join(scores)}",
            f"**Image Source:** {r.get('image_source', 'N/A')}",
            "",
        ])

        for d in r.get("retrieved", []):
            lines.extend([
                f"<details><summary>Evidence #{d['rank']} - Case {d['case_id']} "
                f"(score: {d['score']})</summary>",
                "",
                f"- **Findings:** {d.get('findings', 'N/A')}",
                f"- **Impression:** {d.get('impression', 'N/A')}",
                "",
                "</details>",
                "",
            ])

        lines.extend([
            f"*Retrieval: {r.get('retrieval_time_sec', 0)}s | "
            f"Generation: {r.get('generation_time_sec', 0)}s | "
            f"Total: {r.get('total_time_sec', 0)}s*",
            "",
            "---",
            "",
        ])

    return "\n".join(lines)


# ============================================================
# 17. Main — 8-Stage Execution
# ============================================================
def main():
    t_start = time.time()

    logger.info("=" * 60)
    logger.info("PHASE 2 VALIDATION (HARDENED)")
    logger.info("ColQwen2 Retrieval + LLaVA Generation on OpenI")
    logger.info("=" * 60)

    # --- GPU Check ---
    if not torch.cuda.is_available():
        logger.error("No GPU detected! Phase 2 requires GPU.")
        logger.error("Enable GPU in Kaggle: Settings -> Accelerator -> GPU T4 x2")
        return

    gpu = torch.cuda.get_device_name(0)
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"GPU: {gpu} | VRAM: {vram_total:.1f} GB")
    logger.info(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")

    # --- Find Dataset ---
    dataset_info = find_dataset()
    if not dataset_info:
        logger.error("OpenI dataset not found! Add it as a Kaggle dataset.")
        logger.error("Search for: 'chest-xrays-indiana-university'")
        return

    n_images = len(list(dataset_info["images_dir"].glob("*.png")))
    logger.info(f"Dataset: {dataset_info['images_dir']} ({n_images} images)")
    if dataset_info.get("reports_csv"):
        logger.info(f"Reports: {dataset_info['reports_csv']}")

    # --- Find Adapter ---
    adapter_path = find_adapter()
    if adapter_path:
        logger.info(f"LoRA Adapter: {adapter_path}")
    else:
        logger.warning("No LoRA adapter found - will use base LLaVA model")

    # ========================================
    # STAGE 1: Load OpenI Data
    # ========================================
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 1: Loading OpenI Dataset")
    logger.info("=" * 60)

    cases = load_openi_cases(dataset_info, max_samples=MAX_INDEX_SAMPLES)
    if not cases:
        logger.error("No cases loaded! Check dataset paths.")
        return
    logger.info(f"Loaded {len(cases)} medical cases")

    # ========================================
    # STAGE 2: Build ColQwen2 Index (OFFLINE)
    # ========================================
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2: Building ColQwen2 Index (Offline)")
    logger.info("=" * 60)

    embedder = ColQwen2Embedder()
    embedder.load()

    embeddings, indexed_cases, build_meta = build_colqwen2_index(
        cases, embedder, str(INDEX_DIR),
    )

    if not embeddings:
        logger.error("Index building failed! No embeddings produced.")
        return

    logger.info(f"Index built: {len(embeddings)} documents indexed")

    # ========================================
    # STAGE 3: Verify Index (save/reload)
    # ========================================
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 3: Verifying Index Persistence")
    logger.info("=" * 60)

    for fname in ["embeddings.pt", "doc_ids.json", "document_store.json", "index_metadata.json"]:
        fpath = INDEX_DIR / fname
        exists = fpath.exists()
        size = fpath.stat().st_size if exists else 0
        status = "OK" if exists else "MISSING"
        logger.info(f"  {fname}: {status} ({size:,} bytes)")

    # Reload and verify (compatible torch.load)
    try:
        reloaded = torch.load(str(INDEX_DIR / "embeddings.pt"), map_location="cpu")
        assert len(reloaded) == len(embeddings), "Count mismatch!"
        logger.info(f"  Reload verified: {len(reloaded)} embeddings")
        del reloaded
    except Exception as e:
        logger.error(f"  Reload failed: {e}")

    # ========================================
    # STAGE 4: Test Retrieval (ONLINE)
    # ========================================
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 4: Testing ColQwen2 Retrieval")
    logger.info("=" * 60)

    # Test text-only retrieval
    test_query = "chest X-ray with cardiomegaly"
    logger.info(f"\nTest query (text-only): '{test_query}'")
    retrieved = retrieve_top_k(test_query, embedder, embeddings, indexed_cases)
    for r in retrieved:
        logger.info(f"  [{r['rank']}] Case {r['case_id']} (score: {r['score']})")
        report_preview = (r.get("report") or "N/A")[:100]
        logger.info(f"      Report: {report_preview}...")

    # Test image+text retrieval
    if indexed_cases:
        try:
            test_img = Image.open(indexed_cases[0].image_path).convert("RGB")
            test_query_img = "What findings are visible in this image?"
            logger.info(f"\nTest query (image+text): '{test_query_img}'")
            retrieved_img = retrieve_top_k(
                test_query_img, embedder, embeddings, indexed_cases,
                query_image=test_img,
            )
            for r in retrieved_img:
                logger.info(f"  [{r['rank']}] Case {r['case_id']} (score: {r['score']})")
        except Exception as e:
            logger.warning(f"Image+text retrieval test failed: {e}")

    # ========================================
    # STAGE 5: Model Swap (ColQwen2 -> LLaVA)
    # ========================================
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 5: Model Swap (ColQwen2 -> LLaVA)")
    logger.info("=" * 60)

    logger.info(f"VRAM before swap: {vram_gb()} GB")

    embedder.unload()
    flush_vram()

    logger.info(f"VRAM after ColQwen2 unload: {vram_gb()} GB")

    llava_model, llava_processor = load_llava(adapter_path)

    logger.info(f"VRAM after LLaVA load: {vram_gb()} GB (free: {vram_free_gb()} GB)")

    # ========================================
    # STAGE 6: VRAM-safe ColQwen2 Reload
    # ========================================
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 6: Reloading ColQwen2 for Online Retrieval")
    logger.info("=" * 60)

    logger.info(f"VRAM available: {vram_free_gb()} GB")

    try:
        embedder.load()
        logger.info(f"ColQwen2 reloaded | VRAM: {vram_gb()} GB (free: {vram_free_gb()} GB)")
        colqwen2_available = True
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            logger.warning(
                f"ColQwen2 reload failed due to VRAM: {e}\n"
                f"Falling back to pre-computed embeddings for retrieval."
            )
            flush_vram()
            colqwen2_available = False
        else:
            raise

    # ========================================
    # STAGE 7: Full RAG Evaluation
    # ========================================
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 7: Full RAG Evaluation (End-to-End)")
    logger.info("=" * 60)

    if colqwen2_available:
        results = run_full_evaluation(
            embedder, embeddings, indexed_cases,
            llava_model, llava_processor,
            dataset_info,
        )
    else:
        logger.warning(
            "ColQwen2 not available for online query encoding. "
            "Running generation-only with pre-retrieved context."
        )
        # Fallback: use pre-computed retrieval from Stage 4
        results = []
        for i, q in enumerate(EVAL_QUERIES):
            logger.info(f"\n--- Fallback Query {i+1}/{len(EVAL_QUERIES)} ---")
            context = build_context(retrieved)  # use Stage 4 results
            if indexed_cases:
                img = Image.open(indexed_cases[0].image_path).convert("RGB")
                gen = generate_answer(llava_model, llava_processor, img, q["query"], context)
                results.append({
                    "query_id": i + 1, "query": q["query"],
                    "query_type": q["type"], "answer": gen["answer"],
                    "retrieved": retrieved, "total_time_sec": gen["time_sec"],
                    "retrieval_time_sec": 0, "generation_time_sec": gen["time_sec"],
                })
                logger.info(f"  Answer: {gen['answer'][:200]}")

    # ========================================
    # STAGE 8: Save Results
    # ========================================
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 8: Saving Results")
    logger.info("=" * 60)

    saved = save_results(results, build_meta)

    # ========================================
    # Final Summary
    # ========================================
    total_time = time.time() - t_start
    successful = sum(1 for r in results if not str(r.get("answer", "")).startswith("ERROR"))

    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2 VALIDATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Index:       {build_meta.get('num_indexed', 0)} documents")
    logger.info(f"  Queries:     {successful}/{len(results)} successful")
    logger.info(f"  Top-K:       {TOP_K}")
    logger.info(f"  GPU:         {gpu}")
    logger.info(f"  VRAM:        {vram_gb()} GB")
    logger.info(f"  Runtime:     {total_time/60:.1f} minutes")
    logger.info("")
    logger.info("OUTPUT FILES:")
    for k, v in saved.items():
        logger.info(f"  {k:8s}: {v}")
    logger.info("")
    logger.info("Download phase2_outputs.zip from the Kaggle Output tab.")
    logger.info("=" * 60)

    # ========================================
    # Export ZIP
    # ========================================
    import shutil
    logger.info("\nCreating final ZIP archive...")
    zip_path = shutil.make_archive(
        "/kaggle/working/phase2_outputs",
        "zip",
        "/kaggle/working"
    )
    logger.info(f"ZIP created: {zip_path}")


# ============================================================
main()
