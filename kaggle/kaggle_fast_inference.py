"""
Healthcare MRAG — Fast Inference with Pre-Built ColQwen2 Index
==============================================================

FAST RAG INFERENCE: Loads a PRE-BUILT ColQwen2 index and runs
queries through the full RAG pipeline in ~20 seconds per query
(vs ~4 hours to rebuild the index).

SETUP:
  1. Kaggle notebook → GPU T4 → Internet ON
  2. Add dataset: "OpenI Chest X-rays Indiana University"
  3. Add dataset: Your saved ColQwen2 index (from previous kaggle_rag.py run)
  4. Set HF_TOKEN as a Kaggle secret
  5. Edit the USER CONFIGURATION section below
  6. Paste this entire script into one cell → Run

ARCHITECTURE:
  Pre-built index → ColQwen2 query encoding → MaxSim retrieval →
  Top 3 cases → Context builder → LLaVA generation → Answer

OUTPUT:
  - Printed results with retrieved evidence and grounded answer
  - JSON results saved to /kaggle/working/fast_inference_results/
  - Markdown report saved to /kaggle/working/fast_inference_results/
"""

# ============================================================
# ██╗   ██╗███████╗███████╗██████╗
# ██║   ██║██╔════╝██╔════╝██╔══██╗
# ██║   ██║███████╗█████╗  ██████╔╝
# ██║   ██║╚════██║██╔══╝  ██╔══██╗
# ╚██████╔╝███████║███████╗██║  ██║
#  ╚═════╝ ╚══════╝╚══════╝╚═╝  ╚═╝
#
# USER CONFIGURATION — Edit these variables before running
# ============================================================

# ----------------------------------------------------------
# QUERY MODE:
#   "text"       → Text-only query (no image, uses best retrieved image)
#   "image"      → Image-only query (default clinical question is used)
#   "image_text" → Image + text query (recommended, best results)
# ----------------------------------------------------------
QUERY_MODE = "image_text"

# ----------------------------------------------------------
# YOUR QUERY:
#   The clinical question you want answered.
#   Ignored when QUERY_MODE = "image" (a default question is used).
# ----------------------------------------------------------
USER_QUERY = "What are the key findings in this chest X-ray?"

# ----------------------------------------------------------
# QUERY IMAGE PATH:
#   Path to the chest X-ray image on Kaggle.
#   Ignored when QUERY_MODE = "text".
#
#   Examples:
#     "/kaggle/input/chest-xrays-indiana-university/images/images_normalized/1_IM-0001-3001.dcm.png"
#     "/kaggle/input/chest-xrays-indiana-university/images/images_normalized/4_IM-2050-1001.dcm.png"
#     "/kaggle/input/chest-xrays-indiana-university/images/images_normalized/7_IM-2263-1001.dcm.png"
#
#   Leave as "auto" to automatically pick a sample image from the dataset.
# ----------------------------------------------------------
QUERY_IMAGE_PATH = "auto"

# ----------------------------------------------------------
# PRE-BUILT INDEX PATH:
#   Path to the ColQwen2 index saved from a previous kaggle_rag.py run.
#   If you saved it as a Kaggle dataset named "colqwen2-openi-index":
#     "/kaggle/input/colqwen2-openi-index"
#   Or if the index is in the working directory from a previous run:
#     "/kaggle/working/colqwen2_index"
# ----------------------------------------------------------
PRE_BUILT_INDEX_PATH = "/kaggle/input/colqwen2-openi-index"

# ----------------------------------------------------------
# RETRIEVAL SETTINGS:
#   TOP_K = how many similar cases to retrieve (default 3)
# ----------------------------------------------------------
TOP_K = 3

# ----------------------------------------------------------
# OPTIONAL: Run multiple queries in batch mode.
#   Set BATCH_MODE = True to run all queries in BATCH_QUERIES list.
#   Set BATCH_MODE = False to run only the single query above.
# ----------------------------------------------------------
BATCH_MODE = False

BATCH_QUERIES = [
    {
        "mode": "image_text",
        "query": "What are the key findings in this chest X-ray?",
        "image": "auto",
    },
    {
        "mode": "image_text",
        "query": "Is there cardiomegaly or any cardiac abnormality?",
        "image": "auto",
    },
    {
        "mode": "text",
        "query": "Describe common findings in a normal chest X-ray.",
        "image": None,
    },
    {
        "mode": "image_text",
        "query": "Are there signs of pleural effusion or pneumonia?",
        "image": "auto",
    },
    {
        "mode": "image",
        "query": None,
        "image": "auto",
    },
]

# ============================================================
# END OF USER CONFIGURATION
# ============================================================


# ============================================================
# 1. Environment Setup & Dependencies
# ============================================================
import os, subprocess, sys, io

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def _pip(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

print("=" * 60)
print("HEALTHCARE MRAG — Fast Inference")
print("Installing dependencies...")
print("=" * 60)

_pip("transformers>=4.46.0")
_pip("accelerate>=0.27.0")
_pip("bitsandbytes>=0.43.0")
_pip("peft>=0.10.0")
_pip("colpali-engine>=0.3.0")
_pip("sentencepiece")

# ── HuggingFace Authentication ──
_hf_token = None
try:
    from kaggle_secrets import UserSecretsClient
    _hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF auth: using Kaggle secret ✓")
except Exception:
    pass

if not _hf_token:
    _hf_token = os.environ.get("HF_TOKEN", os.environ.get("HUGGINGFACE_TOKEN"))
    if _hf_token:
        print("HF auth: using environment variable ✓")

if not _hf_token:
    print("⚠ No HF_TOKEN found. Set it as a Kaggle secret (Add-ons → Secrets → HF_TOKEN)")
    print("  Some gated models may not load without authentication.")

if _hf_token:
    try:
        from huggingface_hub import login
        login(token=_hf_token)
        print("HuggingFace authentication successful ✓")
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

# ============================================================
# 3. Logging
# ============================================================
log_buffer = io.StringIO()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(log_buffer),
    ],
)
logger = logging.getLogger("fast_inference")


# ============================================================
# 4. Model Configuration
# ============================================================
LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
LLAVA_GEN_CONFIG = {
    "max_new_tokens": 256,
    "do_sample": False,
    "repetition_penalty": 1.3,
    "no_repeat_ngram_size": 3,
}

COLQWEN2_MODEL_ID = "vidore/colqwen2-v1.0-hf"
COLQWEN2_BATCH_SIZE = 2

MAX_CONTEXT_CHARS = 3000
MAX_EVIDENCE_CHARS = 800

DEFAULT_QUESTION = "Describe the key findings visible in this chest X-ray image."

BASE_DIR = Path("/kaggle/working")
RESULTS_DIR = BASE_DIR / "fast_inference_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── LoRA adapter search paths ──
ADAPTER_PATHS = [
    "/kaggle/input/llava-medical-adapter/final_adapter",
    "/kaggle/input/llava-medical-adapter",
    "/kaggle/working/llava-medical-vqa/final_adapter",
]

# ── OpenI dataset search paths ──
DATASET_ROOTS = [
    "/kaggle/input/datasets/raddar/chest-xrays-indiana-university",
    "/kaggle/input/chest-xrays-indiana-university",
    "/kaggle/input/openi-chest-xray",
]


# ============================================================
# 5. VRAM Utilities
# ============================================================
def vram_gb():
    if torch.cuda.is_available():
        return round(torch.cuda.memory_allocated() / 1e9, 2)
    return 0.0

def vram_free_gb():
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        used = torch.cuda.memory_allocated() / 1e9
        return round(total - used, 2)
    return 0.0

def flush_vram():
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

    for d in Path("/kaggle/input").rglob("images_normalized"):
        if d.is_dir():
            reports = next(Path("/kaggle/input").rglob("indiana_reports.csv"), None)
            projections = next(Path("/kaggle/input").rglob("indiana_projections.csv"), None)
            return {"images_dir": d, "reports_csv": reports, "projections_csv": projections}

    return None


def find_sample_image(dataset_info, offset=0):
    """Pick a sample image from the dataset for 'auto' mode."""
    if not dataset_info:
        return None
    images_dir = dataset_info["images_dir"]
    images = sorted(images_dir.glob("*.png"))
    if not images:
        return None
    idx = min(offset * 3, len(images) - 1)
    return str(images[idx])


# ============================================================
# 7. Data Loading (self-contained)
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


def load_openi_cases(dataset_info, max_samples=None):
    """Load OpenI cases from CSV files."""
    images_dir = dataset_info["images_dir"]
    reports_csv = dataset_info.get("reports_csv")
    projections_csv = dataset_info.get("projections_csv")

    if not reports_csv or not reports_csv.exists():
        logger.error("Reports CSV not found")
        return []

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

            try:
                img = Image.open(image_path)
                img.verify()
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


# ============================================================
# 8. Safe Embedding Extraction
# ============================================================
def _extract_embeddings(outputs, context=""):
    """
    Safely extract embedding tensor from ColQwen2 model outputs.
    Handles version differences across transformers and colpali-engine.
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


# ============================================================
# 9. ColQwen2 Embedder
# ============================================================
class ColQwen2Embedder:
    """ColQwen2 multi-vector encoder for document retrieval."""

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self._loaded = False

    def load(self, model_id=COLQWEN2_MODEL_ID):
        """Load ColQwen2 model."""
        from transformers import ColQwen2ForRetrieval, ColQwen2Processor

        logger.info(f"Loading ColQwen2: {model_id}")
        logger.info(f"  VRAM before: {vram_gb()} GB")

        self.processor = ColQwen2Processor.from_pretrained(model_id)
        self.model = ColQwen2ForRetrieval.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()

        self.device = self.model.device
        self._loaded = True

        logger.info(f"  ColQwen2 loaded on {self.device}")
        logger.info(f"  VRAM after: {vram_gb()} GB (free: {vram_free_gb()} GB)")

    def encode_queries(self, queries, batch_size=COLQWEN2_BATCH_SIZE):
        """Encode text queries → list of multi-vector tensors."""
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

        return all_embeddings

    def encode_image_queries(self, images, queries, batch_size=COLQWEN2_BATCH_SIZE):
        """Encode image+text queries → list of multi-vector tensors."""
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
                    inputs = self.processor(images=batch_imgs, return_tensors="pt")
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

    def encode_images(self, images, batch_size=COLQWEN2_BATCH_SIZE):
        """Encode document images → list of multi-vector tensors (for image-only queries)."""
        all_embeddings = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            try:
                inputs = self.processor(images=batch, return_tensors="pt")
            except Exception as e:
                logger.error(f"  Image processor failed: {e}")
                continue

            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)

            embeddings = _extract_embeddings(outputs, context="encode_images")
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
        """
        if not query_embs or not doc_embs:
            return torch.zeros(max(len(query_embs), 1), max(len(doc_embs), 1))

        n_queries = len(query_embs)
        n_docs = len(doc_embs)

        logger.info(f"  MaxSim scoring: {n_queries} queries × {n_docs} docs")
        scores = torch.zeros(n_queries, n_docs)

        for qi in range(n_queries):
            q = query_embs[qi].to(self.device).float()
            q = torch.nn.functional.normalize(q, p=2, dim=-1)

            for di in range(n_docs):
                d = doc_embs[di].to(self.device).float()
                d = torch.nn.functional.normalize(d, p=2, dim=-1)

                sim_matrix = torch.matmul(q, d.transpose(0, 1))
                max_sim = sim_matrix.max(dim=-1).values
                scores[qi, di] = max_sim.sum().item()

            if n_docs > 100 and (qi + 1) % max(1, n_queries // 5) == 0:
                logger.info(f"    Query {qi+1}/{n_queries} scored")

        logger.info(f"  Score range: [{scores.min():.2f}, {scores.max():.2f}]")
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


# ============================================================
# 10. Pre-Built Index Loader
# ============================================================
def load_prebuilt_index(index_path, cases):
    """
    Load a pre-built ColQwen2 index from disk.

    This SKIPS the 4-hour indexing step entirely.
    The index was built by a previous run of kaggle_rag.py.

    Returns:
        (embeddings, indexed_cases) where:
          - embeddings: list of torch.Tensor (one per document)
          - indexed_cases: list of MedicalCase aligned with embeddings
    """
    index_path = Path(index_path)

    # Find the embeddings file
    embeddings_file = None
    for name in ["embeddings.pt", "doc_embeddings.pt"]:
        candidate = index_path / name
        if candidate.exists():
            embeddings_file = candidate
            break

    # Also search subdirectories (in case index is nested)
    if embeddings_file is None:
        for f in index_path.rglob("embeddings.pt"):
            embeddings_file = f
            break
        if embeddings_file is None:
            for f in index_path.rglob("doc_embeddings.pt"):
                embeddings_file = f
                break

    if embeddings_file is None:
        raise FileNotFoundError(
            f"No embeddings file found in {index_path}. "
            f"Expected 'embeddings.pt' or 'doc_embeddings.pt'. "
            f"Contents: {list(index_path.rglob('*'))[:20]}"
        )

    logger.info(f"Loading pre-built index from: {embeddings_file}")

    t0 = time.time()
    embeddings = torch.load(str(embeddings_file), map_location="cpu")
    load_time = time.time() - t0

    logger.info(f"  Loaded {len(embeddings)} document embeddings in {load_time:.1f}s")

    if embeddings:
        logger.info(f"  Embedding shape: {embeddings[0].shape}")
        logger.info(f"  Embedding dim: {embeddings[0].shape[-1]}")

    # Load document IDs to align embeddings with cases
    doc_ids_file = embeddings_file.parent / "doc_ids.json"
    if doc_ids_file.exists():
        with open(doc_ids_file, "r") as f:
            doc_ids = json.load(f)
        logger.info(f"  Loaded {len(doc_ids)} document IDs")

        # Build case lookup by case_id
        case_lookup = {c.case_id: c for c in cases}
        indexed_cases = []
        for doc_id in doc_ids:
            if doc_id in case_lookup:
                indexed_cases.append(case_lookup[doc_id])
            else:
                # Create a placeholder if the case isn't in our loaded dataset
                indexed_cases.append(MedicalCase(
                    case_id=doc_id,
                    image_path="",
                    findings=None,
                    impression=None,
                    report="",
                ))
    else:
        # No doc_ids file — fall back to using cases in order
        logger.warning("  No doc_ids.json found, aligning embeddings with cases by order")
        indexed_cases = cases[:len(embeddings)]

    # Try loading document store for richer metadata
    doc_store_file = embeddings_file.parent / "document_store.json"
    if doc_store_file.exists():
        with open(doc_store_file, "r", encoding="utf-8") as f:
            doc_store = json.load(f)
        docs = doc_store.get("documents", {})
        logger.info(f"  Loaded document store with {len(docs)} entries")

        # Enrich cases from document store
        for i, case in enumerate(indexed_cases):
            if case.case_id in docs:
                doc = docs[case.case_id]
                if not case.findings:
                    case.findings = doc.get("findings")
                if not case.impression:
                    case.impression = doc.get("impression")
                if not case.report:
                    case.report = doc.get("report", "")
                if not case.image_path:
                    case.image_path = doc.get("image_path", "")
                if not case.metadata:
                    case.metadata = doc.get("metadata", {})

    # Load index metadata if available
    meta_file = embeddings_file.parent / "index_metadata.json"
    if meta_file.exists():
        with open(meta_file, "r") as f:
            index_meta = json.load(f)
        logger.info(f"  Index metadata:")
        logger.info(f"    Built: {index_meta.get('build_timestamp', 'unknown')}")
        logger.info(f"    Documents: {index_meta.get('num_indexed', 'unknown')}")
        logger.info(f"    Build time: {index_meta.get('build_time_seconds', 'unknown')}s")
        logger.info(f"    Embedding dim: {index_meta.get('embedding_dim', 'unknown')}")

    # Validate alignment
    if len(embeddings) != len(indexed_cases):
        n = min(len(embeddings), len(indexed_cases))
        logger.warning(
            f"  Embedding count ({len(embeddings)}) != case count ({len(indexed_cases)}). "
            f"Truncating to {n}."
        )
        embeddings = embeddings[:n]
        indexed_cases = indexed_cases[:n]

    logger.info(f"  ✓ Index ready: {len(embeddings)} documents")
    return embeddings, indexed_cases


# ============================================================
# 11. Retrieval
# ============================================================
def retrieve_top_k(query, embedder, embeddings, cases,
                   query_image=None, top_k=TOP_K):
    """
    Retrieve top-k cases for a query using MaxSim scoring.

    Supports:
      - text-only:  query_image=None
      - image+text: query_image=PIL.Image
      - image-only: query="", query_image=PIL.Image
    """
    if not embeddings:
        logger.warning("Empty index, cannot retrieve")
        return []

    t0 = time.time()

    # Encode query based on mode
    if query_image is not None and query:
        logger.info("  Retrieval mode: image + text")
        query_embs = embedder.encode_image_queries(
            images=[query_image], queries=[query],
        )
    elif query_image is not None:
        logger.info("  Retrieval mode: image-only")
        query_embs = embedder.encode_images([query_image])
    else:
        logger.info("  Retrieval mode: text-only")
        query_embs = embedder.encode_queries([query])

    if not query_embs:
        logger.error("Failed to encode query!")
        return []

    # MaxSim scoring
    scores = embedder.score(query_embs, embeddings)

    # Flatten to 1D
    if scores.dim() == 2:
        scores = scores[0]
    scores = scores.reshape(-1)

    # Ensure alignment
    if scores.shape[0] != len(cases):
        n = min(scores.shape[0], len(cases))
        scores = scores[:n]
        cases = cases[:n]

    # Top-k
    k = min(top_k, len(cases))
    top_scores, top_indices = torch.topk(scores, k=k)

    retrieval_time = time.time() - t0

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
        f"  Retrieved {len(results)} cases in {retrieval_time:.2f}s: "
        f"{[(r['case_id'], r['score']) for r in results]}"
    )
    return results


# ============================================================
# 12. Context Builder
# ============================================================
def build_context(retrieved_results):
    """Build structured, token-budget-aware context from retrieved cases."""
    if not retrieved_results:
        return ""

    parts = ["=== Retrieved Medical Evidence ===\n"]
    total_chars = len(parts[0])

    for r in retrieved_results:
        block_parts = [f"--- Evidence #{r['rank']} ---"]
        block_parts.append(f"Case ID: {r['case_id']}")
        block_parts.append(f"Relevance Score: {r['score']}")

        if r.get("findings"):
            findings = r["findings"][:MAX_EVIDENCE_CHARS]
            block_parts.append(f"Findings: {findings}")

        if r.get("impression"):
            impression = r["impression"][:MAX_EVIDENCE_CHARS]
            block_parts.append(f"Impression: {impression}")

        if not r.get("findings") and not r.get("impression") and r.get("report"):
            report = r["report"][:MAX_EVIDENCE_CHARS]
            block_parts.append(f"Report: {report}")

        mesh = r.get("metadata", {}).get("mesh_terms")
        if mesh:
            block_parts.append(f"MeSH Terms: {'; '.join(mesh[:5])}")

        block_parts.append("")
        block = "\n".join(block_parts)

        if total_chars + len(block) > MAX_CONTEXT_CHARS:
            logger.info(f"  Context budget reached at evidence #{r['rank']}")
            break

        parts.append(block)
        total_chars += len(block)

    parts.append("=== End of Retrieved Evidence ===")
    context = "\n".join(parts)

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

    # Build RAG-grounded prompt
    if context:
        prompt = (
            f"USER: <image>\n"
            f"You are a medical imaging specialist. Use the following retrieved "
            f"clinical evidence from similar cases to help answer the question.\n"
            f"\n{context}\n\n"
            f"Based on the image and the retrieved evidence above, "
            f"answer the following question:\n"
            f"{question}\n"
            f"Provide a clinically relevant, concise answer.\n"
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

    # Clean repetition artifacts
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
# 14. Result Formatting & Saving
# ============================================================
def print_result(result, query_num=1):
    """Print a single query result in a professional format."""
    print("\n" + "=" * 70)
    print(f"  QUERY {query_num}")
    print("=" * 70)
    print(f"  Mode:     {result['query_mode']}")
    print(f"  Query:    {result['query']}")
    if result.get("query_image_path"):
        print(f"  Image:    {Path(result['query_image_path']).name}")
    print("-" * 70)

    print(f"\n  ▸ RETRIEVED EVIDENCE ({len(result['retrieved'])} cases):\n")
    for r in result["retrieved"]:
        print(f"    [{r['rank']}] Case {r['case_id']}  (score: {r['score']})")
        if r.get("findings"):
            print(f"        Findings:   {r['findings'][:150]}...")
        if r.get("impression"):
            print(f"        Impression: {r['impression'][:150]}")

    print(f"\n  ▸ GENERATED ANSWER:\n")
    # Word-wrap the answer for readability
    answer = result["answer"]
    words = answer.split()
    line = "    "
    for word in words:
        if len(line) + len(word) + 1 > 76:
            print(line)
            line = "    " + word
        else:
            line += " " + word if line.strip() else "    " + word
    if line.strip():
        print(line)

    print(f"\n  ▸ TIMING:")
    print(f"    Retrieval:  {result['retrieval_time_sec']}s")
    print(f"    Generation: {result['generation_time_sec']}s")
    print(f"    Total:      {result['total_time_sec']}s")
    print("=" * 70)


def save_results(all_results, console_log=""):
    """Save results as JSON and Markdown report."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON
    json_path = RESULTS_DIR / f"fast_inference_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "num_queries": len(all_results),
            "results": all_results,
        }, f, indent=2, ensure_ascii=False, default=str)

    # Markdown report
    md_path = RESULTS_DIR / f"fast_inference_report_{ts}.md"
    md_lines = [
        "# Healthcare MRAG — Fast Inference Report", "",
        f"**Date:** {datetime.now().isoformat()}",
        f"**Index:** Pre-built ColQwen2 index",
        f"**Queries:** {len(all_results)}", "",
        "---", "",
    ]

    for r in all_results:
        md_lines.extend([
            f"## Query {r.get('query_num', '?')}: {r['query_mode']}", "",
            f"**Query:** {r['query']}", "",
        ])
        if r.get("query_image_path"):
            md_lines.append(f"**Image:** `{Path(r['query_image_path']).name}`\n")

        md_lines.append("### Retrieved Evidence\n")
        md_lines.append("| Rank | Case ID | Score | Impression |")
        md_lines.append("|------|---------|-------|------------|")
        for ev in r["retrieved"]:
            imp = (ev.get("impression") or ev.get("findings") or "N/A")[:80]
            md_lines.append(f"| {ev['rank']} | {ev['case_id']} | {ev['score']} | {imp} |")
        md_lines.append("")

        md_lines.extend([
            "### Generated Answer", "",
            f"> {r['answer']}", "",
            f"*Retrieval: {r['retrieval_time_sec']}s | "
            f"Generation: {r['generation_time_sec']}s | "
            f"Total: {r['total_time_sec']}s*", "",
            "---", "",
        ])

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # Console log
    if console_log:
        log_path = RESULTS_DIR / f"console_log_{ts}.txt"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(console_log)

    logger.info(f"Results saved to: {RESULTS_DIR}")
    logger.info(f"  JSON:   {json_path.name}")
    logger.info(f"  Report: {md_path.name}")

    return json_path, md_path


# ============================================================
# 15. Main Pipeline
# ============================================================
def main():
    pipeline_start = time.time()

    print("\n" + "=" * 70)
    print("  HEALTHCARE MRAG — Fast Inference with Pre-Built Index")
    print("=" * 70)

    # ── GPU Check ──
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        print(f"  GPU: {gpu_name} ({gpu_mem} GB)")
    else:
        print("  ⚠ No GPU detected. This script requires a CUDA GPU.")
        print("    Enable GPU in Kaggle: Settings → Accelerator → GPU T4")
        return

    # ── Step 1: Find dataset ──
    print("\n[Step 1/7] Locating OpenI dataset...")
    dataset_info = find_dataset()
    if dataset_info is None:
        print("  ✗ OpenI dataset not found!")
        print("    Add 'OpenI Chest X-rays Indiana University' to your Kaggle notebook.")
        return
    print(f"  ✓ Images: {dataset_info['images_dir']}")
    print(f"  ✓ Reports: {dataset_info.get('reports_csv', 'N/A')}")

    # ── Step 2: Load cases ──
    print("\n[Step 2/7] Loading OpenI cases...")
    cases = load_openi_cases(dataset_info)
    if not cases:
        print("  ✗ No cases loaded!")
        return
    print(f"  ✓ {len(cases)} cases loaded")

    # ── Step 3: Load pre-built index ──
    print(f"\n[Step 3/7] Loading pre-built ColQwen2 index...")
    print(f"  Path: {PRE_BUILT_INDEX_PATH}")
    try:
        embeddings, indexed_cases = load_prebuilt_index(PRE_BUILT_INDEX_PATH, cases)
    except FileNotFoundError as e:
        print(f"  ✗ {e}")
        print("\n  To create the index, run kaggle_rag.py first, then save")
        print("  the colqwen2_index/ folder as a Kaggle dataset.")
        return
    print(f"  ✓ {len(embeddings)} document embeddings loaded")

    # ── Step 4: Load ColQwen2 for query encoding ──
    print("\n[Step 4/7] Loading ColQwen2 for query encoding...")
    embedder = ColQwen2Embedder()
    embedder.load()
    print(f"  ✓ ColQwen2 ready | VRAM: {vram_gb()} GB")

    # ── Step 5: Process queries ──
    print("\n[Step 5/7] Processing queries...")

    # Build query list
    if BATCH_MODE:
        queries_to_run = []
        for i, bq in enumerate(BATCH_QUERIES):
            mode = bq["mode"]
            query = bq.get("query") or DEFAULT_QUESTION
            img_path = bq.get("image")

            if img_path == "auto" and mode in ("image", "image_text"):
                img_path = find_sample_image(dataset_info, offset=i)
            elif img_path is None:
                img_path = None

            queries_to_run.append({
                "mode": mode,
                "query": query,
                "image_path": img_path,
            })
    else:
        # Single query mode
        query = USER_QUERY if QUERY_MODE != "image" else DEFAULT_QUESTION
        img_path = QUERY_IMAGE_PATH

        if img_path == "auto" and QUERY_MODE in ("image", "image_text"):
            img_path = find_sample_image(dataset_info)
        elif QUERY_MODE == "text":
            img_path = None

        queries_to_run = [{
            "mode": QUERY_MODE,
            "query": query,
            "image_path": img_path,
        }]

    # Run retrieval for all queries (while ColQwen2 is loaded)
    retrieval_results = []
    for i, q in enumerate(queries_to_run):
        print(f"\n  --- Query {i+1}/{len(queries_to_run)}: {q['mode']} ---")
        print(f"  Query: {q['query'][:80]}")

        query_image = None
        if q["image_path"] and q["mode"] in ("image", "image_text"):
            try:
                query_image = Image.open(q["image_path"]).convert("RGB")
                print(f"  Image: {Path(q['image_path']).name}")
            except Exception as e:
                print(f"  ⚠ Failed to load image: {e}")

        t_ret = time.time()
        retrieved = retrieve_top_k(
            q["query"] if q["mode"] != "image" else "",
            embedder, embeddings, indexed_cases,
            query_image=query_image, top_k=TOP_K,
        )
        ret_time = time.time() - t_ret

        retrieval_results.append({
            "query_info": q,
            "query_image": query_image,
            "retrieved": retrieved,
            "retrieval_time": round(ret_time, 2),
        })

        print(f"  ✓ Retrieved {len(retrieved)} cases in {ret_time:.2f}s")
        for r in retrieved:
            print(f"    [{r['rank']}] Case {r['case_id']} (score: {r['score']})")

    # ── Step 6: Unload ColQwen2, Load LLaVA ──
    print(f"\n[Step 6/7] Switching models: ColQwen2 → LLaVA...")
    embedder.unload()
    print(f"  ✓ ColQwen2 unloaded | VRAM: {vram_gb()} GB")

    adapter_path = find_adapter()
    if adapter_path:
        print(f"  LoRA adapter found: {adapter_path}")
    else:
        print("  No LoRA adapter found (using base LLaVA)")

    llava_model, llava_processor = load_llava(adapter_path=adapter_path)
    print(f"  ✓ LLaVA loaded | VRAM: {vram_gb()} GB")

    # ── Step 7: Generate answers ──
    print(f"\n[Step 7/7] Generating RAG-grounded answers...")

    all_results = []
    for i, rr in enumerate(retrieval_results):
        q = rr["query_info"]
        print(f"\n  --- Generating answer for query {i+1}/{len(retrieval_results)} ---")

        # Build context from retrieved evidence
        context = build_context(rr["retrieved"])

        # Select image for LLaVA
        if rr["query_image"] is not None:
            llava_image = rr["query_image"]
            image_source = "query"
        elif rr["retrieved"]:
            try:
                llava_image = Image.open(rr["retrieved"][0]["image_path"]).convert("RGB")
                image_source = f"retrieved (case {rr['retrieved'][0]['case_id']})"
            except Exception:
                llava_image = None
                image_source = "none"
        else:
            llava_image = None
            image_source = "none"

        # Generate
        t_gen = time.time()
        if llava_image is not None:
            question = q["query"] if q["mode"] != "image" else DEFAULT_QUESTION
            gen_result = generate_answer(
                llava_model, llava_processor,
                llava_image, question, context,
            )
        else:
            gen_result = {
                "answer": "[No image available for generation]",
                "time_sec": 0, "input_tokens": 0, "output_tokens": 0,
            }
        gen_time = time.time() - t_gen

        result = {
            "query_num": i + 1,
            "query_mode": q["mode"],
            "query": q["query"],
            "query_image_path": q["image_path"],
            "image_source": image_source,
            "answer": gen_result["answer"],
            "retrieved": rr["retrieved"],
            "retrieval_time_sec": rr["retrieval_time"],
            "generation_time_sec": round(gen_time, 2),
            "total_time_sec": round(rr["retrieval_time"] + gen_time, 2),
            "input_tokens": gen_result["input_tokens"],
            "output_tokens": gen_result["output_tokens"],
        }
        all_results.append(result)

        # Print this result immediately
        print_result(result, query_num=i + 1)

    # ── Save results ──
    console_log = log_buffer.getvalue()
    json_path, md_path = save_results(all_results, console_log)

    # ── Summary ──
    total_time = time.time() - pipeline_start
    print("\n" + "=" * 70)
    print("  FAST INFERENCE COMPLETE")
    print("=" * 70)
    print(f"  Queries processed:  {len(all_results)}")
    print(f"  Index size:         {len(embeddings)} documents")
    print(f"  Total pipeline:     {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Results saved:      {RESULTS_DIR}")
    print(f"  JSON:               {json_path.name}")
    print(f"  Report:             {md_path.name}")

    avg_query_time = sum(r["total_time_sec"] for r in all_results) / max(len(all_results), 1)
    print(f"\n  Avg query time:     {avg_query_time:.1f}s")
    print(f"  (vs ~4 hours to rebuild the full index)")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    main()
else:
    main()
