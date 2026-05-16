# Phase 1 Implementation — Step-by-Step Explained

> **What this document covers:** How every file was created, why it exists, how it connects to other files, and how data flows through the system.

---

## 1. Project Setup — How It Started

### The Goal

Build a modular codebase that can grow from simple VQA (Phase 1) to full RAG (Phase 5) **without rewriting code**. Each phase adds new components while keeping existing ones unchanged.

### Environment Setup

```
healthcare_mrag/
├── .venv/                  ← Python virtual environment
├── environment.yml         ← Conda environment definition
├── requirements.txt        ← pip dependencies
└── setup.py                ← Makes the project installable as a package
```

**Why `setup.py`?**  
Running `pip install -e .` installs the project as an editable package. This means you can write `from src.models.llava_model import LLaVAModel` from anywhere — no `sys.path` hacking needed.

**Key dependencies:**
```
transformers    → HuggingFace model loading (LLaVA, processors)
peft            → LoRA adapter management
bitsandbytes    → 4-bit quantization
torch           → Neural network operations
Pillow          → Image loading and processing
PyYAML          → Configuration file parsing
```

---

## 2. Folder Structure — Why Each Folder Exists

```
healthcare_mrag/
│
├── configs/                ← YAML configuration files
│   ├── model_config.yaml   ← Which model, quantization, LoRA settings
│   ├── data_config.yaml    ← Dataset paths, preprocessing params
│   └── training_config.yaml← Training hyperparameters
│
├── src/                    ← Core library code (imported by scripts)
│   ├── __init__.py
│   ├── data/               ← Dataset loading and preprocessing
│   │   ├── base_dataset.py     ← Abstract dataset interface
│   │   ├── openi_dataset.py    ← OpenI CSV parser + image loader
│   │   └── preprocessing.py    ← Image resize, normalize
│   │
│   ├── models/             ← Model abstractions
│   │   ├── base_vlm.py         ← Abstract VLM interface
│   │   ├── llava_model.py      ← LLaVA-specific implementation
│   │   └── model_factory.py    ← Config → model instance
│   │
│   └── utils/              ← Shared utilities
│       ├── device.py           ← GPU detection, VRAM reporting
│       ├── image_utils.py      ← Image loading helpers
│       └── logging_utils.py    ← Consistent log formatting
│
├── pipelines/              ← End-to-end orchestration
│   └── simple_vqa.py       ← Phase 1 pipeline: Image→Question→Answer
│
├── scripts/                ← Executable scripts (CLI entry points)
│   ├── inference.py        ← Local inference + batch eval
│   ├── quick_test.py       ← Standalone zero-config test
│   ├── setup_adapter.py    ← Extract adapter ZIP files
│   ├── validate.py         ← Config/environment validation
│   └── _test_pipeline.py   ← End-to-end pipeline test
│
├── kaggle/                 ← Kaggle notebook scripts
│   ├── train_kaggle.py     ← QLoRA training on Kaggle T4
│   └── kaggle_inference.py ← GPU inference + evaluation
│
├── checkpoints/            ← Saved model weights
│   └── llava-medical-vqa/
│       └── final_adapter/  ← LoRA adapter (76 MB)
│
├── data/                   ← Local dataset files
│   └── openi/
│       ├── images/         ← Chest X-ray PNG files
│       └── reports/        ← CSV files (reports + projections)
│
└── outputs/                ← Generated results
    └── evaluation/         ← JSON, CSV, Markdown reports
```

### Design Principle: Separation of Concerns

Every folder has ONE job:

| Folder | Job | Never does |
|---|---|---|
| `configs/` | Stores settings | Never contains code |
| `src/` | Reusable library code | Never runs directly |
| `scripts/` | CLI entry points | Never defines reusable classes |
| `pipelines/` | Orchestrates components | Never loads data or models directly |
| `kaggle/` | Self-contained Kaggle scripts | Never imports from `src/` (different environment) |

---

## 3. Configuration Files — The Control Center

### Why YAML Configs?

Instead of hardcoding values like `model_id = "llava-hf/llava-1.5-7b-hf"` inside code, we put all settings in YAML files. This means:
- Change the model without editing code
- Different configs for local vs. Kaggle
- Easy to reproduce experiments

### `model_config.yaml` — What It Controls

```yaml
model:
  name: "llava-1.5-7b"                  # Which model class to use
  model_id: "llava-hf/llava-1.5-7b-hf"  # HuggingFace model ID
  device: "auto"                          # GPU selection

  quantization:
    enabled: true                         # Use 4-bit quantization?
    bnb_4bit_quant_type: "nf4"           # Quantization algorithm

  generation:
    max_new_tokens: 512                   # Max answer length
    do_sample: false                      # Greedy decoding

  lora:
    r: 16                                 # LoRA rank
    lora_alpha: 32                        # LoRA scaling
    target_modules: [q_proj, k_proj, v_proj, o_proj]
```

### `data_config.yaml` — What It Controls

```yaml
dataset:
  name: "openi"
  images_dir: "data/openi/images"
  reports_dir: "data/openi/reports"
  prefer_frontal: true                   # Use PA (front) views

preprocessing:
  image:
    resize: [336, 336]                   # LLaVA's expected resolution
    normalize: true
    mean: [0.48145466, 0.4578275, 0.40821073]  # CLIP normalization
```

---

## 4. Dataset Loading — OpenI

### What is OpenI?

OpenI (Open-i) is a public medical image dataset from the NIH. It contains:
- **7,470 chest X-ray images** (PNG files)
- **3,955 radiology reports** (CSV files)
- Each report has: findings, impression, MeSH tags, patient info

### How the Loader Works

**File:** `src/data/openi_dataset.py`

```
Step 1: Read indiana_reports.csv
        → Each row has: uid, findings, impression, image filenames

Step 2: Read indiana_projections.csv
        → Maps uid → image filename + projection type (PA/lateral)

Step 3: Match images to reports
        → For each uid, find the PNG file in the images directory
        → Prefer PA (frontal) views over lateral

Step 4: Build samples
        → Each sample = (image_path, report_text, uid, question)
        → Question defaults to "What are the findings?"
        → Answer = the impression or findings text from the CSV
```

**The matching logic:**
```python
# CSV row: uid=1000, image="1000_IM-0003-1001"
# Image file: data/openi/images/1000_IM-0003-1001.dcm.png
# Report text: "No acute cardiopulmonary abnormality."
```

### Base Dataset Interface

**File:** `src/data/base_dataset.py`

```python
@dataclass
class MedicalSample:
    sample_id: str          # "openi_1000"
    image: Image.Image      # PIL Image
    image_path: str         # "data/openi/images/1000_IM-0003.png"
    question: str           # "What are the findings?"
    answer: str             # "No acute cardiopulmonary abnormality."

class BaseDataset(ABC):
    def load(self) → None: ...
    def __getitem__(self, idx) → MedicalSample: ...
    def __len__(self) → int: ...
```

**Why abstract?**  
When we add a new dataset (e.g., CheXpert, MIMIC-CXR), we just create a new class that implements `BaseDataset`. All pipelines work unchanged because they program against the interface.

---

## 5. Model Abstraction — The Factory Pattern

### The Problem

We use LLaVA now, but later we might switch to Qwen2-VL or another model. If inference scripts directly import `LLaVAModel`, switching requires editing every script.

### The Solution: Registry + Factory

**File:** `src/models/model_factory.py`

```python
MODEL_REGISTRY = {
    "llava-1.5-7b": "src.models.llava_model.LLaVAModel",
    # "qwen2-vl-7b": "src.models.qwen2vl_model.Qwen2VLModel",  # Phase 5
}

def create_model(config):
    name = config["model"]["name"]  # "llava-1.5-7b"
    class_path = MODEL_REGISTRY[name]
    # Dynamically import and instantiate
    return LLaVAModel()
```

**How it works:**
```python
# In any script:
config = yaml.load("configs/model_config.yaml")
model = create_model(config)   # Returns a LLaVAModel instance
model.load(config)             # Loads weights onto GPU
output = model.generate(image, question)  # Runs inference
```

To switch models, just change `model_config.yaml`:
```yaml
model:
  name: "qwen2-vl-7b"  # ← change this one line
```

### Base VLM Interface

**File:** `src/models/base_vlm.py`

```python
class BaseVLM(ABC):
    def load(config) → None          # Load weights
    def generate(image, question) → VLMOutput  # Run inference
    def caption(image) → str         # Generate caption (Phase 2)
    def get_memory_footprint() → dict # Report VRAM usage
```

Every model implements this interface. Pipelines only know about `BaseVLM`.

---

## 6. LLaVA Integration — The Model Wrapper

**File:** `src/models/llava_model.py` (311 lines)

This is the most important file. It wraps HuggingFace's LLaVA implementation with our project's interface.

### Load Method — What Happens

```python
def load(self, config):
    # 1. Load processor (tokenizer + image processor)
    self._processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    
    # 2. Load base model with 4-bit quantization
    self._model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, ...),
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # 3. Load LoRA adapter (if specified)
    if adapter_path:
        self._model = PeftModel.from_pretrained(self._model, adapter_path)
```

### Generate Method — What Happens

```python
def generate(self, image, question, context=None):
    # 1. Build prompt
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    
    # 2. Process inputs (image → pixels, text → tokens)
    inputs = self._processor(text=prompt, images=image, return_tensors="pt")
    
    # 3. Align dtypes (processor outputs float32, model expects float16)
    inputs = {k: v.to(dtype=model_dtype) if float else v for ...}
    
    # 4. Generate answer
    output_ids = self._model.generate(**inputs, max_new_tokens=64)
    
    # 5. Decode tokens → text
    answer = self._processor.decode(output_ids[...])
    
    # 6. Return structured output
    return VLMOutput(answer=answer, ...)
```

### Key Design Decisions

1. **`LlavaProcessor` instead of `AutoProcessor`**  
   `AutoProcessor` broke in transformers 5.8.1 — it can't auto-detect LLaVA's processor class. Explicit import fixes this.

2. **float16 for CPU mode**  
   float32 needs 28 GB RAM (crashes on 16 GB systems). float16 needs ~14 GB.

3. **dtype alignment**  
   The processor always outputs float32 tensors, but the model may be float16. We cast all floating-point inputs to match the model's dtype.

---

## 7. Kaggle Training — How the Adapter Was Created

**File:** `kaggle/train_kaggle.py` (390 lines)

This script runs entirely inside a Kaggle notebook. It fine-tunes LLaVA on the OpenI dataset.

### Training Flow

```
Step 1: Install dependencies (transformers, peft, bitsandbytes)
Step 2: Find OpenI dataset in /kaggle/input/
Step 3: Parse indiana_reports.csv + indiana_projections.csv
Step 4: Match 3,826 valid image-report pairs
Step 5: Split into train/val sets (80/20)
Step 6: Load LLaVA-1.5-7B in 4-bit
Step 7: Attach LoRA adapters (r=16) to q/k/v/o_proj
Step 8: Freeze vision tower + projector LoRA weights
Step 9: Train for ~400 steps (batch_size=2, grad_accum=4)
Step 10: Save adapter weights to /kaggle/working/final_adapter/
```

### The Custom Collator

The training script has a custom data collator that:
- Loads the image from disk
- Creates the prompt: `"USER: <image>\nWhat are the findings?\nASSISTANT: {report_text}"`
- Processes through LlavaProcessor
- Handles padding and batching
- **Disables truncation** to avoid cutting image tokens

### LoRA Targeting Strategy

```python
# 1. Apply LoRA to ALL q/k/v/o_proj modules (including vision tower)
model = get_peft_model(model, lora_config)

# 2. THEN freeze the adapters in vision tower and projector
for name, param in model.named_parameters():
    if "vision_tower" in name or "multi_modal_projector" in name:
        param.requires_grad = False
```

This "attach-then-freeze" strategy is more robust than regex-based targeting because it works regardless of the model's internal module naming.

### Output: The Adapter

The trained adapter is saved as:
```
final_adapter/
├── adapter_config.json          ← LoRA configuration (r, alpha, targets)
├── adapter_model.safetensors    ← Trained LoRA weights (76 MB)
├── tokenizer.json               ← Tokenizer state
├── tokenizer_config.json
├── processor_config.json
└── chat_template.jinja
```

76 MB instead of 14 GB — that's the power of LoRA.

---

## 8. Local Inference — How It Works

### Quick Test (zero-config)

**File:** `scripts/quick_test.py`

Self-contained script with no config files needed:

```bash
python scripts/quick_test.py --cpu --image "data/openi/images/1000.dcm.png"
```

Directly imports LLaVA + LoRA, runs inference, prints answer.

### Modular Inference

**File:** `scripts/inference.py`

Uses the full modular architecture:

```
inference.py
  → reads configs/model_config.yaml
  → calls model_factory.create_model()
    → instantiates LLaVAModel
  → calls model.load(config)
    → loads LLaVA + quantization + adapter
  → calls model.generate(image, question)
    → returns VLMOutput with answer
```

### Three Modes

```bash
# Single image
python scripts/inference.py --image path/to/xray.png --question "What do you see?"

# Interactive
python scripts/inference.py
# → prompts for image path and question

# Batch evaluation
python scripts/inference.py --batch-eval --max-samples 5
# → selects 5 OpenI samples, runs inference, saves JSON/CSV/Markdown
```

---

## 9. Kaggle Inference — GPU Evaluation

**File:** `kaggle/kaggle_inference.py` (638 lines)

Self-contained script for Kaggle — paste into one notebook cell.

### What It Does Automatically

```
1. Install dependencies
2. Detect GPU (T4/P100)
3. Scan /kaggle/input/ for adapter and dataset
4. Load LLaVA in 4-bit (~4 GB VRAM)
5. Load LoRA adapter (~4.15 GB total)
6. Select 5 evaluation samples (mix normal/abnormal)
7. Ask 5 different clinical questions
8. Generate answers (~14s each on T4)
9. Clean answers (remove repetition, hallucination filler)
10. Save JSON + CSV + Markdown report
11. Enter interactive mode
```

### Generation Quality Controls

```python
GEN_CONFIG = {
    "max_new_tokens": 64,        # Short answers only
    "do_sample": False,           # Deterministic (reproducible)
    "repetition_penalty": 1.5,    # Penalize repeated tokens
    "no_repeat_ngram_size": 3,    # Block 3-word repetitions
}

# Prompt engineering
prompt = f"USER: <image>\n{question} Answer in 1-2 sentences as a radiologist.\nASSISTANT:"
```

### Post-Processing

```python
def clean_answer(text):
    # 1. Remove ". . . ." patterns
    # 2. Cut at last complete sentence
    # 3. Remove hallucination filler ("If you need further...", etc.)
    # 4. Collapse whitespace
```

---

## 10. Batch Evaluation — How Results Are Saved

### Sample Selection

The system picks 5 diverse samples:
- Reads `indiana_reports.csv`
- Categorizes by findings (normal vs. abnormal)
- Picks ~2 normal + ~3 abnormal for a diverse demo

### Output Files

```
/kaggle/working/results/
├── evaluation_results_20260515.json    ← Machine-readable
├── evaluation_results_20260515.csv     ← Spreadsheet-friendly
├── evaluation_report_20260515.md       ← Professor presentation
└── console_output_20260515.txt         ← Full log
```

### JSON Structure

```json
{
  "summary": {
    "total_samples": 5,
    "avg_time_sec": 14.07,
    "vram_gb": 4.15,
    "gpu": "Tesla T4",
    "model": "llava-hf/llava-1.5-7b-hf"
  },
  "results": [
    {
      "sample_id": 1,
      "uid": "1",
      "question": "Briefly describe the key findings.",
      "answer": "No acute cardiopulmonary abnormality.",
      "ground_truth": "Normal chest x-ray.",
      "time_sec": 13.03,
      "input_tokens": 601,
      "output_tokens": 12
    }
  ]
}
```

---

## 11. How All Files Connect — The Dependency Map

```
model_config.yaml ─────────────────────────────┐
data_config.yaml ──────────────────────────┐    │
                                           │    │
scripts/inference.py ──────────────────────┤    │
  │                                        │    │
  ├── src/models/model_factory.py ─────────┤────┘
  │     │                                  │
  │     └── src/models/llava_model.py      │
  │           │                            │
  │           └── src/models/base_vlm.py   │
  │                                        │
  ├── src/data/openi_dataset.py ───────────┘
  │     │
  │     └── src/data/base_dataset.py
  │
  ├── src/data/preprocessing.py
  ├── src/utils/device.py
  ├── src/utils/logging_utils.py
  └── src/utils/image_utils.py
```

**Key flow:**
```
Config YAML → Factory → Model Class → Load Weights → Generate
Config YAML → Dataset → Load CSV → Match Images → MedicalSample
Script → Pipeline → Model + Dataset → Run → Save Results
```

---

## 12. One Complete Example — Image to Answer

Let's trace one sample through the entire codebase:

### Input
```
Image: data/openi/images/1000_IM-0003-1001.dcm.png
Question: "Briefly describe the key findings in this chest X-ray."
```

### Step 1: Dataset Loading (`openi_dataset.py`)
```python
# Read CSV
row = {"uid": "1000", "impression": "Normal chest x-ray.", "image": "1000_IM-0003-1001"}
# Build sample
sample = MedicalSample(
    sample_id="openi_1000",
    image=Image.open("data/openi/images/1000_IM-0003-1001.dcm.png"),
    question="What are the findings?",
    answer="Normal chest x-ray.",
)
```

### Step 2: Model Loading (`llava_model.py`)
```python
# Load processor → LlavaProcessor(tokenizer + CLIPImageProcessor)
# Load model → LlavaForConditionalGeneration (4-bit, ~4 GB)
# Load adapter → PeftModel (76 MB LoRA weights)
```

### Step 3: Preprocessing (`preprocessing.py`)
```python
image = Image.open(path).convert("RGB")  # 2800×2400 → RGB
image = image.resize((336, 336))          # LLaVA resolution
# Normalization happens inside the processor
```

### Step 4: Tokenization (inside `generate()`)
```python
prompt = "USER: <image>\nBriefly describe the key findings. Answer in 1-2 sentences.\nASSISTANT:"
inputs = processor(text=prompt, images=image, return_tensors="pt")
# inputs["input_ids"]     → shape [1, 23]  (text tokens)
# inputs["pixel_values"]  → shape [1, 3, 336, 336]  (image tensor)
```

### Step 5: Visual Encoding (CLIP ViT-L/14)
```
pixel_values [1, 3, 336, 336]
  → ViT splits into 24×24 patches
  → Each patch → 1024-dim vector
  → 576 visual tokens [1, 576, 1024]
  → These encode: lung fields, heart silhouette, rib patterns
```

### Step 6: Multi-Modal Projection
```
576 visual tokens [1, 576, 1024]
  → 2-layer MLP
  → 576 tokens in Llama's embedding space [1, 576, 4096]
  → Now the language model can "read" the image
```

### Step 7: Language Model Generation
```
Input: [576 visual tokens] + [23 text tokens] = 599 tokens
  → Llama-2-7B + LoRA adapters
  → Attention over ALL tokens (image + text)
  → Generate one token at a time:
    
    Token 1: "No"       (looks at lungs → clear)
    Token 2: "acute"    (no emergency findings)
    Token 3: "cardio"   (heart looks normal)
    Token 4: "pulmonary" 
    Token 5: "abnormality"
    Token 6: "."        (done)
    Token 7: [EOS]      (stop generating)
```

### Step 8: Decoding
```python
answer_ids = output_ids[0, 599:]  # Skip input tokens
answer = processor.decode(answer_ids, skip_special_tokens=True)
# → "No acute cardiopulmonary abnormality."
```

### Step 9: Post-Processing
```python
answer = clean_answer(answer)
# → "No acute cardiopulmonary abnormality."  (already clean)
```

### Step 10: Output
```python
return VLMOutput(
    answer="No acute cardiopulmonary abnormality.",
    raw_output="No acute cardiopulmonary abnormality.",
    generation_time_sec=14.07,
    input_token_count=599,
    output_token_count=6,
)
```

---

## 13. How Local and Kaggle Differ

| Aspect | Local (your PC) | Kaggle |
|---|---|---|
| GPU | None (CPU only) | Tesla T4 (16 GB VRAM) |
| dtype | float16 | 4-bit quantized |
| VRAM | N/A (uses RAM) | ~4 GB |
| RAM needed | ~14 GB | ~2 GB |
| Speed | 10-15 min/image | 14 sec/image |
| Code source | `scripts/inference.py` imports `src/` | `kaggle/kaggle_inference.py` is self-contained |
| Adapter path | `checkpoints/llava-medical-vqa/final_adapter/` | `/kaggle/input/llava-medical-adapter/` |
| Dataset path | `data/openi/` | `/kaggle/input/chest-xrays-indiana-university/` |

---

## 14. Summary

Phase 1 implements a working multimodal medical VQA system:

1. **Dataset** → OpenI chest X-rays parsed from CSV into structured samples
2. **Model** → LLaVA-1.5-7B loaded with 4-bit quantization and LoRA adapters
3. **Training** → QLoRA fine-tuning on Kaggle T4, producing a 76 MB adapter
4. **Inference** → Image + question → processor → vision encoder → projector → LLM → answer
5. **Evaluation** → 5 diverse samples, JSON/CSV/Markdown outputs

The architecture is modular — every component can be swapped without rewriting other parts. Phase 2 will add retrieval to this same architecture.
