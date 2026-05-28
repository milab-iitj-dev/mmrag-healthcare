# Phase 1 — Medical VQA Evaluation Report

**Date:** 2026-05-15T20:28:35.557353
**Model:** llava-hf/llava-1.5-7b-hf
**Adapter:** LoRA (r=16, α=32)
**Quantization:** 4-bit NF4
**GPU:** Tesla T4
**VRAM Usage:** 4.15 GB

---

## Summary

| Metric | Value |
|---|---|
| Samples Evaluated | 5/5 |
| Avg Inference Time | 7.26s |
| Total Time | 36.32s |
| GPU | Tesla T4 |
| VRAM | 4.15 GB |

---

## Results

### Sample 1 — UID: 1

**Image:** `1_IM-0001-3001.dcm.png`

**Question:** Briefly describe the key findings in this chest X-ray.

**Model Answer:**
> No acute cardiopulmonary abnormality identified on examination of thorax and upper extremities with no evidence for active disease or significant underlying pathology. Stable appearance compared to prior studies, without any new nodules noted at that time frame either.

**Ground Truth:**
> Normal chest x-.

*Inference: 8.61s | Tokens: 616 → 64*

---

### Sample 2 — UID: 2

**Image:** `2_IM-0652-1001.dcm.png`

**Question:** Is there cardiomegaly? Answer briefly.

**Model Answer:**
> No acute abnormality identified on this examination. Chest x-. Normal chest X-.Heart size is normal for age and body surface area, with no evidence of congestion or edema noted at the base stations.

**Ground Truth:**
> No acute pulmonary findings.

*Inference: 7.0s | Tokens: 610 → 64*

---

### Sample 3 — UID: 4

**Image:** `4_IM-2050-1001.dcm.png`

**Question:** Are the lungs clear or abnormal? State findings only.

**Model Answer:**
> No acute cardiopulmonary disease identified on this examination.

**Ground Truth:**
> 1. Bullous emphysema and interstitial fibrosis. 2. Probably scarring in the left apex, although difficult to exclude a cavitary lesion. 3. Opacities in the bilateral upper lobes could represent scarring, however the absence of comparison exam, recommend short interval followup radiograph or CT thora

*Inference: 6.93s | Tokens: 614 → 64*

---

### Sample 4 — UID: 7

**Image:** `7_IM-2263-1001.dcm.png`

**Question:** Is there pleural effusion? Answer in one sentence.

**Model Answer:**
> No acute cardiopulmonary abnormality.

**Ground Truth:**
> Basilar atelectasis. No confluent lobar consolidation or pleural effusion.

*Inference: 6.96s | Tokens: 612 → 64*

---

### Sample 5 — UID: 9

**Image:** `9_IM-2407-1001.dcm.png`

**Question:** What is the radiological impression? Be concise.

**Model Answer:**
> No acute cardiopulmonary abnormality identified on this examination.

**Ground Truth:**
> Increased size of density in the left cardiophrenic . Primary differential considerations include increased size of prominent epicardial fat, pericardial mass, pleural mass or cardiac aneurysm. CT chest with contrast is recommended. These findings and recommendations were discussed .  by Dr.   telep

*Inference: 6.82s | Tokens: 611 → 64*

---

## Pipeline Architecture

```
Chest X-ray (PNG)
  → PIL.Image.open().convert('RGB')
  → LlavaProcessor
      → CLIPImageProcessor: resize 336×336, normalize
      → LlamaTokenizer: tokenize prompt
  → LlavaForConditionalGeneration
      → CLIP ViT-L/14: 576 visual tokens (frozen)
      → Multi-Modal Projector: visual → LLM space
      → Llama-2-7B + LoRA (r=16, α=32): generate answer
  → Decode → Medical Answer
```

## Notes

- Model uses 4-bit NF4 quantization (~4 GB VRAM)
- LoRA adapters trained on OpenI chest X-ray dataset
- Greedy decoding with repetition_penalty=1.5
- Phase 1 only — no retrieval/RAG augmentation
