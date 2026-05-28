# Phase 2 - ColQwen2 Retrieval + LLaVA Generation Report

**Date:** 2026-05-22T02:15:42.126316
**GPU:** Tesla T4 | **VRAM:** 8.58 GB

## Pipeline Architecture

```
OpenI Image-Report Pairs
  -> ColQwen2 Encoding (multi-vector embeddings)
  -> Saved Retrieval Index

User Query (text or image+text)
  -> ColQwen2 Query Embedding
  -> MaxSim Similarity Search
  -> Top 3 Retrieved Cases
  -> Context Builder (evidence formatting)
  -> LLaVA-1.5-7B (4-bit) + LoRA
  -> Grounded Medical Answer
```

---

## Index Summary

| Metric | Value |
|--------|-------|
| Documents Indexed | 3826 |
| Skipped | 0 |
| Embedding Dim | 128 |
| Index Build Time | 14734.65s |
| ColQwen2 Model | `vidore/colqwen2-v1.0-hf` |

---

## Evaluation Summary

| Metric | Value |
|--------|-------|
| Total Queries | 3 |
| Successful | 3 |
| Text-Only Queries | 0 |
| Image+Text Queries | 3 |
| Avg Retrieval Time | 5.8s |
| Avg Generation Time | 7.53s |
| Avg Total Time | 13.33s |

---

## Detailed Results

### Query 1 (image_text)

**Query:** What are the key findings in this chest X-ray?

**Answer:**
> Clear lung fields without bibasilar airspace disease or nodules. Heart size appears normal for age with mildly increased left atrial size which may be due to hypertension rather than congestive heart failure. Mediastinal contour also appears normal. Right lower lobe granulomas seen as well as some interstitial fibrosis. This could represent prior tuberculosis. Overall, there do not appear any significant radiographic abnormalities that would require further evaluation such as CT scan.

**Retrieved Cases:** 1, 1785, 1738
**Scores:** 761.0, 661.4942, 659.9733
**Image Source:** query

<details><summary>Evidence #1 - Case 1 (score: 761.0)</summary>

- **Findings:** The cardiac silhouette and mediastinum size are within normal limits. There is no pulmonary edema. There is no focal consolidation. There are no of a pleural effusion. There is no evidence of pneumothorax
- **Impression:** Normal chest x-

</details>

<details><summary>Evidence #2 - Case 1785 (score: 661.4942)</summary>

- **Findings:** Cardiac and mediastinal contours are within normal limits. Granulomatous calcifications in the paratracheal region. Mild streaky scarring in the right upper lobe. No active pneumonia. Bony structures are intact
- **Impression:** No lobar pneumonia is present

</details>

<details><summary>Evidence #3 - Case 1738 (score: 659.9733)</summary>

- **Findings:** Cardiac and mediastinal contours are within normal limits. Mild aortic tortuosity. The lungs are clear. Bony structures are intact
- **Impression:** No acute findings

</details>

*Retrieval: 5.86s | Generation: 13.84s | Total: 19.7s*

---

### Query 2 (image_text)

**Query:** Is there cardiomegaly or any cardiac abnormality?

**Answer:**
> Normal heart size and medial contours. Clear lungs without active airspace disease. Stable findings compared to prior studies.

**Retrieved Cases:** 4, 1689, 315
**Scores:** 761.0, 670.7937, 670.7493
**Image Source:** query

<details><summary>Evidence #1 - Case 4 (score: 761.0)</summary>

- **Findings:** There are diffuse bilateral interstitial and alveolar opacities consistent with chronic obstructive lung disease and bullous emphysema. There are irregular opacities in the left lung apex, that could represent a cavitary lesion in the left lung apex.There are streaky opacities in the right upper lobe, scarring. The cardiomediastinal silhouette is normal in size and contour. There is no pneumothorax or large pleural effusion
- **Impression:** 1. Bullous emphysema and interstitial fibrosis. 2. Probably scarring in the left apex, although difficult to exclude a cavitary lesion. 3. Opacities in the bilateral upper lobes could represent scarring, however the absence of comparison exam, recommend short interval followup radiograph or CT thorax to document resolution

</details>

<details><summary>Evidence #2 - Case 1689 (score: 670.7937)</summary>

- **Findings:** Heart size and mediastinal contours appear within normal limits. Pulmonary vascularity is within normal limits. No focal consolidation, suspicious pulmonary opacity, pneumothorax or definite pleural effusion. Visualized osseous structures appear intact
- **Impression:** No acute cardiopulmonary abnormality

</details>

<details><summary>Evidence #3 - Case 315 (score: 670.7493)</summary>

- **Findings:** None
- **Impression:** Heart size is normal. Lungs are clear. Old fusion of approximately T9-T10

</details>

*Retrieval: 5.76s | Generation: 4.09s | Total: 9.85s*

---

### Query 3 (image_text)

**Query:** Are there signs of pleural effusion or pneumonia?

**Answer:**
> No evidence suggestive of pleuritis, pneumonitis, or edema. Normal chest X-ray appearance with no infiltrates.

**Retrieved Cases:** 7, 2140, 3284
**Scores:** 740.0, 662.9229, 662.0481
**Image Source:** query

<details><summary>Evidence #1 - Case 7 (score: 740.0)</summary>

- **Findings:** The cardiac contours are normal. basilar atelectasis. The lungs are clear. Thoracic spondylosis. Lower cervical arthritis
- **Impression:** Basilar atelectasis. No confluent lobar consolidation or pleural effusion

</details>

<details><summary>Evidence #2 - Case 2140 (score: 662.9229)</summary>

- **Findings:** The lungs are clear. There are calcified left hilar lymph . The heart and mediastinum are normal. The skeletal structures are notable for an old apparent fracture at T12-L1 or congenital fusion unchanged from the prior study
- **Impression:** 1. No acute pulmonary disease. 2. Possible old injury or developmental anomaly partially T12-L1

</details>

<details><summary>Evidence #3 - Case 3284 (score: 662.0481)</summary>

- **Findings:** The heart size and pulmonary vascularity appear within normal limits. The lungs are free of focal airspace disease. No pleural effusion or pneumothorax is seen
- **Impression:** 1. No evidence of active disease

</details>

*Retrieval: 5.77s | Generation: 4.66s | Total: 10.44s*

---
