# Phase 2 - ColQwen2 Retrieval + LLaVA Generation Report

**Date:** 2026-05-21T20:37:20.490190
**GPU:** Tesla T4 | **VRAM:** 13.02 GB

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
| Documents Indexed | 50 |
| Skipped | 0 |
| Embedding Dim | 128 |
| Index Build Time | 239.27s |
| ColQwen2 Model | `vidore/colqwen2-v1.0-hf` |

---

## Evaluation Summary

| Metric | Value |
|--------|-------|
| Total Queries | 5 |
| Successful | 5 |
| Text-Only Queries | 3 |
| Image+Text Queries | 2 |
| Avg Retrieval Time | 2.71s |
| Avg Generation Time | 4.34s |
| Avg Total Time | 7.07s |

---

## Detailed Results

### Query 1 (text_only)

**Query:** What are the key findings in this chest X-ray?

**Answer:**
> Clear lungs without focal airspace disease. Heart size appears normal for age. Mediastinal contours appear stable. Right lower lung base opacity represents calcified granulation tissue. Left basilar opacities could be due to underlying pulmonary edema secondary to congestive heart failure but also can be related to other causes such as mild emphysematous change. Recommend follow up echocardiography if there's concern about fluid overload.

**Retrieved Cases:** 31, 9, 38
**Scores:** 15.2067, 15.1985, 15.1868
**Image Source:** retrieved (case 31)

<details><summary>Evidence #1 - Case 31 (score: 15.2067)</summary>

- **Findings:** None
- **Impression:** Suggestion of slightly more prominent interstitial markings, which may represent some bronchitic/bronchiolitis changes. No suspicious nodules, pneumonia, effusions, or CHF. Stable mediastinal contour

</details>

<details><summary>Evidence #2 - Case 9 (score: 15.1985)</summary>

- **Findings:** The examination consists of frontal and lateral radiographs of the chest. The cardiac silhouette is not enlarged. There has been apparent interval increase in low density convexity at the left cardiophrenic . Calcified granuloma is again seen in the right upper lobe. There is no consolidation, pleural effusion or pneumothorax
- **Impression:** Increased size of density in the left cardiophrenic . Primary differential considerations include increased size of prominent epicardial fat, pericardial mass, pleural mass or cardiac aneurysm. CT chest with contrast is recommended. These findings and recommendations were discussed . by Dr. telephone at p.m. /. Dr. <>technologist receipt of the results

</details>

<details><summary>Evidence #3 - Case 38 (score: 15.1868)</summary>

- **Findings:** Lungs are clear. There is no pneumothorax or pleural effusion. The heart and mediastinum are within normal limits. Bony structures are intact
- **Impression:** No acute cardiopulmonary process

</details>

*Retrieval: 0.86s | Generation: 11.88s | Total: 12.79s*

---

### Query 2 (text_only)

**Query:** Is there cardiomegaly or any cardiac abnormality?

**Answer:**
> Normal heart size and no acute radiographic cardiopathy identified.

**Retrieved Cases:** 27, 21, 12
**Scores:** 15.2753, 15.0643, 15.0054
**Image Source:** retrieved (case 27)

<details><summary>Evidence #1 - Case 27 (score: 15.2753)</summary>

- **Findings:** Lungs are overall hyperexpanded with flattening of the diaphragms. No focal consolidation. No pleural effusions or pneumothoraces. Heart and mediastinum of normal size and contour. Degenerative changes in the thoracic spine
- **Impression:** Hyperexpanded but clear lungs

</details>

<details><summary>Evidence #2 - Case 21 (score: 15.0643)</summary>

- **Findings:** None
- **Impression:** Heart size normal. Mediastinal silhouettes and pulmonary vascularity are within normal limits. Calcified lingular granuloma. No focal consolidations or pleural effusions. No pneumothorax. Breast implants there is a moderate wedge deformity of the midthoracic vertebrae, T6, age-indeterminate

</details>

<details><summary>Evidence #3 - Case 12 (score: 15.0054)</summary>

- **Findings:** Lungs are clear bilaterally. Cardiac and mediastinal silhouettes are normal. Pulmonary vasculature is normal. No pneumothorax or pleural effusion. No acute bony abnormality
- **Impression:** No acute cardiopulmonary abnormality

</details>

*Retrieval: 0.75s | Generation: 2.95s | Total: 3.75s*

---

### Query 3 (text_only)

**Query:** Describe any pulmonary abnormalities visible.

**Answer:**
> No significant findings.

**Retrieved Cases:** 45, 25, 5
**Scores:** 15.0347, 14.993, 14.9765
**Image Source:** retrieved (case 45)

<details><summary>Evidence #1 - Case 45 (score: 15.0347)</summary>

- **Findings:** Stable cardiomegaly with vascular prominence without overt edema. No focal airspace disease. No large pleural effusion or pneumothorax. The are intact
- **Impression:** Stable cardiomegaly without overt pulmonary edema

</details>

<details><summary>Evidence #2 - Case 25 (score: 14.993)</summary>

- **Findings:** The heart is within normal limits in size. Surgical suture material projects over the right lung apex. The lungs are hyperlucent and hyperinflated compatible with emphysema. There is left lower lobe airspace disease identified. There is moderate left pleural effusion and small right pleural effusion. No visualized pneumothorax
- **Impression:** 1. Left lower lobe airspace disease and bilateral pleural effusions, left greater than right. This may be secondary to inhalational injury. Recommend followup to ensure complete resolution

</details>

<details><summary>Evidence #3 - Case 5 (score: 14.9765)</summary>

- **Findings:** The cardiomediastinal silhouette and pulmonary vasculature are within normal limits. There is no pneumothorax or pleural effusion. There are no focal areas of consolidation. Cholecystectomy clips are present. Small T-spine osteophytes. There is biapical pleural thickening, unchanged from prior. Mildly hyperexpanded lungs
- **Impression:** No acute cardiopulmonary abnormality

</details>

*Retrieval: 0.74s | Generation: 2.07s | Total: 2.86s*

---

### Query 4 (image_text)

**Query:** Are the lungs clear? Describe any opacities or effusions.

**Answer:**
> Clear lungs with no active pulmonary findings.

**Retrieved Cases:** 10, 30, 51
**Scores:** 761.0, 662.3975, 661.4435
**Image Source:** query

<details><summary>Evidence #1 - Case 10 (score: 761.0)</summary>

- **Findings:** The cardiomediastinal silhouette is within normal limits for size and contour. The lungs are normally inflated without evidence of focal airspace disease, pleural effusion, or pneumothorax. Stable calcified granuloma within the right upper lung. No acute bone abnormality
- **Impression:** No acute cardiopulmonary process

</details>

<details><summary>Evidence #2 - Case 30 (score: 662.3975)</summary>

- **Findings:** Lungs are clear without focal consolidation, effusion, or pneumothorax. Normal heart size. Negative for pneumoperitoneum. Bony thorax and soft tissue grossly unremarkable
- **Impression:** Negative acute cardiopulmonary abnormality

</details>

<details><summary>Evidence #3 - Case 51 (score: 661.4435)</summary>

- **Findings:** Heart size is normal and cardiomediastinal silhouette is normal. There are scattered calcified granulomas throughout both lung . Lungs are clear bilaterally otherwise. No bony or soft tissue abnormalities
- **Impression:** No acute cardiopulmonary abnormality

</details>

*Retrieval: 5.79s | Generation: 2.63s | Total: 8.42s*

---

### Query 5 (image_text)

**Query:** What is the overall radiological impression?

**Answer:**
> Normal chest X-ray.

**Retrieved Cases:** 13, 4, 10
**Scores:** 761.0, 641.6029, 635.4353
**Image Source:** query

<details><summary>Evidence #1 - Case 13 (score: 761.0)</summary>

- **Findings:** The cardiac silhouette is borderline enlarged. Otherwise, there is no focal opacity. Mediastinal contours are within normal limits. There is no large pleural effusion. No pneumothorax
- **Impression:** Borderline enlargement of the cardiac silhouette without acute pulmonary disease

</details>

<details><summary>Evidence #2 - Case 4 (score: 641.6029)</summary>

- **Findings:** There are diffuse bilateral interstitial and alveolar opacities consistent with chronic obstructive lung disease and bullous emphysema. There are irregular opacities in the left lung apex, that could represent a cavitary lesion in the left lung apex.There are streaky opacities in the right upper lobe, scarring. The cardiomediastinal silhouette is normal in size and contour. There is no pneumothorax or large pleural effusion
- **Impression:** 1. Bullous emphysema and interstitial fibrosis. 2. Probably scarring in the left apex, although difficult to exclude a cavitary lesion. 3. Opacities in the bilateral upper lobes could represent scarring, however the absence of comparison exam, recommend short interval followup radiograph or CT thorax to document resolution

</details>

<details><summary>Evidence #3 - Case 10 (score: 635.4353)</summary>

- **Findings:** The cardiomediastinal silhouette is within normal limits for size and contour. The lungs are normally inflated without evidence of focal airspace disease, pleural effusion, or pneumothorax. Stable calcified granuloma within the right upper lung. No acute bone abnormality
- **Impression:** No acute cardiopulmonary process

</details>

*Retrieval: 5.39s | Generation: 2.15s | Total: 7.54s*

---
