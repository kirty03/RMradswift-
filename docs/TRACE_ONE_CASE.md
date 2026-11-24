# One example, start to finish

## A. One chest X-ray study: uid 25

**1. Raw data (Kaggle).** Two files mention this study.

`indiana_projections.csv`:

| uid | filename | projection |
|---|---|---|
| 25 | 25_IM-1024-2001.dcm.png | Frontal |
| 25 | 25_IM-1024-3001.dcm.png | Lateral |

`indiana_reports.csv`, row with uid 25:

- Problems: `Sutures;Lung, Hyperlucent;Lung;Pulmonary Emphysema;Airspace Disease;Pleural Effusion;Pleural Effusion`
- findings: "The heart is within normal limits in size. Surgical suture material projects over the right lung
  apex. The lungs are hyperlucent and hyperinflated compatible with emphysema. There is left lower lobe airspace
  disease identified. There is moderate left pleural effusion and small right pleural effusion. No visualized
  pneumothorax."
- impression: "1. Left lower lobe airspace disease and bilateral pleural effusions, left greater than right.
  This may be secondary to inhalational injury. Recommend followup to ensure complete resolution."

**2. Step 3 (`build_study_table`).**
- Keeps the frontal image only, joined to the report by `uid`.
- Cleans the text (lower-case, `XXXX` removed).
- Has both sections, and Problems is not "No Indexing": the study is kept.
- Problems is not "normal": `label_abnormal = 1`.
- `report_text = "findings: the heart is within normal limits in size. ... impression: 1. left lower lobe
  airspace disease and bilateral pleural effusions, ..."`
- The split puts it in **test** (seed 42), so the model never trains on it.
- The image is shrunk and saved as `data/xray/images_512/25.jpg`.

**3. Step 4.** Not used: it is a test study.

**4. Step 5.**
- `EVAL_TRANSFORM` resizes `25.jpg` to 224 x 224 and scales the pixel values.
- ViT cuts the image into 196 patches of 16 x 16 pixels and returns 197 vectors (196 patches + 1 summary token).
- `abnormal_head` turns the summary token into the abnormal score.
- `project` resizes the 197 vectors; T5 reads them and writes the draft word by word.
- The row is saved in `results/test_predictions.csv` with `ai_draft` and `abnormal_score`.
- Our finding check reads the true report as {opacity, pleural_effusion}. The draft is counted as "acceptable"
  only if it states exactly the same findings.

> AI draft and abnormal score for uid 25: **copy them here from `results/test_predictions.csv` after the Kaggle
> run.** Do not write them from memory.

**5. Demo.** If uid 25 is among the 60 cases in `demo_bundle/`, it appears on the Worklist page, and on the
Report workspace page with its image, score, editable draft and the original report.

## B. One simulated imaging request: R00679

Created by Step 1 (simulated; no real patient).

`imaging_requests.csv`: R00679, patient P00766, CT, Chest, Routine.

`radiology_log.csv` and the stage each gap belongs to (Step 2, `add_stage_hours`):

| Timestamp | Value | Stage ending here | Hours |
|---|---|---|---|
| scan_completed | 2026-09-03 00:22 | | |
| images_available | 2026-09-03 01:46 | acquisition | 1.40 |
| read_started | 2026-09-03 12:02 | queue | 10.27 |
| preliminary_report | 2026-09-03 16:56 | interpretation | 4.90 |
| report_signed | 2026-09-03 19:36 | finalization | 2.67 |
| report_sent | 2026-09-03 21:24 | distribution | 1.80 |
| | | **turnaround** | **21.03** |

Why the queue wait is 10.27 h: the images arrived at 01:46, when no radiologist was reading. The session
starts at 08:00, and the studies left from the day before, plus all urgent ones, are read first. This request
reached the front at 12:02.

In Step 6, request R00679 exists in every scenario with the same arrival time and the same random durations.
Only what the scenario changes is different, which is why scenarios can be compared fairly.
