# Data

## data/workflow/ (synthetic, created by Step 1)

| File | One row is | Columns |
|---|---|---|
| `patients.csv` | a made-up patient | patient_id, patient_name, age_gender, city |
| `imaging_requests.csv` | one imaging order | request_id, patient_id, modality, body_part, priority |
| `radiology_log.csv` | the timeline of one order | request_id and six timestamps: scan_completed, images_available, read_started, preliminary_report, report_signed, report_sent |
| `workflow_merged.csv` | one order with everything joined (Step 2) | the above plus age, gender, five stage durations in hours, tat_h |

Links: `patients.patient_id = imaging_requests.patient_id`, `imaging_requests.request_id = radiology_log.request_id`.

The timestamps come from the simulator `src/workflow_sim.py` with the settings in
`config/workflow_assumptions.yaml`. Names are of the form "Patient 00001". Nothing here describes a real person.

## data/xray/ (real, public, created by Step 3; not stored in this repository)

Source: Indiana University chest X-ray collection, Open-i, U.S. National Library of Medicine
(Demner-Fushman et al., 2016, *JAMIA* 23(2)). Downloaded from Kaggle: `raddar/chest-xrays-indiana-university`.
The collection was de-identified by its publishers; removed words appear as `XXXX` in the reports.

Raw files: `indiana_reports.csv` (3,851 reports), `indiana_projections.csv` (7,466 images: 3,818 frontal,
3,648 lateral), and the PNG images. The two CSVs are linked by `uid`.

What Step 3 keeps:

| Rule | Studies left |
|---|---|
| Reports in the CSV | 3,851 |
| Has a frontal image (first one if several) | 3,689 |
| Has both a findings and an impression section | 3,193 |
| `Problems` is not "No Indexing" | 3,111 |

Split by study, seed 42, same abnormal share in each part: 2,488 train, 311 validation, 312 test.
`label_abnormal` is 0 when the `Problems` column says "normal" and 1 otherwise (62.6% abnormal).

Output: `xray_studies.csv` (uid, source_png, image_file, problems, findings, impression, report_text,
label_abnormal, split) and `images_512/<uid>.jpg`.

The dataset's licence does not allow sharing changed copies, so the prepared table, the images and
`demo_bundle/` are listed in `.gitignore`. Anyone can rebuild them with Step 3 and Step 5.
