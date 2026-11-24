# File guide

For every file: what it is for, what goes in, what comes out, the functions to know, and how to check it works.

## config/workflow_assumptions.yaml
- **Purpose:** the single place for every number the simulator uses. Each is tagged `[assumed]`, `[calibrated]`,
  `[measured]` or `[cited]`.
- **Sections:** `department` (how the simulated department works) and `scenario_defaults` (what a countermeasure
  can change; all off = baseline).
- **Check:** change `radiologists` from 11 to 12, rerun Step 6, and the baseline wait drops. Change it back.

## src/workflow_sim.py (helper: the simulator)
- **Purpose:** moves about 10,000 simulated studies through five stages and records six timestamps for each.
- **In:** the YAML file; `results/calibration.json` if it exists. **Out:** a table, one row per study.
- **Functions:**
  - `load_assumptions()` reads the YAML and the calibrated value.
  - `simulate(config, scenario, seed)` runs one 30-day replication. Arrivals and random durations are drawn
    first, then the scenario's changes are applied, then the worklist is processed minute by minute: whenever a
    radiologist on duty is free, they take the waiting study that is urgent first, AI-flagged second, oldest third.
  - `stage_means(df)` mean hours per stage, warm-up days removed.
  - `calibrate_reading_end()` bisection search for the hour the daily reading session must end so that the
    baseline queue wait is 10.33 h.
- **Key idea:** the queue wait is not typed in anywhere. It results from studies competing for radiologists.
  The other four stages are random durations around an assumed mean.
- **Check:** `python tests/test_pipeline.py`.

## src/step1_generate_workflow_data.py
- **Purpose:** calibrate the simulator, run the baseline once, save it as three linked tables.
- **Out:** `results/calibration.json`, `data/workflow/patients.csv`, `imaging_requests.csv`, `radiology_log.csv`.
- **Functions:** `calibrate()`, `to_time()` (minutes to timestamp text), `build_tables()`.
- **Check:** it prints stage means that add up to about 26.6 h.

## src/step2_analyze_workflow.py
- **Purpose:** treat the three tables as if they came from a hospital: merge, clean, measure.
- **Out:** `data/workflow/workflow_merged.csv`, `results/workflow_stage_means.csv`, `results/workflow_tests.json`,
  three charts.
- **Functions:** `load_and_merge()` (left join requests to patients, inner join to the log), `clean()` (splits
  "67/F" into age and gender, parses timestamps, drops the name), `add_stage_hours()`, `run_tests()`
  (one-sample t-test against 6 h; ANOVA and Kruskal-Wallis by modality and by priority; eta squared).
- **Honest reading:** the t-test against 6 h only confirms that the data we generated is far from the target.
  It is a check of the baseline, not a discovery. Modality shows no difference because we did not build one in;
  priority differs because urgent studies jump the queue.

## src/text_metrics.py (helper)
- **Purpose:** clean report text and score generated reports.
- **Functions:** `clean_report_text()` (lower-case, removes `XXXX`, repairs "x-xxxx" to "x-ray"),
  `overlap_scores()` (BLEU-1, BLEU-4, ROUGE-L), `positive_findings()` (which of six findings a report states as
  present, with a negation rule), `clinical_scores()` (precision, recall, F1 over those findings, and the share
  of drafts whose findings match the reference exactly: the draft-acceptance proxy).
- **Limits:** six findings, simple negation rule, a normal report matches another normal report trivially.

## src/step3_prepare_xray_data.py
- **Purpose:** turn the raw Kaggle dataset into one clean table and small images.
- **In:** `indiana_reports.csv`, `indiana_projections.csv`, PNG images. **Out:** `data/xray/xray_studies.csv`,
  `data/xray/images_512/`, `results/xray_data_summary.json`.
- **Functions:** `find_file()` (searches the input folder; no hard-coded path), `build_study_table()` (all
  inclusion rules and the 80/10/10 split), `save_small_images()`.
- **Check:** the printed counts must read 3851, 3689, 3193, 3111.

## src/report_model.py (helper: the model)
- **Purpose:** defines `XrayReportModel`.
- **Parts:** `vit` (pretrained image encoder), `project` (new linear layer 768 to 512), `t5` (pretrained text
  model), `abnormal_head` (new linear layer 768 to 1).
- **Functions:** `encode_image()`, `forward()` (training: returns the report loss and the abnormal logit),
  `generate()` (writes a report by beam search), `save_model()`, `load_model()`.
- **Key line:** `self.t5(inputs_embeds=image_tokens, labels=labels)`. The image tokens are T5's only input, and
  the true report is passed only as `labels`.

## src/xray_dataset.py (helper)
- **Purpose:** gives the model one study at a time: image tensor, report token ids, abnormal label.
- **Details:** training images get a small random crop, rotation and brightness change; never a flip. Padding
  tokens in the labels are set to -100 so the loss ignores them.

## src/step4_train_report_model.py
- **Purpose:** fine-tuning.
- **In:** Step 3 output. **Out:** `models/`, `results/training_log.csv`, `results/fig_training_loss.png`.
- **Functions:** `run_epoch()` (one pass; learns when given an optimizer, only measures otherwise), `main()`.
- **Settings:** batch 16; up to 15 epochs; stop after 3 epochs without a better validation loss; AdamW with
  learning rate 2e-5 for ViT and 1e-4 for the rest; loss = report loss + abnormal/normal loss.
- **Check:** `--smoke` finishes in a minute or two; in the full run the validation loss falls, then flattens.

## src/step5_evaluate_report_model.py
- **Purpose:** measure the model on the 312 test studies.
- **Out:** `results/test_predictions.csv`, `results/metrics.json`, `results/fig_roc_curve.png`, `demo_bundle/`.
- **Functions:** `predict()`, `score_reports()` (all / normal / abnormal), `best_threshold()` (Youden's J on the
  validation set), `triage_scores()` (AUC, sensitivity, specificity, bootstrap 95% intervals),
  `image_shuffle_check()`, `save_demo_bundle()`.
- **Baselines:** `template` (the most common training report for everyone) and `untrained` (same network
  before fine-tuning).

## src/step6_simulate_radswift.py
- **Purpose:** the experiment. Seven scenarios, 30 replications each, same random studies in every scenario.
- **In:** YAML, calibration, `results/metrics.json`. **Out:** `results/simulation_summary.csv`,
  `simulation_tests.json`, `sensitivity.csv`, three charts.
- **Functions:** `measured_model_numbers()`, `build_scenarios()`, `run()`, `compare()` (paired t-test, 95%
  interval, one-sided test against 6 h).
- **Scenarios:** S0 baseline; S1 triage; S2 drafting; S3 both (RadSwift, published evidence only);
  S4 RadSwift plus assumed workflow gains; S5 radiologists round the clock (a non-AI alternative);
  S6 hypothetical drafts for all modalities.

## app/demo_app.py
- **Purpose:** the demo. Pages: Worklist, Report workspace, Turnaround simulation, About.
- **In:** `demo_bundle/`, `models/` (optional), `results/`.
- **Modes:** "Live model" when `models/radswift_model.pt` is present, otherwise "Saved predictions".

## notebooks/kaggle_train_and_evaluate.ipynb
- **Purpose:** runs Steps 3 to 6 on Kaggle and packs the outputs for download. It only calls the scripts.

## tests/test_pipeline.py
- **Purpose:** ten checks: timestamps in order; same seed, same result; baseline near 26.6 h; identical
  studies across scenarios; flagged studies wait less; drafts never slow reading; cleaning; negation rule;
  clinical scores; model output changes when the image changes.
