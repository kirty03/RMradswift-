"""Step 1 - generate the synthetic workflow dataset.

Input : config/workflow_assumptions.yaml
Output: results/calibration.json
        data/workflow/patients.csv, imaging_requests.csv, radiology_log.csv

The three CSV files copy how a hospital stores this information: a patient
registry, an order system and the radiology (RIS) log, linked by IDs.
Everything is simulated. No real patient appears anywhere.

Run:  python src/step1_generate_workflow_data.py
"""
import json

import numpy as np
import pandas as pd

import workflow_sim as ws

DATA_DIR = ws.ROOT / "data" / "workflow"
START = pd.Timestamp("2026-09-01 00:00")
BODY_PARTS = {"X-ray": ["Hand", "Knee", "Spine", "Abdomen", "Shoulder"],
              "CT": ["Head", "Abdomen", "Chest"],
              "MRI": ["Brain", "Spine", "Knee"]}
CITIES = ["Thunder Bay", "Toronto", "Ottawa", "Sudbury", "Kenora"]
N_PATIENTS = 3000


def calibrate():
    """Find when the daily reading session must end so that the baseline
    queue wait matches its target, then save that one number."""
    end_hour = ws.calibrate_reading_end()
    ws.CALIBRATION_FILE.parent.mkdir(exist_ok=True)
    ws.CALIBRATION_FILE.write_text(json.dumps({
        "reading_end_hour": end_hour,
        "target_queue_wait_h": ws.CONFIG["stage_mean_h"]["queue"],
        "note": "Calibrated by bisection in workflow_sim.calibrate_reading_end()",
    }, indent=2))
    ws.CONFIG["reading_end_hour"] = end_hour
    return end_hour


def to_time(minutes):
    """Minutes since the start of the simulation -> timestamp text."""
    return (START + pd.to_timedelta(np.round(minutes), unit="min")).dt.strftime("%Y-%m-%d %H:%M")


def build_tables(studies, seed=1):
    """Split the simulator output into the three linked tables."""
    rng = np.random.default_rng(seed)
    n = len(studies)

    patient_ids = [f"P{k:05d}" for k in range(1, N_PATIENTS + 1)]
    ages = rng.integers(18, 95, N_PATIENTS)
    sexes = rng.choice(["F", "M"], N_PATIENTS)
    patients = pd.DataFrame({
        "patient_id": patient_ids,
        "patient_name": [f"Patient {k:05d}" for k in range(1, N_PATIENTS + 1)],   # fake names
        "age_gender": [f"{a}/{s}" for a, s in zip(ages, sexes)],                 # composite field, as RIS exports often have
        "city": rng.choice(CITIES, N_PATIENTS),
    })

    body_part = [("Chest" if chest else rng.choice(BODY_PARTS[m]))
                 for m, chest in zip(studies["modality"], studies["is_chest_xray"])]
    requests = pd.DataFrame({
        "request_id": studies["request_id"],
        "patient_id": rng.choice(patient_ids, n),
        "modality": studies["modality"],
        "body_part": body_part,
        "priority": studies["priority"],
    })

    log = pd.DataFrame({
        "request_id": studies["request_id"],
        "scan_completed": to_time(studies["scan_completed_min"]),
        "images_available": to_time(studies["images_available_min"]),
        "read_started": to_time(studies["read_started_min"]),
        "preliminary_report": to_time(studies["prelim_report_min"]),
        "report_signed": to_time(studies["report_signed_min"]),
        "report_sent": to_time(studies["report_sent_min"]),
    })
    return patients, requests, log


def main():
    end_hour = calibrate()
    print(f"Calibrated end of reading session: {end_hour:.2f} h (about {int(end_hour)}:{int(end_hour % 1 * 60):02d})")

    studies = ws.simulate(seed=1)
    studies = studies[~studies["in_warmup"]].reset_index(drop=True)   # drop the warm-up days
    means = studies[["acquisition_h", "queue_h", "interpretation_h", "finalization_h", "distribution_h", "tat_h"]].mean()
    print("Baseline stage means (hours):")
    print(means.round(2).to_string())

    patients, requests, log = build_tables(studies)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    patients.to_csv(DATA_DIR / "patients.csv", index=False)
    requests.to_csv(DATA_DIR / "imaging_requests.csv", index=False)
    log.to_csv(DATA_DIR / "radiology_log.csv", index=False)
    print(f"Saved {len(patients)} patients, {len(requests)} requests, {len(log)} log rows to {DATA_DIR}")


if __name__ == "__main__":
    main()
