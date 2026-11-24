"""workflow_sim.py - RadSwift radiology workflow simulator.

What it does
------------
Simulates imaging studies moving through five workflow stages:

  1. acquisition   scan completed -> images available on the worklist
  2. queue         waiting on the worklist until a radiologist starts reading
  3. interpretation  hands-on reading + reporting lag -> preliminary report
  4. finalization  preliminary report -> signed report
  5. distribution  signed report -> sent to the ordering doctor

Stages 1, 3 (lag part), 4 and 5 are random durations drawn around an assumed
mean. Stage 2 is NOT drawn from a distribution: it emerges, minute by minute,
from studies competing for the radiologists who are on duty.

Every assumption lives in config/workflow_assumptions.yaml.
Nothing here is hospital data.
"""
import heapq
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

MODALITIES = ["X-ray", "CT", "MRI"]

ROOT = Path(__file__).resolve().parents[1]
ASSUMPTIONS_FILE = ROOT / "config" / "workflow_assumptions.yaml"
CALIBRATION_FILE = ROOT / "results" / "calibration.json"


def load_assumptions():
    """Read the YAML file. Returns (department settings, baseline scenario settings).
    If Step 1 has saved a calibrated reading_end_hour, it replaces the YAML value."""
    with open(ASSUMPTIONS_FILE) as f:
        data = yaml.safe_load(f)
    config = data["department"]
    if CALIBRATION_FILE.exists():
        config["reading_end_hour"] = json.loads(CALIBRATION_FILE.read_text())["reading_end_hour"]
    return config, data["scenario_defaults"]


CONFIG, BASELINE = load_assumptions()


def _lognormal(rng, mean, cv, size):
    """Random positive durations with the requested mean and coefficient of variation."""
    sigma2 = np.log(1.0 + cv ** 2)
    mu = np.log(mean) - sigma2 / 2.0
    return rng.lognormal(mu, np.sqrt(sigma2), size)


def _staff_on_duty(minute, cfg, sc):
    """Number of radiologists reading at this minute of the simulation."""
    hour = (minute / 60.0) % 24
    if cfg["reading_start_hour"] <= hour < cfg["reading_end_hour"]:
        return cfg["radiologists"]
    if hour >= cfg["reading_end_hour"]:
        return sc["evening_staff"]
    return sc["night_staff"]


def simulate(config=None, scenario=None, seed=0):
    """Run one replication. Returns a DataFrame with one row per study.

    The same seed gives the same studies and the same underlying random numbers
    in every scenario, so scenarios can be compared study by study.
    """
    cfg = {**CONFIG, **(config or {})}
    sc = {**BASELINE, **(scenario or {})}
    rng = np.random.default_rng(seed)

    # ---------- 1. who arrives, and when (identical in every scenario) ----------
    share = np.array(cfg["hourly_arrival_share"], dtype=float)
    share = share / share.sum()
    scan_min = []
    for day in range(cfg["n_days"]):
        counts = rng.poisson(cfg["studies_per_day"] * share)
        for hour, k in enumerate(counts):
            scan_min.extend(day * 1440 + hour * 60 + rng.uniform(0, 60, k))
    scan_min = np.sort(np.array(scan_min))
    n = len(scan_min)

    probs = [cfg["modality_share"][m] for m in MODALITIES]
    modality = rng.choice(MODALITIES, size=n, p=probs)
    urgent = rng.random(n) < cfg["urgent_share"]
    is_chest = (modality == "X-ray") & (rng.random(n) < cfg["chest_share_of_xray"])
    abnormal = rng.random(n) < cfg["abnormal_prevalence"]

    # base random numbers, drawn once so every scenario reuses them
    u_flag = rng.random(n)
    u_accept = rng.random(n)
    cv = cfg["duration_cv"]
    lag_factor = np.where(urgent, cfg["urgent_lag_factor"], 1.0)
    lag_factor = lag_factor / lag_factor.mean()          # keeps the overall mean on target
    sm = cfg["stage_mean_h"]
    acquisition = _lognormal(rng, sm["acquisition"] * 60, cv, n)
    read_mean = np.array([cfg["read_minutes"][m] for m in modality]) * cfg["workload_scale"]
    read = _lognormal(rng, 1.0, cfg["read_cv"], n) * read_mean
    lag_mean = max(sm["interpretation"] * 60 - read_mean.mean(), 1.0)
    reporting_lag = _lognormal(rng, lag_mean, cv, n) * lag_factor
    finalization = _lognormal(rng, sm["finalization"] * 60, cv, n) * lag_factor
    distribution = _lognormal(rng, sm["distribution"] * 60, cv, n) * lag_factor

    # ---------- 2. what the scenario changes ----------
    ai_scope = np.ones(n, dtype=bool) if sc["draft_all_modalities"] else is_chest
    ai_flag = np.zeros(n, dtype=bool)
    if sc["ai_triage"]:
        p_flag = np.where(abnormal, sc["triage_sensitivity"], 1.0 - sc["triage_specificity"])
        ai_flag = is_chest & (u_flag < p_flag)
    draft_accepted = np.zeros(n, dtype=bool)
    if sc["ai_draft"]:
        draft_accepted = ai_scope & (u_accept < sc["draft_acceptance"])
        draft_rejected = ai_scope & ~draft_accepted
        read = np.where(draft_accepted, read * (1 - sc["read_time_reduction"]), read)
        read = np.where(draft_rejected, read * (1 + sc["rejected_draft_penalty"]), read)
        reporting_lag = np.where(draft_accepted, reporting_lag * (1 - sc["reporting_lag_reduction"]), reporting_lag)
        finalization = np.where(draft_accepted, finalization * (1 - sc["finalization_reduction"]), finalization)
    distribution = distribution * (1 - sc["distribution_reduction"])

    # ---------- 3. the worklist queue, minute by minute ----------
    ready = scan_min + acquisition                 # study appears on the worklist
    order = np.argsort(ready)
    start = np.full(n, np.nan)
    max_staff = max(cfg["radiologists"], sc["evening_staff"], sc["night_staff"])
    free_at = np.zeros(max_staff)                  # minute each radiologist becomes free
    waiting = []                                   # heap of (urgent rank, AI rank, ready time, study)
    nxt, served, minute = 0, 0, int(ready.min())
    while served < n:
        while nxt < n and ready[order[nxt]] <= minute:
            i = order[nxt]
            heapq.heappush(waiting, (0 if urgent[i] else 1, 0 if ai_flag[i] else 1, ready[i], i))
            nxt += 1
        for r in range(_staff_on_duty(minute, cfg, sc)):
            if waiting and free_at[r] <= minute:
                _, _, _, i = heapq.heappop(waiting)
                start[i] = minute
                free_at[r] = minute + read[i]
                served += 1
        minute += 1

    # ---------- 4. timestamps and stage durations ----------
    prelim = start + read + reporting_lag
    signed = prelim + finalization
    sent = signed + distribution
    out = pd.DataFrame({
        "request_id": [f"R{k:05d}" for k in range(1, n + 1)],
        "modality": modality,
        "priority": np.where(urgent, "Urgent", "Routine"),
        "is_chest_xray": is_chest,
        "truly_abnormal": abnormal,
        "ai_flag": ai_flag,
        "draft_accepted": draft_accepted,
        "scan_completed_min": scan_min,
        "images_available_min": ready,
        "read_started_min": start,
        "prelim_report_min": prelim,
        "report_signed_min": signed,
        "report_sent_min": sent,
        "acquisition_h": acquisition / 60,
        "queue_h": (start - ready) / 60,
        "interpretation_h": (prelim - start) / 60,
        "finalization_h": finalization / 60,
        "distribution_h": distribution / 60,
    })
    out["tat_h"] = (sent - scan_min) / 60
    out["in_warmup"] = scan_min < cfg["warmup_days"] * 1440
    return out


def stage_means(df):
    """Mean hours per stage and in total, warm-up days excluded."""
    d = df[~df["in_warmup"]]
    cols = ["acquisition_h", "queue_h", "interpretation_h", "finalization_h", "distribution_h", "tat_h"]
    return d[cols].mean()


def calibrate_reading_end(config=None, seeds=(101, 102, 103, 104, 105), lo=12.0, hi=20.0, steps=14):
    """Find the hour at which the daily reading session must end so that the
    baseline mean queue wait equals its target.

    A longer session means shorter waits, so a simple bisection works.
    """
    cfg = {**CONFIG, **(config or {})}
    target = cfg["stage_mean_h"]["queue"]
    for _ in range(steps):
        mid = (lo + hi) / 2
        wait = np.mean([stage_means(simulate({**cfg, "reading_end_hour": mid}, None, s))["queue_h"] for s in seeds])
        if wait > target:
            lo = mid          # waits too long -> the session must end later
        else:
            hi = mid
    return round((lo + hi) / 2, 2)
