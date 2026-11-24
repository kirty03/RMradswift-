"""Step 6 - simulate RadSwift and the other countermeasures.

Input : config/workflow_assumptions.yaml, results/calibration.json,
        results/metrics.json (from Step 5; placeholders are used if missing)
Output: results/simulation_summary.csv, results/simulation_tests.json,
        results/sensitivity.csv, results/fig_scenarios.png,
        results/fig_abnormal_wait.png, results/fig_workload.png

Every scenario is run 30 times with different random seeds. Within one seed
all scenarios see exactly the same studies, so they can be compared fairly.

Run:  python src/step6_simulate_radswift.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import workflow_sim as ws

RESULTS = ws.ROOT / "results"
SEEDS = range(1, 31)
TARGET_H = 6.0
STAGE_COLS = ["acquisition_h", "queue_h", "interpretation_h", "finalization_h", "distribution_h"]


def measured_model_numbers():
    """Sensitivity, specificity and draft acceptance from our own model evaluation."""
    path = RESULTS / "metrics.json"
    if not path.exists():
        print("WARNING: results/metrics.json not found. Using PLACEHOLDER accuracy numbers from the YAML file.")
        return {}, "placeholder"
    m = json.loads(path.read_text())
    return {"triage_sensitivity": m["triage"]["sensitivity"],
            "triage_specificity": m["triage"]["specificity"],
            "draft_acceptance": m["fine_tuned"]["all"]["draft_acceptance_proxy"]}, "measured"


def build_scenarios(model_numbers):
    triage = {"ai_triage": True, **model_numbers}
    draft = {"ai_draft": True, **model_numbers}
    return {
        "S0 Baseline": {},
        "S1 AI triage only": triage,
        "S2 AI drafting only (evidence)": draft,
        "S3 RadSwift (evidence only)": {**triage, **draft},
        "S4 RadSwift + assumed workflow gains": {**triage, **draft, "reporting_lag_reduction": 0.25,
                                                 "finalization_reduction": 0.25, "distribution_reduction": 0.50},
        "S5 24/7 radiologist cover": {"evening_staff": 3, "night_staff": 1},
        "S6 Hypothetical: drafts for all modalities": {**triage, **draft, "draft_all_modalities": True},
    }


def run(scenario, config=None):
    """One row per seed: stage means, total TAT and chest X-ray queue waits."""
    rows = []
    for seed in SEEDS:
        df = ws.simulate(config, scenario, seed)
        df = df[~df["in_warmup"]]
        chest = df[df["is_chest_xray"]]
        row = df[STAGE_COLS + ["tat_h"]].mean().to_dict()
        row["abnormal_chest_queue_h"] = chest[chest["truly_abnormal"]]["queue_h"].mean()
        row["normal_chest_queue_h"] = chest[~chest["truly_abnormal"]]["queue_h"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def compare(base, other):
    """Paired comparison across the 30 seeds, plus the 6-hour target check."""
    diff = other["tat_h"] - base["tat_h"]
    ci = stats.t.interval(0.95, len(diff) - 1, loc=diff.mean(), scale=stats.sem(diff))
    _, p_pair = stats.ttest_rel(other["tat_h"], base["tat_h"])
    _, p_target = stats.ttest_1samp(other["tat_h"], TARGET_H, alternative="less")
    return {"mean_difference_h": float(diff.mean()), "ci95_low": float(ci[0]), "ci95_high": float(ci[1]),
            "paired_t_p": float(p_pair), "target_6h_p": float(p_target),
            "reaches_6h_target": bool(other["tat_h"].mean() <= TARGET_H and p_target < 0.05)}


def main():
    RESULTS.mkdir(exist_ok=True)
    model_numbers, source = measured_model_numbers()
    scenarios = build_scenarios(model_numbers)

    runs = {name: run(sc) for name, sc in scenarios.items()}
    base = runs["S0 Baseline"]
    gap = base["tat_h"].mean() - TARGET_H

    summary, tests = [], {"model_numbers_source": source, "model_numbers": model_numbers, "replications": len(SEEDS)}
    for name, r in runs.items():
        row = {"scenario": name, **r.mean().round(2).to_dict()}
        if name != "S0 Baseline":
            c = compare(base, r)
            tests[name] = c
            row.update({"tat_change_h": round(c["mean_difference_h"], 2),
                        "ci95": f"[{c['ci95_low']:.2f}, {c['ci95_high']:.2f}]",
                        "share_of_gap_closed_pct": round(-100 * c["mean_difference_h"] / gap, 1),
                        "reaches_6h_target": c["reaches_6h_target"]})
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary["model_numbers_source"] = source      # "measured" or "placeholder": never present placeholder rows as results
    summary.to_csv(RESULTS / "simulation_summary.csv", index=False)
    (RESULTS / "simulation_tests.json").write_text(json.dumps(tests, indent=2))
    print(summary.to_string(index=False))

    # ---- sensitivity: how much do the answers depend on our assumptions? ----
    s3 = scenarios["S3 RadSwift (evidence only)"]
    sens = []
    for acceptance in [0.3, 0.5, 0.7, 0.9]:
        d = run({**s3, "draft_acceptance": acceptance})["tat_h"].mean() - base["tat_h"].mean()
        sens.append({"varied": "draft_acceptance", "value": acceptance, "tat_change_h": round(d, 3)})
    for reduction in [0.05, 0.10, 0.155, 0.20, 0.30]:
        d = run({**s3, "read_time_reduction": reduction})["tat_h"].mean() - base["tat_h"].mean()
        sens.append({"varied": "read_time_reduction", "value": reduction, "tat_change_h": round(d, 3)})
    workload_rows = []
    for scale in [1.0, 1.1, 1.2]:
        b = run({}, {"workload_scale": scale})["tat_h"].mean()
        r = run(s3, {"workload_scale": scale})["tat_h"].mean()
        sens.append({"varied": "workload_scale", "value": scale, "tat_change_h": round(r - b, 3)})
        workload_rows.append((scale, b, r))
    pd.DataFrame(sens).to_csv(RESULTS / "sensitivity.csv", index=False)
    print(pd.DataFrame(sens).to_string(index=False))

    # ---- charts ----
    fig, ax = plt.subplots(figsize=(11, 5))
    bottom = np.zeros(len(summary))
    for col, colour in zip(STAGE_COLS, ["#14205A", "#E8912D", "#3C52C3", "#8FA6D9", "#B9BCC6"]):
        ax.barh(summary["scenario"], summary[col], left=bottom, label=col.replace("_h", ""), color=colour)
        bottom += summary[col].values
    ax.axvline(TARGET_H, color="black", linestyle="--", label="6 h target")
    ax.invert_yaxis()
    ax.set_xlabel("mean hours (30 replications)")
    ax.set_title("Turnaround by scenario, split by stage")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=6, fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(RESULTS / "fig_scenarios.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(2)
    ax.bar(x - 0.2, [base["abnormal_chest_queue_h"].mean(), base["normal_chest_queue_h"].mean()], 0.4, label="Baseline", color="#B9BCC6")
    s1 = runs["S1 AI triage only"]
    ax.bar(x + 0.2, [s1["abnormal_chest_queue_h"].mean(), s1["normal_chest_queue_h"].mean()], 0.4, label="AI triage", color="#3C52C3")
    ax.set_xticks(x); ax.set_xticklabels(["Abnormal chest X-rays", "Normal chest X-rays"])
    ax.set_ylabel("mean worklist wait (hours)"); ax.set_title("Who waits less with AI triage?"); ax.legend()
    fig.tight_layout(); fig.savefig(RESULTS / "fig_abnormal_wait.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([w[0] for w in workload_rows], [w[2] - w[1] for w in workload_rows], marker="o", color="#3C52C3")
    ax.set_xlabel("workload (1.0 = baseline)"); ax.set_ylabel("TAT change with RadSwift, evidence only (hours)")
    ax.set_title("RadSwift helps more when radiologists are busier")
    fig.tight_layout(); fig.savefig(RESULTS / "fig_workload.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
