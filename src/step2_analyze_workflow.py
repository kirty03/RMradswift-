"""Step 2 - merge, clean and analyse the synthetic workflow data.

Input : data/workflow/patients.csv, imaging_requests.csv, radiology_log.csv
Output: data/workflow/workflow_merged.csv
        results/workflow_stage_means.csv, results/workflow_tests.json
        results/fig_stage_means.png, fig_tat_by_modality.png, fig_tat_by_priority.png

Run:  python src/step2_analyze_workflow.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "workflow"
RESULTS = ROOT / "results"
TARGET_H = 6.0
TIME_COLUMNS = ["scan_completed", "images_available", "read_started",
                "preliminary_report", "report_signed", "report_sent"]
STAGES = {  # stage name -> (start column, end column)
    "acquisition_h": ("scan_completed", "images_available"),
    "queue_h": ("images_available", "read_started"),
    "interpretation_h": ("read_started", "preliminary_report"),
    "finalization_h": ("preliminary_report", "report_signed"),
    "distribution_h": ("report_signed", "report_sent"),
}


def load_and_merge():
    patients = pd.read_csv(DATA_DIR / "patients.csv")
    requests = pd.read_csv(DATA_DIR / "imaging_requests.csv")
    log = pd.read_csv(DATA_DIR / "radiology_log.csv")
    # left join: keep every request even if a patient detail were missing
    merged = requests.merge(patients, on="patient_id", how="left")
    # inner join: keep only requests that have workflow timestamps
    return merged.merge(log, on="request_id", how="inner")


def clean(df):
    df = df.copy()
    df[["age", "gender"]] = df["age_gender"].str.split("/", expand=True)   # "67/F" -> 67, F
    df["age"] = df["age"].astype(int)
    for col in TIME_COLUMNS:
        df[col] = pd.to_datetime(df[col])
    return df.drop(columns=["patient_name", "age_gender"])                 # remove personal fields


def add_stage_hours(df):
    df = df.copy()
    for stage, (start, end) in STAGES.items():
        df[stage] = (df[end] - df[start]).dt.total_seconds() / 3600
    df["tat_h"] = (df["report_sent"] - df["scan_completed"]).dt.total_seconds() / 3600
    return df


def eta_squared(groups):
    """Share of the variation in TAT explained by the grouping (effect size for ANOVA)."""
    all_values = pd.concat(groups)
    between = sum(len(g) * (g.mean() - all_values.mean()) ** 2 for g in groups)
    total = ((all_values - all_values.mean()) ** 2).sum()
    return float(between / total)


def run_tests(df):
    results = {"n_studies": int(len(df)), "mean_tat_h": float(df["tat_h"].mean())}
    t, p = stats.ttest_1samp(df["tat_h"], TARGET_H, alternative="greater")
    results["baseline_vs_6h"] = {"test": "one-sample t-test, H1: mean TAT > 6 h", "t": float(t), "p": float(p)}
    for factor in ["modality", "priority"]:
        groups = [g["tat_h"] for _, g in df.groupby(factor)]
        f, p_anova = stats.f_oneway(*groups)
        h, p_kw = stats.kruskal(*groups)
        results[f"tat_by_{factor}"] = {
            "group_means_h": df.groupby(factor)["tat_h"].mean().round(2).to_dict(),
            "anova_F": float(f), "anova_p": float(p_anova),
            "kruskal_H": float(h), "kruskal_p": float(p_kw),
            "eta_squared": eta_squared(groups),
        }
    return results


def bar_chart(series, title, ylabel, filename):
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(series.index, series.values, color="#3C52C3")
    ax.bar_label(bars, fmt="%.2f")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(RESULTS / filename, dpi=150)
    plt.close(fig)


def main():
    RESULTS.mkdir(exist_ok=True)
    df = add_stage_hours(clean(load_and_merge()))
    df.to_csv(DATA_DIR / "workflow_merged.csv", index=False)

    stage_means = df[list(STAGES) + ["tat_h"]].mean().round(2)
    stage_means.to_csv(RESULTS / "workflow_stage_means.csv", header=["mean_hours"])
    print("Stage means (hours):")
    print(stage_means.to_string())

    tests = run_tests(df)
    (RESULTS / "workflow_tests.json").write_text(json.dumps(tests, indent=2))
    print(json.dumps(tests, indent=2))

    labels = [s.replace("_h", "") for s in STAGES]
    bar_chart(pd.Series(stage_means[list(STAGES)].values, index=labels),
              "Baseline: mean hours in each stage", "hours", "fig_stage_means.png")
    bar_chart(df.groupby("modality")["tat_h"].mean(), "Baseline: mean turnaround by modality", "hours", "fig_tat_by_modality.png")
    bar_chart(df.groupby("priority")["tat_h"].mean(), "Baseline: mean turnaround by priority", "hours", "fig_tat_by_priority.png")


if __name__ == "__main__":
    main()
