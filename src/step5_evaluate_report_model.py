"""Step 5 - evaluate the fine-tuned model on studies it has never seen.

Input : models/ (from Step 4), data/xray/xray_studies.csv, data/xray/images_512/
Output: results/test_predictions.csv   every test study: true report, AI draft, abnormal score
        results/metrics.json           all scores (this file feeds Step 6 and the slides)
        results/fig_roc_curve.png
        demo_bundle/                   up to 30 normal + 30 abnormal test cases for the demo

Three systems are compared on the same test studies:
  template    always answers with the most common training report (no AI at all)
  untrained   same network before fine-tuning
  fine_tuned  our model after Step 4

Run:  python src/step5_evaluate_report_model.py
"""
import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from report_model import XrayReportModel, load_model, pick_device
from text_metrics import clinical_scores, overlap_scores
from xray_dataset import EVAL_TRANSFORM, XrayDataset

ROOT = Path(__file__).resolve().parents[1]
XRAY_DIR = ROOT / "data" / "xray"
RESULTS = ROOT / "results"
SEED = 42


def predict(model, tokenizer, loader, device, write_reports=True):
    """Abnormal score (and, if asked, a generated report) for every study in the loader."""
    reports, scores = [], []
    model.eval()
    with torch.no_grad():
        for pixel_values, _, _ in loader:
            pixel_values = pixel_values.to(device)
            if write_reports:
                token_ids, score = model.generate(pixel_values)
                reports += tokenizer.batch_decode(token_ids, skip_special_tokens=True)
            else:
                score = torch.sigmoid(model.encode_image(pixel_values)[1])
            scores += score.cpu().tolist()
    return reports, np.array(scores)


def score_reports(references, candidates, abnormal):
    """Text scores for all test studies, and for normal and abnormal studies separately."""
    references, candidates = np.array(references), np.array(candidates)
    out = {}
    for group, keep in [("all", np.ones(len(abnormal), bool)), ("normal", abnormal == 0), ("abnormal", abnormal == 1)]:
        refs, cands = list(references[keep]), list(candidates[keep])
        out[group] = {"n": int(keep.sum()), **overlap_scores(refs, cands), **clinical_scores(refs, cands)}
    return out


def best_threshold(labels, scores):
    """Youden's J: the cut-off where sensitivity + specificity is highest."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return float(thresholds[np.argmax(tpr - fpr)])


def triage_scores(labels, scores, threshold, n_boot=1000):
    """AUC, sensitivity and specificity on the test set, with bootstrap 95% intervals."""
    def compute(y, s):
        flagged = s >= threshold
        return (roc_auc_score(y, s), flagged[y == 1].mean(), (~flagged)[y == 0].mean())
    auc, sens, spec = compute(labels, scores)
    rng, boot = np.random.default_rng(SEED), []
    for _ in range(n_boot):
        pick = rng.integers(0, len(labels), len(labels))      # resample the test studies with replacement
        if 0 < labels[pick].sum() < len(pick):
            boot.append(compute(labels[pick], scores[pick]))
    low, high = np.percentile(np.array(boot), [2.5, 97.5], axis=0)
    return {"threshold": threshold, "auc": float(auc), "sensitivity": float(sens), "specificity": float(spec),
            "auc_ci95": [float(low[0]), float(high[0])], "sensitivity_ci95": [float(low[1]), float(high[1])],
            "specificity_ci95": [float(low[2]), float(high[2])]}


def image_shuffle_check(model, loader, device):
    """Report loss with the right image vs with another study's image.
    If the model really reads the image, the shuffled loss must be clearly higher."""
    matched, shuffled, batches = 0.0, 0.0, 0
    model.eval()
    with torch.no_grad():
        for pixel_values, labels, _ in loader:
            if len(pixel_values) < 2:
                continue
            pixel_values, labels = pixel_values.to(device), labels.to(device)
            matched += model(pixel_values, labels)[0].item()
            shuffled += model(pixel_values.roll(1, dims=0), labels)[0].item()   # neighbour's image
            batches += 1
    return {"loss_with_correct_image": matched / batches, "loss_with_wrong_image": shuffled / batches}


def save_demo_bundle(test, per_class=30):
    """Copy a balanced sample of test cases (image + texts + score) for the demo app."""
    bundle = ROOT / "demo_bundle"
    (bundle / "images").mkdir(parents=True, exist_ok=True)
    parts = [g.sample(min(per_class, len(g)), random_state=SEED) for _, g in test.groupby("label_abnormal")]
    cases = pd.concat(parts).sample(frac=1, random_state=SEED)
    for image_file in cases["image_file"]:
        shutil.copy(XRAY_DIR / "images_512" / image_file, bundle / "images" / image_file)
    columns = ["uid", "image_file", "problems", "label_abnormal", "report_text", "ai_draft", "abnormal_score"]
    cases[columns].to_csv(bundle / "demo_cases.csv", index=False)
    print(f"Saved {len(cases)} demo cases to {bundle}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="use only the first N test studies (quick check)")
    args = parser.parse_args()

    device = pick_device()
    studies = pd.read_csv(XRAY_DIR / "xray_studies.csv")
    train, val = studies[studies["split"] == "train"], studies[studies["split"] == "val"]
    test = studies[studies["split"] == "test"].head(args.limit).copy()
    labels_test = test["label_abnormal"].to_numpy()
    references = list(test["report_text"])

    model, tokenizer = load_model(ROOT / "models", device)
    image_dir = XRAY_DIR / "images_512"
    test_loader = DataLoader(XrayDataset(test, image_dir, tokenizer, EVAL_TRANSFORM), batch_size=16)
    val_loader = DataLoader(XrayDataset(val, image_dir, tokenizer, EVAL_TRANSFORM), batch_size=16)

    # --- fine-tuned model ---
    drafts, scores_test = predict(model, tokenizer, test_loader, device)
    _, scores_val = predict(model, tokenizer, val_loader, device, write_reports=False)
    threshold = best_threshold(val["label_abnormal"].to_numpy(), scores_val)   # chosen on validation, never on test
    metrics = {
        "n_test": int(len(test)),
        "fine_tuned": score_reports(references, drafts, labels_test),
        "triage": triage_scores(labels_test, scores_test, threshold),
        "image_shuffle_check": image_shuffle_check(model, test_loader, device),
    }
    test["ai_draft"], test["abnormal_score"] = drafts, scores_test.round(4)
    test["ai_flag"] = (scores_test >= threshold).astype(int)

    # --- baseline 1: one fixed template for everybody ---
    template = train["report_text"].mode().iloc[0]
    metrics["template_text"] = template
    metrics["template"] = score_reports(references, [template] * len(test), labels_test)

    # --- baseline 2: the same network before fine-tuning ---
    torch.manual_seed(SEED)
    untrained = XrayReportModel().to(device)
    untrained_drafts, untrained_scores = predict(untrained, tokenizer, test_loader, device)
    metrics["untrained"] = score_reports(references, untrained_drafts, labels_test)
    metrics["untrained"]["auc"] = float(roc_auc_score(labels_test, untrained_scores))
    test["untrained_draft"] = untrained_drafts

    RESULTS.mkdir(exist_ok=True)
    test.to_csv(RESULTS / "test_predictions.csv", index=False)
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "template_text"}, indent=2))

    fpr, tpr, _ = roc_curve(labels_test, scores_test)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="#3C52C3", label=f"RadSwift (AUC {metrics['triage']['auc']:.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="chance")
    ax.set_xlabel("false positive rate (1 - specificity)"); ax.set_ylabel("sensitivity")
    ax.set_title("Normal vs abnormal on the test set"); ax.legend()
    fig.tight_layout(); fig.savefig(RESULTS / "fig_roc_curve.png", dpi=150); plt.close(fig)

    save_demo_bundle(test)


if __name__ == "__main__":
    main()
