"""text_metrics.py - text cleaning, word-overlap scores and a simple clinical check.

Used by Step 3 (cleaning) and Step 5 (scoring). No model code in this file.
"""
import re

import numpy as np


def clean_report_text(text):
    """Lower-case, repair the de-identification marks (XXXX) and tidy spaces."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"x-xxxx", "x-ray", text)
    text = re.sub(r"\bxxxx\b", "", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"^[.,;: ]+", "", text)


def overlap_scores(references, candidates):
    """BLEU-1, BLEU-4 and ROUGE-L: how many words and word sequences the
    generated report shares with the radiologist's report."""
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu   # imported here so the
    from rouge_score import rouge_scorer                                    # rest works without them
    refs = [[r.split()] for r in references]
    cands = [c.split() for c in candidates]
    smooth = SmoothingFunction().method1
    scorer = rouge_scorer.RougeScorer(["rougeL"])
    return {
        "bleu1": corpus_bleu(refs, cands, weights=(1, 0, 0, 0), smoothing_function=smooth),
        "bleu4": corpus_bleu(refs, cands, weights=(0.25,) * 4, smoothing_function=smooth),
        "rougeL": float(np.mean([scorer.score(r, c)["rougeL"].fmeasure for r, c in zip(references, candidates)])),
    }


FINDING_TERMS = {
    "cardiomegaly": ["cardiomegaly", "enlarged heart", "heart is enlarged", "heart size is enlarged",
                     "cardiac enlargement", "enlarged cardiac"],
    "pleural_effusion": ["effusion"],
    "pneumothorax": ["pneumothorax"],
    "opacity": ["opacity", "opacities", "consolidation", "infiltrate", "airspace disease", "pneumonia"],
    "atelectasis": ["atelectasis", "atelectatic"],
    "edema": ["edema"],
}
NEGATION = re.compile(r"\b(no|not|without|negative for|free of|clear of|resolved)\b")


def positive_findings(report):
    """Findings the report states as present. A term counts as absent when a
    negation word comes before it in the same sentence ("no pleural effusion")."""
    found = set()
    for sentence in re.split(r"[.;]", report.lower()):
        negations = [m.start() for m in NEGATION.finditer(sentence)]
        for finding, terms in FINDING_TERMS.items():
            for term in terms:
                at = sentence.find(term)
                if at >= 0 and not any(n < at for n in negations):
                    found.add(finding)
    return found


def clinical_scores(references, candidates):
    """Micro precision, recall and F1 over the six findings, plus the share of
    reports whose finding set matches the reference exactly (draft-acceptance proxy)."""
    tp = fp = fn = same = 0
    for ref, cand in zip(references, candidates):
        r, c = positive_findings(ref), positive_findings(cand)
        tp += len(r & c); fp += len(c - r); fn += len(r - c); same += int(r == c)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"clinical_precision": precision, "clinical_recall": recall, "clinical_f1": f1,
            "draft_acceptance_proxy": same / len(references)}
