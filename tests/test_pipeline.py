"""Checks that the pipeline behaves as we claim.

Run:  python tests/test_pipeline.py      (or: pytest tests/)
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import workflow_sim as ws                                                  # noqa: E402
from text_metrics import clean_report_text, clinical_scores, positive_findings   # noqa: E402

SHORT = {"n_days": 10}      # shorter runs keep the tests fast


def test_timestamps_are_in_order():
    df = ws.simulate(SHORT, seed=3)
    cols = ["scan_completed_min", "images_available_min", "read_started_min",
            "prelim_report_min", "report_signed_min", "report_sent_min"]
    assert (np.diff(df[cols].to_numpy(), axis=1) >= 0).all()


def test_same_seed_gives_same_result():
    assert ws.simulate(SHORT, seed=5)["tat_h"].equals(ws.simulate(SHORT, seed=5)["tat_h"])


def test_baseline_is_close_to_26_6_hours():
    tat = np.mean([ws.stage_means(ws.simulate(seed=s))["tat_h"] for s in (11, 12, 13)])
    assert abs(tat - 26.6) < 0.5, tat


def test_same_studies_in_every_scenario():
    base = ws.simulate(SHORT, seed=4)
    triage = ws.simulate(SHORT, {"ai_triage": True}, seed=4)
    assert base["scan_completed_min"].equals(triage["scan_completed_min"])
    assert base["truly_abnormal"].equals(triage["truly_abnormal"])


def test_flagged_studies_wait_less_under_triage():
    df = ws.simulate(scenario={"ai_triage": True}, seed=6)
    chest = df[df["is_chest_xray"] & ~df["in_warmup"] & (df["priority"] == "Routine")]
    assert chest[chest["ai_flag"]]["queue_h"].mean() < chest[~chest["ai_flag"]]["queue_h"].mean()


def test_perfect_drafts_never_slow_reading():
    base = ws.stage_means(ws.simulate(SHORT, seed=8))
    draft = ws.stage_means(ws.simulate(SHORT, {"ai_draft": True, "draft_acceptance": 1.0}, seed=8))
    assert draft["tat_h"] <= base["tat_h"]


def test_text_cleaning():
    assert clean_report_text("The XXXX is normal . Chest X-XXXX clear.") == "the is normal. chest x-ray clear."
    assert clean_report_text(float("nan")) == ""


def test_finding_labeler_handles_negation():
    assert positive_findings("no pleural effusion or pneumothorax. the lungs are clear.") == set()
    assert positive_findings("there is a small left pleural effusion. no pneumothorax.") == {"pleural_effusion"}
    assert positive_findings("mild cardiomegaly. bibasilar atelectasis.") == {"cardiomegaly", "atelectasis"}


def test_clinical_scores():
    refs = ["small pleural effusion.", "lungs are clear."]
    assert clinical_scores(refs, refs)["clinical_f1"] == 1.0
    assert clinical_scores(refs, ["lungs are clear.", "lungs are clear."])["draft_acceptance_proxy"] == 0.5


def test_model_output_depends_on_the_image():
    """A tiny random-weights copy of the model: a different image must give a different loss."""
    try:
        import torch
        from transformers import T5Config, ViTConfig
        from report_model import XrayReportModel
    except ImportError:
        print("  (skipped: torch/transformers not installed)")
        return
    torch.manual_seed(0)
    vit = ViTConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64)
    t5 = T5Config(vocab_size=100, d_model=16, d_kv=8, d_ff=32, num_layers=1, num_heads=2, decoder_start_token_id=0)
    model = XrayReportModel(vit, t5).eval()
    labels = torch.randint(1, 100, (2, 6))
    loss_a, logit_a = model(torch.randn(2, 3, 224, 224), labels)
    loss_b, logit_b = model(torch.randn(2, 3, 224, 224), labels)
    assert loss_a.item() != loss_b.item() and not torch.equal(logit_a, logit_b)
    token_ids, score = model.generate(torch.randn(2, 3, 224, 224), max_new_tokens=5, num_beams=1)
    assert token_ids.shape[0] == 2 and ((score >= 0) & (score <= 1)).all()


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("passed:", test.__name__)
    print(f"{len(tests)} checks finished")
