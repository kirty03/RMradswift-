"""demo_app.py - the RadSwift demo (research prototype, not for clinical use).

Pages:
  1. Worklist              how AI priority changes the order of waiting studies
  2. Report workspace      X-ray, abnormal score, AI draft that the radiologist edits and signs
  3. Turnaround simulation live run of the workflow simulator
  4. About                 what is implemented, simulated, assumed and future work

Run:  streamlit run app/demo_app.py

Works in two modes, shown in the sidebar:
  Live model         models/ exists, so uploaded X-rays are read by the fine-tuned model
  Saved predictions  only demo_bundle/ exists; drafts were produced earlier by Step 5
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import workflow_sim as ws  # noqa: E402

BUNDLE = ROOT / "demo_bundle"
MODEL_DIR = ROOT / "models"
RESULTS = ROOT / "results"
STAGES = ["acquisition_h", "queue_h", "interpretation_h", "finalization_h", "distribution_h"]

st.set_page_config(page_title="RadSwift demo", layout="wide")


@st.cache_data
def load_cases():
    path = BUNDLE / "demo_cases.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_metrics():
    path = RESULTS / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_resource
def load_live_model():
    """The fine-tuned model, or None when the weights are not on this computer."""
    if not (MODEL_DIR / "radswift_model.pt").exists():
        return None
    try:
        from report_model import load_model
        return load_model(MODEL_DIR, "cpu")
    except Exception as error:      # missing library or damaged file: fall back to saved predictions
        st.sidebar.warning(f"Model could not be loaded: {error}")
        return None


def read_xray(image):
    """Run the live model on one image. Returns (draft text, abnormal score)."""
    from xray_dataset import EVAL_TRANSFORM
    model, tokenizer = load_live_model()
    pixel_values = EVAL_TRANSFORM(image.convert("RGB")).unsqueeze(0)
    token_ids, score = model.generate(pixel_values)
    return tokenizer.decode(token_ids[0], skip_special_tokens=True), float(score[0])


def threshold():
    metrics = load_metrics()
    return metrics["triage"]["threshold"] if metrics else 0.5


# ----------------------------------------------------------------- pages
def page_worklist(cases):
    st.header("Worklist")
    st.caption("X-rays and AI scores are real model output on public test studies. "
               "Arrival times and priority are SIMULATED for this demo.")
    rng = np.random.default_rng(7)
    work = cases.head(40).copy()
    work["waiting_min"] = np.sort(rng.integers(5, 600, len(work)))[::-1]       # longest wait first
    work["priority"] = np.where(rng.random(len(work)) < 0.2, "Urgent", "Routine")
    work["ai_flag"] = work["abnormal_score"] >= threshold()

    use_ai = st.toggle("RadSwift priority (AI-flagged studies move up)")
    keys = ["priority", "ai_flag", "waiting_min"] if use_ai else ["priority", "waiting_min"]
    ascending = [False, False, False] if use_ai else [False, False]            # Urgent > Routine, flagged first, longest wait first
    work = work.sort_values(keys, ascending=ascending).reset_index(drop=True)
    work.insert(0, "position", work.index + 1)

    truly_abnormal = work[work["label_abnormal"] == 1]
    st.metric("Average position of truly abnormal studies", f"{truly_abnormal['position'].mean():.1f}",
              help="Lower is better: abnormal studies are read sooner.")
    table = work[["position", "uid", "priority", "waiting_min", "abnormal_score", "ai_flag"]].copy()
    table["truth (hidden from radiologist)"] = np.where(work["label_abnormal"] == 1, "abnormal", "normal")
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption("Order rule: urgent first; then (with RadSwift) AI-flagged first; then longest wait first. "
               "This is the same rule the simulator uses.")


def page_workspace(cases):
    st.header("Report workspace")
    source = st.radio("Study", ["Demo case", "Upload a chest X-ray"], horizontal=True)
    reference = None
    if source == "Demo case":
        uid = st.selectbox("Study ID", cases["uid"])
        case = cases[cases["uid"] == uid].iloc[0]
        from PIL import Image
        image = Image.open(BUNDLE / "images" / case["image_file"])
        draft, score, reference = case["ai_draft"], case["abnormal_score"], case["report_text"]
        if load_live_model() and st.button("Re-run the live model on this image"):
            draft, score = read_xray(image)
    else:
        if load_live_model() is None:
            st.info("Upload needs the fine-tuned weights in models/. This computer is in saved-predictions mode.")
            return
        upload = st.file_uploader("Frontal chest X-ray (PNG or JPEG)", type=["png", "jpg", "jpeg"])
        if upload is None:
            return
        from PIL import Image
        image = Image.open(upload)
        uid = upload.name
        with st.spinner("Reading the image..."):
            draft, score = read_xray(image)

    left, right = st.columns(2)
    left.image(image, use_container_width=True)
    with right:
        flagged = score >= threshold()
        st.metric("Abnormal score", f"{score:.2f}", "flagged for early reading" if flagged else "not flagged",
                  delta_color="inverse" if flagged else "off")
        st.caption(f"Flag threshold {threshold():.2f}, chosen on the validation set. "
                   "The score is for ordering the worklist. It is not a diagnosis.")
        edited = st.text_area("AI draft (edit before signing)", draft, height=260, key=f"draft_{uid}")
        accept, reject = st.columns(2)
        decisions = st.session_state.setdefault("decisions", {})
        if accept.button("Accept and sign", type="primary"):
            decisions[str(uid)] = "accepted" if edited == draft else "accepted with edits"
        if reject.button("Reject draft"):
            decisions[str(uid)] = "rejected"
        if str(uid) in decisions:
            st.success(f"Radiologist decision recorded: {decisions[str(uid)]}")
        if reference is not None:
            with st.expander("Radiologist's original report (for evaluation only)"):
                st.write(reference)
    if st.session_state.get("decisions"):
        st.caption(f"Decisions this session: {st.session_state['decisions']}")


def page_simulation():
    st.header("Turnaround simulation")
    st.caption("SIMULATED department. Every number comes from config/workflow_assumptions.yaml.")
    metrics = load_metrics()
    sens0 = metrics["triage"]["sensitivity"] if metrics else ws.BASELINE["triage_sensitivity"]
    spec0 = metrics["triage"]["specificity"] if metrics else ws.BASELINE["triage_specificity"]
    acc0 = metrics["fine_tuned"]["all"]["draft_acceptance_proxy"] if metrics else ws.BASELINE["draft_acceptance"]
    if not metrics:
        st.warning("results/metrics.json not found: accuracy sliders start from placeholder values.")

    a, b, c = st.columns(3)
    with a:
        triage = st.checkbox("AI triage", True)
        sens = st.slider("Triage sensitivity", 0.0, 1.0, float(sens0))
        spec = st.slider("Triage specificity", 0.0, 1.0, float(spec0))
    with b:
        draft = st.checkbox("AI draft reports", True)
        acceptance = st.slider("Share of drafts accepted", 0.0, 1.0, float(acc0))
        reduction = st.slider("Reading time saved on accepted drafts", 0.0, 0.5, 0.155)
    with c:
        evening = st.slider("Radiologists in the evening", 0, 5, 0)
        night = st.slider("Radiologists at night", 0, 5, 0)
        workload = st.slider("Workload (1.0 = baseline)", 0.8, 1.2, 1.0, 0.05)

    if st.button("Run simulation (3 runs of 30 days)", type="primary"):
        scenario = {"ai_triage": triage, "triage_sensitivity": sens, "triage_specificity": spec,
                    "ai_draft": draft, "draft_acceptance": acceptance, "read_time_reduction": reduction,
                    "evening_staff": evening, "night_staff": night}
        config = {"workload_scale": workload}
        with st.spinner("Simulating..."):
            base = pd.concat([ws.stage_means(ws.simulate(config, {}, s)) for s in (1, 2, 3)], axis=1).mean(axis=1)
            new = pd.concat([ws.stage_means(ws.simulate(config, scenario, s)) for s in (1, 2, 3)], axis=1).mean(axis=1)
        m1, m2, m3 = st.columns(3)
        m1.metric("Baseline turnaround", f"{base['tat_h']:.1f} h")
        m2.metric("With these settings", f"{new['tat_h']:.1f} h", f"{new['tat_h'] - base['tat_h']:+.2f} h", delta_color="inverse")
        m3.metric("6-hour target reached?", "Yes" if new["tat_h"] <= 6 else "No")
        chart = pd.DataFrame({"Baseline": base[STAGES], "Scenario": new[STAGES]})
        chart.index = [s.replace("_h", "") for s in STAGES]
        st.bar_chart(chart, stack=False)

    saved = RESULTS / "simulation_summary.csv"
    if saved.exists():
        st.subheader("Saved result: 30 runs per scenario (Step 6)")
        st.dataframe(pd.read_csv(saved), hide_index=True, use_container_width=True)


def page_about():
    st.header("About RadSwift")
    st.error("Research prototype for a course project. Not a medical device. Not for clinical use.")
    st.table(pd.DataFrame({
        "Status": ["Implemented", "Simulated", "Assumed", "Future work"],
        "What": [
            "Data preparation, fine-tuned report model, evaluation, abnormal score, this demo, workflow simulator",
            "Workflow timings, turnaround effects, arrival times and priority shown in the worklist",
            "26.6 h baseline and its stage split, staffing, reading times, workflow-gain percentages (see config/workflow_assumptions.yaml)",
            "Validation on hospital data, PACS/RIS integration, CT and MRI, lateral view, regulatory approval",
        ]}))
    st.markdown(
        "**X-ray data:** Indiana University chest X-ray collection (Open-i, U.S. National Library of Medicine; "
        "Demner-Fushman et al., 2016), de-identified and public.  \n"
        "**Workflow data:** generated by `src/step1_generate_workflow_data.py`. No real patients.  \n"
        "**Model:** ViT image encoder + T5-small text decoder, fine-tuned by `src/step4_train_report_model.py`.")
    metrics = load_metrics()
    if metrics:
        st.subheader("Measured on the test set (Step 5)")
        st.json({"test studies": metrics["n_test"], "triage": metrics["triage"],
                 "fine-tuned report scores": metrics["fine_tuned"]["all"],
                 "template baseline": metrics["template"]["all"]})


# ----------------------------------------------------------------- layout
cases = load_cases()
live = load_live_model() is not None
st.sidebar.title("RadSwift")
if live:
    st.sidebar.success("Live model")
else:
    st.sidebar.info("Saved predictions")
st.sidebar.caption("Research prototype. Not for clinical use.")
page = st.sidebar.radio("Page", ["Worklist", "Report workspace", "Turnaround simulation", "About"])

if page == "Turnaround simulation":
    page_simulation()
elif page == "About":
    page_about()
elif cases is None:
    st.warning("demo_bundle/demo_cases.csv not found. Run Step 5 on Kaggle and copy demo_bundle/ into this folder.")
elif page == "Worklist":
    page_worklist(cases)
else:
    page_workspace(cases)
