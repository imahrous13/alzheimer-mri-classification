"""Interactive Alzheimer MRI classifier - Streamlit application.

Run locally:

    streamlit run app.py

The app loads the trained HOG baseline and the compact CNN produced by the
pipeline, scores uploaded MRI images, and shows per-class probabilities,
confidence flags, and a Grad-CAM saliency map for the CNN. It is an educational
demo, not a medical device.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from alzheimer_mri.baseline import hog_histogram_features  # noqa: E402
from alzheimer_mri.dataset import CLASS_NAMES  # noqa: E402
from alzheimer_mri.explain import grad_cam  # noqa: E402
from alzheimer_mri.model import build_model  # noqa: E402
from alzheimer_mri.paths import MODELS_DIR, PROJECT_ROOT, RESULTS_DIR  # noqa: E402
from alzheimer_mri.predict import prediction_transform  # noqa: E402

CLASS_COLORS = {
    "MildDemented": "#4C72B0",
    "ModerateDemented": "#C44E52",
    "NonDemented": "#55A868",
    "VeryMildDemented": "#CCB974",
}
DISCLAIMER = (
    "Educational research prototype. This is NOT a medical diagnostic system and must not be "
    "used for clinical decisions. Directory labels are treated as supplied ground truth."
)
FIGURES = ROOT / "docs" / "figures"


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_baseline() -> dict | None:
    path = MODELS_DIR / "hog_sgd_classifier.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def load_cnn() -> tuple | None:
    path = MODELS_DIR / "best_model.pt"
    if not path.exists():
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(len(checkpoint["class_names"]), checkpoint.get("dropout", 0.35)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint, device


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def baseline_probabilities(bundle: dict, image: Image.Image) -> np.ndarray:
    features = hog_histogram_features(image, bundle["image_size"])
    return bundle["model"].predict_proba([features])[0]


def cnn_probabilities(model, checkpoint, device, image: Image.Image):
    transform = prediction_transform(checkpoint["image_size"])
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = F.softmax(model(tensor), dim=1)[0].cpu().numpy()
    return probabilities, tensor


def gradcam_figure(model, tensor, class_index: int, display: Image.Image, size: int) -> plt.Figure:
    heatmap = grad_cam(model, tensor, class_index)
    figure, axis = plt.subplots(figsize=(4.6, 4.6))
    axis.imshow(display.convert("L").resize((size, size)), cmap="gray")
    axis.imshow(heatmap, cmap="magma", alpha=0.45, vmin=0, vmax=1)
    axis.set_title(f"Grad-CAM: {CLASS_NAMES[class_index]}")
    axis.axis("off")
    figure.tight_layout()
    return figure


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def probability_chart(probabilities: np.ndarray) -> None:
    frame = pd.DataFrame(
        {"probability": probabilities}, index=[str(name) for name in CLASS_NAMES]
    )
    st.bar_chart(frame, height=260, color="#38bdf8")


def render_prediction(image: Image.Image, probabilities: np.ndarray, threshold: float) -> None:
    best = int(np.argmax(probabilities))
    confidence = float(probabilities[best])
    left, right = st.columns([1, 1.3])
    with left:
        st.image(image, caption="Input MRI", width="stretch")
    with right:
        st.markdown(f"### Predicted class: `{CLASS_NAMES[best]}`")
        st.metric("Confidence", f"{confidence * 100:.1f}%")
        if confidence < threshold:
            st.warning("Low confidence - flagged for human review.")
        else:
            st.success("Confidence above review threshold.")
        probability_chart(probabilities)
    with st.expander("All class probabilities"):
        st.dataframe(
            pd.DataFrame(
                {"class": CLASS_NAMES, "probability": np.round(probabilities, 4)}
            ),
            hide_index=True,
            width="stretch",
        )


def sample_image() -> Image.Image | None:
    directory = PROJECT_ROOT / "data" / "raw" / "original"
    if not directory.exists():
        return None
    candidates = sorted(directory.rglob("*.jpg"))
    if not candidates:
        return None
    rng = np.random.default_rng()
    return Image.open(candidates[int(rng.integers(0, len(candidates)))])


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
def tab_predict() -> None:
    st.subheader("Score an MRI slice")
    baseline = load_baseline()
    cnn = load_cnn()

    if baseline is None and cnn is None:
        st.error(
            "No trained models found. Run the pipeline first:\n\n"
            "`python -m alzheimer_mri.baseline` and `python -m alzheimer_mri.train`."
        )
        return

    options = []
    if baseline is not None:
        options.append("HOG baseline (linear)")
    if cnn is not None:
        options.append("Compact CNN")

    controls = st.columns(3)
    model_name = controls[0].radio("Model", options, horizontal=False)
    threshold = controls[1].slider("Low-confidence review threshold", 0.3, 0.95, 0.60, 0.05)
    show_gradcam = controls[2].checkbox("Show Grad-CAM (CNN)", value=True)

    uploaded = st.file_uploader(
        "Upload MRI image(s)", type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=True
    )

    use_sample = st.button("Use a random held-out style example")
    images: list[tuple[str, Image.Image]] = []
    if uploaded:
        images = [(item.name, Image.open(item).convert("RGB")) for item in uploaded]
    elif use_sample:
        example = sample_image()
        if example is None:
            st.info("No local dataset found to sample from; please upload an image.")
        else:
            images = [("random dataset example", example.convert("RGB"))]

    if not images:
        st.caption("Upload an image or use a random example to see a prediction.")
        return

    for name, image in images:
        st.markdown(f"#### {name}")
        if model_name == "HOG baseline (linear)" and baseline is not None:
            probabilities = baseline_probabilities(baseline, image)
            render_prediction(image, probabilities, threshold)
        elif cnn is not None:
            model, checkpoint, device = cnn
            probabilities, tensor = cnn_probabilities(model, checkpoint, device, image)
            render_prediction(image, probabilities, threshold)
            if show_gradcam:
                best = int(np.argmax(probabilities))
                figure = gradcam_figure(model, tensor, best, image, checkpoint["image_size"])
                st.pyplot(figure, width="content")
                plt.close(figure)
        st.divider()


def tab_performance() -> None:
    st.subheader("Held-out performance")
    baseline = read_json(RESULTS_DIR / "baseline" / "metrics.json")
    baseline_original = read_json(RESULTS_DIR / "baseline_original" / "metrics.json")
    cnn = read_json(RESULTS_DIR / "evaluation" / "metrics.json")

    rows = []
    if baseline:
        m = baseline["test_metrics"]
        rows.append({"model": "HOG baseline (all data)", **{k: m[k] for k in ("accuracy", "balanced_accuracy", "macro_f1")}})
    if baseline_original:
        m = baseline_original["test_metrics"]
        rows.append({"model": "HOG baseline (original only)", **{k: m[k] for k in ("accuracy", "balanced_accuracy", "macro_f1")}})
    if cnn:
        rows.append({"model": "Compact CNN (original only)", **{k: cnn[k] for k in ("accuracy", "balanced_accuracy", "macro_f1")}})

    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("model").style.format("{:.3f}"), width="stretch")
    else:
        st.info("No metrics found. Run the evaluation steps of the pipeline.")

    left, right = st.columns(2)
    with left:
        for figure in ["model_comparison.png", "calibration.png"]:
            if (FIGURES / figure).exists():
                st.image(str(FIGURES / figure))
    with right:
        for figure in ["roc_curves.png", "error_gallery.png"]:
            if (FIGURES / figure).exists():
                st.image(str(FIGURES / figure))


def tab_data_quality() -> None:
    st.subheader("Data-quality audit")
    audit = read_json(RESULTS_DIR / "data_quality" / "audit_summary.json")
    if audit:
        columns = st.columns(4)
        columns[0].metric("Images audited", f"{audit['total_files']:,}")
        columns[1].metric("Decoded", f"{audit['readable_files']:,}")
        columns[2].metric("Cross-source pHash groups", f"{audit['cross_dataset_perceptual_hash_groups']:,}")
        columns[3].metric("Exact-duplicate rows", f"{audit['exact_duplicate_rows']:,}")
    st.markdown("**Leakage example** - one scan and its transforms share a perceptual hash:")
    for figure in ["leakage_example.png", "source_statistics.png", "class_montage.png"]:
        if (FIGURES / figure).exists():
            st.image(str(FIGURES / figure))


def tab_about() -> None:
    st.subheader("About this demo")
    st.markdown(
        """
        This application scores 2D brain-MRI slices into four dementia categories using models
        trained by the repository's reproducible, leakage-controlled pipeline.

        - **HOG baseline** — histogram-of-oriented-gradients + intensity histogram, linear classifier.
        - **Compact CNN** — a small PyTorch convolutional network with Grad-CAM explanations.

        Validation and test sets contain **original images only**; augmented copies of held-out
        scans are quarantined so they cannot leak into training. See the README and
        `reports/PROJECT_REPORT.md` for methodology and results.
        """
    )
    st.error(DISCLAIMER)
    st.markdown("#### Known limitations")
    st.markdown(
        """
        - No patient identifiers, so perceptual grouping is a proxy for patient-level splitting.
        - `ModerateDemented` has only 64 original images; its metrics are preliminary.
        - No external cohort, scanner metadata, or clinical review.
        - Models may exploit source cues (augmented images are smoother) rather than pathology.
        """
    )


def main() -> None:
    st.set_page_config(page_title="Alzheimer MRI Classifier", layout="wide")
    st.title("Alzheimer MRI Classification - interactive demo")
    st.caption(DISCLAIMER)

    tabs = st.tabs(["Predict", "Model performance", "Data quality", "About"])
    with tabs[0]:
        tab_predict()
    with tabs[1]:
        tab_performance()
    with tabs[2]:
        tab_data_quality()
    with tabs[3]:
        tab_about()

    st.sidebar.markdown("### Alzheimer MRI Classifier")
    st.sidebar.markdown(
        "Leakage-aware MRI classification demo. Upload a slice or use a random example, "
        "choose a model, and inspect the prediction and Grad-CAM."
    )
    st.sidebar.info("Educational prototype - not for clinical use.")


if __name__ == "__main__":
    main()
