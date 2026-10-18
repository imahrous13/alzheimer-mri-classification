"""Generate the portfolio figure gallery from the committed, audited outputs.

Every figure is derived from reviewed artifacts (audit summary, splits, cached
features, and held-out predictions); nothing is hand-authored or fabricated.
Outputs are written to ``docs/figures/`` and referenced by the README, report,
and notebook.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image
from sklearn.decomposition import PCA

from alzheimer_mri.dataset import CLASS_NAMES
from alzheimer_mri.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR, RESULTS_DIR


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "figures"

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

CLASS_COLORS = {
    "MildDemented": "#4C72B0",
    "ModerateDemented": "#C44E52",
    "NonDemented": "#55A868",
    "VeryMildDemented": "#CCB974",
}
SOURCE_COLORS = {"original": "#1F4E79", "augmented": "#E08A1E"}

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "font.size": 10,
    }
)


def load_inputs() -> dict:
    return {
        "audit": json.loads((RESULTS_DIR / "data_quality" / "audit_summary.json").read_text()),
        "splits": pd.read_csv(PROCESSED_DATA_DIR / "dataset_splits.csv"),
        "manifest": pd.read_csv(PROCESSED_DATA_DIR / "image_manifest.csv"),
        "baseline": json.loads((RESULTS_DIR / "baseline" / "metrics.json").read_text()),
        "baseline_original": json.loads(
            (RESULTS_DIR / "baseline_original" / "metrics.json").read_text()
        ),
        "cnn": json.loads((RESULTS_DIR / "evaluation" / "metrics.json").read_text()),
        "baseline_predictions": pd.read_csv(RESULTS_DIR / "baseline" / "test_predictions.csv"),
    }


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name)
    plt.close(fig)
    print("wrote", FIGURES / name)


# --------------------------------------------------------------------------- #
# 1. Pipeline diagram
# --------------------------------------------------------------------------- #
def figure_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3)
    ax.axis("off")

    stages = [
        ("Audit\n40,384 images", "#1F4E79"),
        ("Leakage-aware\nsplit", "#2E7D8A"),
        ("Modeling\nHOG + CNN", "#3E8E41"),
        ("Held-out\nevaluation", "#B7791F"),
        ("Dashboard\n& report", "#8E44AD"),
    ]
    width, gap = 2.05, 0.55
    for index, (label, color) in enumerate(stages):
        x = 0.35 + index * (width + gap)
        ax.add_patch(
            FancyBboxPatch(
                (x, 1.0), width, 1.0,
                boxstyle="round,pad=0.08,rounding_size=0.12",
                linewidth=0, facecolor=color,
            )
        )
        ax.text(x + width / 2, 1.5, label, ha="center", va="center",
                color="white", fontsize=11, fontweight="bold")
        if index < len(stages) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + width + 0.05, 1.5), (x + width + gap - 0.05, 1.5),
                    arrowstyle="-|>", mutation_scale=18, linewidth=2, color="#94a3b8",
                )
            )

    details = [
        "integrity, intensity,\nentropy, pHash",
        "perceptual groups,\noriginal-only holdouts",
        "class-weighted,\nvalidated selection",
        "ROC, calibration,\nbootstrap CIs",
        "notebook + HTML\ndashboard",
    ]
    for index, text in enumerate(details):
        x = 0.35 + index * (width + gap) + width / 2
        ax.text(x, 0.55, text, ha="center", va="center", color="#475569", fontsize=8.5)

    ax.text(6.5, 2.55, "Reproducible, leakage-controlled Alzheimer MRI pipeline",
            ha="center", va="center", fontsize=14, fontweight="bold", color="#0f172a")
    save(fig, "pipeline.png")


# --------------------------------------------------------------------------- #
# 2. Class montage
# --------------------------------------------------------------------------- #
def figure_montage(inputs: dict) -> None:
    manifest = inputs["manifest"]
    fig, axes = plt.subplots(2, len(CLASS_NAMES), figsize=(13, 6.6))
    for column, label in enumerate(CLASS_NAMES):
        for row, dataset in enumerate(["original", "augmented"]):
            subset = manifest[(manifest["label"] == label) & (manifest["dataset"] == dataset)]
            path = RAW_DATA_DIR / subset.iloc[0]["path"]
            with Image.open(path) as image:
                axes[row, column].imshow(image.convert("L"), cmap="gray")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if row == 0:
                axes[row, column].set_title(label, color=CLASS_COLORS[label])
            if column == 0:
                axes[row, column].set_ylabel(dataset.capitalize(), fontsize=13, fontweight="bold")
    fig.suptitle("Representative MRI per class and source collection", fontsize=15, fontweight="bold")
    save(fig, "class_montage.png")


# --------------------------------------------------------------------------- #
# 3. Cohort balance + split composition
# --------------------------------------------------------------------------- #
def figure_distributions(inputs: dict) -> None:
    audit = inputs["audit"]
    splits = inputs["splits"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5))

    counts = pd.DataFrame(audit["class_counts"]).pivot(
        index="label", columns="dataset", values="count"
    ).reindex(CLASS_NAMES)
    counts.plot(kind="bar", ax=left, color=[SOURCE_COLORS["augmented"], SOURCE_COLORS["original"]])
    left.set_title("Class counts by source collection")
    left.set_xlabel("")
    left.set_ylabel("Images")
    left.set_ylim(0, 11800)
    left.tick_params(axis="x", rotation=20)
    for container in left.containers:
        left.bar_label(container, fontsize=8, padding=2)
    left.legend(title="", loc="upper center", ncol=2)

    order = ["train", "validation", "test", "excluded_leakage_guard"]
    split_pivot = (
        splits.groupby(["split", "dataset"]).size().unstack(fill_value=0).reindex(order)
    ).rename(index={"excluded_leakage_guard": "leakage guard"})
    split_pivot.plot(
        kind="barh", stacked=True, ax=right,
        color=[SOURCE_COLORS["augmented"], SOURCE_COLORS["original"]],
    )
    right.set_title("Split composition (holdouts are original-only)")
    right.set_xlabel("Images")
    right.set_ylabel("")
    right.legend(title="", loc="upper right")
    for bar in right.patches:
        width = bar.get_width()
        if width > 1500:
            right.text(width / 2, bar.get_y() + bar.get_height() / 2, f"{int(width):,}",
                       ha="center", va="center", color="white", fontsize=8)
    save(fig, "distributions.png")


# --------------------------------------------------------------------------- #
# 4. Leakage example (same scan, transformed)
# --------------------------------------------------------------------------- #
def figure_leakage(inputs: dict) -> None:
    manifest = inputs["manifest"]
    grouped = manifest.groupby(["label", "phash"])
    chosen = None
    for (label, phash), group in grouped:
        datasets = set(group["dataset"])
        if datasets == {"original", "augmented"}:
            augmented = group[group["dataset"] == "augmented"]
            if len(augmented) >= 3:
                chosen = (label, phash, group)
                break
    if chosen is None:
        return
    label, phash, group = chosen
    original = group[group["dataset"] == "original"].iloc[0]
    augmented = group[group["dataset"] == "augmented"].iloc[:3]

    paths = [("Original", RAW_DATA_DIR / original["path"])]
    paths += [(f"Augmented #{i + 1}", RAW_DATA_DIR / row["path"]) for i, (_, row) in enumerate(augmented.iterrows())]

    fig, axes = plt.subplots(1, len(paths), figsize=(4 * len(paths), 4.4))
    for axis, (title, path) in zip(axes, paths):
        with Image.open(path) as image:
            axis.imshow(image.convert("L"), cmap="gray")
        axis.set_title(title, color="#1F4E79" if title == "Original" else "#E08A1E")
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(
        f"Leakage example: one {label} scan and its transforms share pHash {phash}",
        fontsize=13, fontweight="bold",
    )
    fig.text(
        0.5, -0.02,
        "A random split would place these in different sets, letting the model memorize transformations.",
        ha="center", fontsize=9.5, color="#475569",
    )
    save(fig, "leakage_example.png")


# --------------------------------------------------------------------------- #
# 5. Source statistics
# --------------------------------------------------------------------------- #
def figure_source_statistics(inputs: dict) -> None:
    stats = pd.DataFrame(inputs["audit"]["intensity_summary"])
    metrics = [
        ("mean_intensity_mean", "Mean intensity"),
        ("std_intensity_mean", "Contrast (std dev)"),
        ("entropy_mean", "Entropy (bits)"),
        ("sharpness_mean", "Sharpness (Laplacian var.)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for axis, (key, title) in zip(axes, metrics):
        sns.barplot(
            data=stats, x="dataset", y=key, hue="dataset",
            palette=SOURCE_COLORS, legend=False, ax=axis,
        )
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("")
        axis.set_ylabel("")
    fig.suptitle("Image statistics by source collection (augmented images are uniformly smoother)",
                 fontsize=13, fontweight="bold")
    save(fig, "source_statistics.png")


# --------------------------------------------------------------------------- #
# 6. Model comparison with confidence intervals
# --------------------------------------------------------------------------- #
def figure_model_comparison(inputs: dict) -> None:
    metrics = ["accuracy", "balanced_accuracy", "macro_f1"]
    labels = ["Accuracy", "Balanced accuracy", "Macro-F1"]
    models = [
        ("HOG baseline (all data)", inputs["baseline"]["test_metrics"], "#1F4E79"),
        ("HOG baseline (original only)", inputs["baseline_original"]["test_metrics"], "#3E8E41"),
        ("Compact CNN (original only)", inputs["cnn"], "#8E44AD"),
    ]
    x = np.arange(len(metrics))
    width = 0.26

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    for index, (name, data, color) in enumerate(models):
        values = [data[m] for m in metrics]
        positions = x + (index - 1) * width
        ax.bar(positions, values, width, label=name, color=color)

    intervals = inputs["baseline"].get("bootstrap_intervals", {})
    headline = inputs["baseline"]["test_metrics"]
    tops = {}
    for metric_index, metric_key in enumerate(
        ["accuracy_95pct", "balanced_accuracy_95pct", "macro_f1_95pct"]
    ):
        center = headline[metric_key.replace("_95pct", "")]
        low, high = intervals.get(metric_key, (center, center))
        xi = x[metric_index] - width
        ax.errorbar(xi, center, yerr=[[center - low], [high - center]],
                    fmt="none", ecolor="#0f172a", capsize=5, linewidth=1.6, zorder=5)
        tops[metric_index] = high

    for index, (_, data, _) in enumerate(models):
        for metric_index, metric in enumerate(metrics):
            xi = x[metric_index] + (index - 1) * width
            top = tops.get(metric_index, data[metric]) if index == 0 else data[metric]
            ax.text(xi, top + 0.015, f"{data[metric]:.3f}", ha="center", fontsize=8.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.92)
    ax.set_ylabel("Score")
    ax.set_title("Held-out performance on the original-only test set (n = 963)")
    ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
    ax.text(0.5, 0.965, "Error bars show 95% bootstrap intervals for the HOG baseline",
            transform=ax.transAxes, ha="center", fontsize=9, color="#64748b")
    save(fig, "model_comparison.png")


# --------------------------------------------------------------------------- #
# 7. ROC curves
# --------------------------------------------------------------------------- #
def _roc_points(labels_true, scores, positive):
    order = np.argsort(-scores)
    y = (labels_true[order] == positive).astype(float)
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    tpr = tps / max(tps[-1], 1)
    fpr = fps / max(fps[-1], 1)
    return np.concatenate([[0], fpr]), np.concatenate([[0], tpr])


def figure_roc(inputs: dict) -> None:
    predictions = inputs["baseline_predictions"]
    labels_true = predictions["label"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    aucs = []
    for name in CLASS_NAMES:
        scores = predictions[f"probability_{name}"].to_numpy()
        fpr, tpr = _roc_points(labels_true, scores, name)
        auc_value = _trapz(tpr, fpr)
        aucs.append(auc_value)
        ax.plot(fpr, tpr, color=CLASS_COLORS[name], linewidth=2.2,
                label=f"{name} (AUC {auc_value:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="0.5", linewidth=1.2, label="Chance")
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate",
           xlim=(0, 1), ylim=(0, 1),
           title=f"One-vs-rest ROC, HOG baseline\nmacro AUC = {np.mean(aucs):.3f}")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    save(fig, "roc_curves.png")


# --------------------------------------------------------------------------- #
# 8. Calibration / reliability
# --------------------------------------------------------------------------- #
def figure_calibration(inputs: dict) -> None:
    predictions = inputs["baseline_predictions"]
    labels_true = predictions["label"].to_numpy()
    confidence = predictions["confidence"].to_numpy()
    correct = predictions["prediction"].to_numpy() == labels_true
    edges = np.linspace(0, 1, 11)
    centers, accuracies, counts = [], [], []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            centers.append(confidence[mask].mean())
            accuracies.append(correct[mask].mean())
            counts.append(int(mask.sum()))
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    ax.plot([0, 1], [0, 1], "--", color="0.5", label="Perfect calibration")
    ax.plot(centers, accuracies, marker="o", color="#1F4E79", linewidth=2, label="HOG baseline")
    for x_value, y_value, count in zip(centers, accuracies, counts):
        ax.annotate(str(count), (x_value, y_value), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8)
    ece = inputs["baseline"]["test_metrics"].get("expected_calibration_error")
    ax.set(xlabel="Mean predicted confidence", ylabel="Observed accuracy",
           xlim=(0, 1), ylim=(0, 1),
           title=f"Reliability curve (ECE = {ece:.3f})")
    ax.legend(loc="upper left", frameon=False)
    save(fig, "calibration.png")


# --------------------------------------------------------------------------- #
# 9. Error gallery
# --------------------------------------------------------------------------- #
def figure_errors(inputs: dict) -> None:
    predictions = inputs["baseline_predictions"]
    wrong = predictions[predictions["prediction"] != predictions["label"]]
    wrong = wrong.sort_values("confidence", ascending=False).head(8)
    if wrong.empty:
        return
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for axis, (_, row) in zip(axes.flat, wrong.iterrows()):
        with Image.open(RAW_DATA_DIR / row["path"]) as image:
            axis.imshow(image.convert("L"), cmap="gray")
        axis.set_title(f"true: {row['label']}\npred: {row['prediction']} ({row['confidence']:.2f})",
                       fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("Most-confident errors of the HOG baseline — the familiar dementia spectrum is easy to confuse",
                 fontsize=12.5, fontweight="bold")
    save(fig, "error_gallery.png")


# --------------------------------------------------------------------------- #
# 10. HOG feature embedding
# --------------------------------------------------------------------------- #
def figure_embedding() -> None:
    cache = sorted((PROCESSED_DATA_DIR / "features").glob("test_64_*.npz"))
    if not cache:
        return
    data = np.load(cache[0])
    features, labels = data["features"], data["labels"]
    pca = PCA(n_components=2, random_state=42)
    embedding = pca.fit_transform(features)
    explained = pca.explained_variance_ratio_ * 100
    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    for index, name in enumerate(CLASS_NAMES):
        mask = labels == index
        rare = name == "ModerateDemented"
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   s=42 if rare else 18, alpha=0.9 if rare else 0.6,
                   color=CLASS_COLORS[name], label=f"{name} (n={int(mask.sum())})",
                   edgecolors="#111827" if rare else "none", linewidths=0.7 if rare else 0)
    ax.set(
        title="PCA of HOG features on the held-out test set\nthe four classes overlap heavily, which caps achievable accuracy",
        xlabel=f"Principal component 1 ({explained[0]:.1f}% var.)",
        ylabel=f"Principal component 2 ({explained[1]:.1f}% var.)",
    )
    ax.legend(frameon=False, fontsize=9)
    save(fig, "hog_embedding.png")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs()
    figure_pipeline()
    figure_montage(inputs)
    figure_distributions(inputs)
    figure_leakage(inputs)
    figure_source_statistics(inputs)
    figure_model_comparison(inputs)
    figure_roc(inputs)
    figure_calibration(inputs)
    figure_errors(inputs)
    figure_embedding()


if __name__ == "__main__":
    main()
