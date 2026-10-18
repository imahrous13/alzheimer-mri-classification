from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import auc, balanced_accuracy_score, roc_curve
from sklearn.preprocessing import label_binarize
from torch.nn import functional as F
from tqdm import tqdm

from .dataset import CLASS_NAMES, load_split_rows, make_loader
from .metrics import bootstrap_interval, classification_metrics
from .model import build_model
from .paths import MODELS_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, ensure_output_dirs


def collect_predictions(model, loader, device):
    labels_all, probabilities_all, paths_all = [], [], []
    model.eval()
    with torch.inference_mode():
        for images, labels, paths in tqdm(loader, desc="Evaluating", unit="batch"):
            logits = model(images.to(device))
            probabilities = F.softmax(logits, dim=1).cpu().numpy()
            labels_all.extend(labels.numpy().tolist())
            probabilities_all.append(probabilities)
            paths_all.extend(paths)
    return (
        np.asarray(labels_all),
        np.concatenate(probabilities_all, axis=0),
        paths_all,
    )


def save_confusion_matrix(matrix: np.ndarray, output: Path) -> None:
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_totals, where=row_totals != 0)
    annotations = np.empty_like(matrix, dtype=object)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            annotations[row, column] = f"{matrix[row, column]}\n{normalized[row, column]:.1%}"

    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(
        normalized,
        annot=annotations,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={"label": "Recall within actual class"},
        ax=ax,
    )
    ax.set(title="Test confusion matrix", xlabel="Predicted class", ylabel="Actual class")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_roc_curves(labels: np.ndarray, probabilities: np.ndarray, output: Path) -> None:
    binary = label_binarize(labels, classes=np.arange(len(CLASS_NAMES)))
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    palette = sns.color_palette("colorblind", len(CLASS_NAMES))
    for index, class_name in enumerate(CLASS_NAMES):
        false_positive, true_positive, _ = roc_curve(binary[:, index], probabilities[:, index])
        score = auc(false_positive, true_positive)
        ax.plot(false_positive, true_positive, label=f"{class_name} (AUC {score:.3f})", color=palette[index])
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.45", label="Chance")
    ax.set(
        title="One-vs-rest ROC curves on the original-image test set",
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_reliability(labels: np.ndarray, probabilities: np.ndarray, output: Path) -> None:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    edges = np.linspace(0, 1, 11)
    centers, accuracies, counts = [], [], []
    for lower, upper in zip(edges[:-1], edges[1:], strict=False):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            centers.append(confidence[mask].mean())
            accuracies.append(correct[mask].mean())
            counts.append(mask.sum())
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.45", label="Perfect calibration")
    ax.plot(centers, accuracies, marker="o", color="#2457A7", label="Model")
    for x_value, y_value, count in zip(centers, accuracies, counts, strict=False):
        ax.annotate(str(count), (x_value, y_value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
    ax.set(
        title="Reliability curve on the original-image test set",
        xlabel="Mean predicted confidence",
        ylabel="Observed accuracy",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Alzheimer MRI classifier.")
    parser.add_argument("--checkpoint", type=Path, default=MODELS_DIR / "best_model.pt")
    parser.add_argument(
        "--splits", type=Path, default=PROCESSED_DATA_DIR / "dataset_splits.csv"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "evaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(
        num_classes=len(checkpoint["class_names"]), dropout=checkpoint.get("dropout", 0.35)
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    test_rows = load_split_rows(args.splits, "test")
    loader = make_loader(
        test_rows,
        image_size=checkpoint["image_size"],
        batch_size=args.batch_size,
        training=False,
        workers=args.workers,
    )
    labels, probabilities, paths = collect_predictions(model, loader, device)
    predictions = probabilities.argmax(axis=1)
    metrics = classification_metrics(labels, probabilities, list(CLASS_NAMES))
    lower, upper = bootstrap_interval(
        labels,
        predictions,
        balanced_accuracy_score,
        iterations=args.bootstrap_iterations,
    )
    metrics["balanced_accuracy_95ci"] = [lower, upper]
    metrics["test_rows"] = int(len(labels))
    metrics["test_source"] = "Original MRI images only"

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    report = pd.DataFrame(metrics["classification_report"]).T.reset_index(
        names="class"
    )
    report.to_csv(output_dir / "classification_report.csv", index=False)

    prediction_rows = pd.DataFrame(
        {
            "path": paths,
            "actual_index": labels,
            "actual_label": [CLASS_NAMES[index] for index in labels],
            "predicted_index": predictions,
            "predicted_label": [CLASS_NAMES[index] for index in predictions],
            "confidence": probabilities.max(axis=1),
            "correct": labels == predictions,
        }
    )
    for index, class_name in enumerate(CLASS_NAMES):
        prediction_rows[f"probability_{class_name}"] = probabilities[:, index]
    prediction_rows.to_csv(output_dir / "test_predictions.csv", index=False)

    matrix = np.asarray(metrics["confusion_matrix"])
    save_confusion_matrix(matrix, output_dir / "confusion_matrix.png")
    save_roc_curves(labels, probabilities, output_dir / "roc_curves.png")
    save_reliability(labels, probabilities, output_dir / "reliability_curve.png")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
