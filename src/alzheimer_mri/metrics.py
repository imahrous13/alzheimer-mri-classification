from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=False):
        mask = (confidences > lower) & (confidences <= upper)
        if not mask.any():
            continue
        bin_accuracy = (predictions[mask] == labels[mask]).mean()
        ece += mask.mean() * abs(bin_accuracy - confidences[mask].mean())
    return float(ece)


def classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray, class_names: list[str]
) -> dict:
    predictions = probabilities.argmax(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_precision": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
        "expected_calibration_error": expected_calibration_error(labels, probabilities),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }
    try:
        metrics["macro_roc_auc_ovr"] = float(
            roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")
        )
    except ValueError:
        metrics["macro_roc_auc_ovr"] = None
    return metrics


def bootstrap_interval(
    labels: np.ndarray,
    predictions: np.ndarray,
    metric,
    iterations: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(iterations):
        indices = rng.integers(0, len(labels), len(labels))
        estimates.append(metric(labels[indices], predictions[indices]))
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return float(lower), float(upper)
