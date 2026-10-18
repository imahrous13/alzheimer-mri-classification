"""Add richer metrics and nonparametric intervals to baseline outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from alzheimer_mri.dataset import CLASS_NAMES, CLASS_TO_INDEX
from alzheimer_mri.metrics import bootstrap_interval, classification_metrics

ROOT = Path(__file__).resolve().parents[1]


def enrich(result_dir: Path) -> None:
    predictions = pd.read_csv(result_dir / "test_predictions.csv")
    labels = predictions["label"].map(CLASS_TO_INDEX).to_numpy()
    probability_columns = [f"probability_{name}" for name in CLASS_NAMES]
    probabilities = predictions[probability_columns].to_numpy()
    predicted = probabilities.argmax(axis=1)

    rich = classification_metrics(labels, probabilities, list(CLASS_NAMES))
    intervals = {
        "accuracy_95pct": bootstrap_interval(labels, predicted, accuracy_score),
        "balanced_accuracy_95pct": bootstrap_interval(labels, predicted, balanced_accuracy_score),
        "macro_f1_95pct": bootstrap_interval(
            labels,
            predicted,
            lambda actual, estimate: f1_score(actual, estimate, average="macro", zero_division=0),
        ),
    }
    metrics_path = result_dir / "metrics.json"
    summary = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary["test_metrics"] = {
        key: value for key, value in rich.items() if key not in {"confusion_matrix", "classification_report"}
    }
    summary["bootstrap_intervals"] = intervals
    summary["evaluation_note"] = (
        "Percentile bootstrap intervals use 1,000 row-level resamples. They do not account "
        "for unknown patient clustering because patient identifiers are unavailable."
    )
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    for relative in ("results/baseline", "results/baseline_original"):
        candidate = ROOT / relative
        if (candidate / "test_predictions.csv").exists():
            enrich(candidate)


if __name__ == "__main__":
    main()
