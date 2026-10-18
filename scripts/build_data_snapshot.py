"""Build bounded, reviewed evidence snapshots for the local dashboard and report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def source(label: str, files: list[str], definitions: list[dict]) -> dict:
    return {
        "label": label,
        "provider": "Local audited project outputs",
        "tables": files,
        "filters": ["Audit and split generated from the local data/raw inventory"],
        "metricDefinitions": definitions,
        "evidenceFlow": [
            {"title": "Inventory", "detail": "data/raw/original and data/raw/augmented"},
            {"title": "Audit", "detail": "python -m alzheimer_mri.audit --workers 12"},
            {"title": "Split", "detail": "python -m alzheimer_mri.splits --phash-radius 0 --seed 42"},
        ],
    }


def build_snapshot(surface: str) -> dict:
    audit = json.loads((ROOT / "results/data_quality/audit_summary.json").read_text())
    split_rows = pd.read_csv(ROOT / "data/processed/dataset_splits.csv")
    metrics = json.loads((ROOT / "results/baseline/metrics.json").read_text())
    search = pd.read_csv(ROOT / "results/baseline/validation_search.csv")
    report = pd.read_csv(ROOT / "results/baseline/classification_report.csv").rename(
        columns={"Unnamed: 0": "class"}
    )

    split_counts = (
        split_rows.groupby(["split", "dataset", "label"])
        .size()
        .rename("count")
        .reset_index()
        .to_dict(orient="records")
    )
    quality_rows = [
        {"check": "Readable files", "value": audit["readable_files"], "total": audit["total_files"], "severity": "pass"},
        {"check": "Unreadable files", "value": audit["unreadable_files"], "total": audit["total_files"], "severity": "pass"},
        {"check": "Exact-duplicate rows", "value": audit["exact_duplicate_rows"], "total": audit["total_files"], "severity": "review"},
        {"check": "Perceptual-duplicate rows", "value": audit["perceptual_duplicate_rows"], "total": audit["total_files"], "severity": "review"},
        {"check": "Cross-source perceptual groups", "value": audit["cross_dataset_perceptual_hash_groups"], "total": None, "severity": "risk"},
        {"check": "Augmented rows quarantined", "value": int((split_rows["split"] == "excluded_leakage_guard").sum()), "total": int((split_rows["dataset"] == "augmented").sum()), "severity": "guarded"},
    ]
    metric_rows = [
        {"metric": key.replace("_", " ").title(), "value": value, "split": "test"}
        for key, value in metrics["test_metrics"].items()
        if isinstance(value, (int, float))
    ] + [
        {"metric": "Validation Macro F1", "value": metrics["best_validation_macro_f1"], "split": "validation"}
    ]

    model_comparison = []
    for relative, label in (("results/baseline", "HOG baseline (all training data)"), ("results/baseline_original", "HOG baseline (original-only training)")):
        candidate = ROOT / relative / "metrics.json"
        if candidate.exists():
            data = json.loads(candidate.read_text())
            model_comparison.append(
                {
                    "model": label,
                    "train_source": data.get("train_source", "all"),
                    "train_rows": data["train_rows"],
                    "accuracy": data["test_metrics"]["accuracy"],
                    "balanced_accuracy": data["test_metrics"]["balanced_accuracy"],
                    "macro_f1": data["test_metrics"]["macro_f1"],
                }
            )
    cnn_metrics = ROOT / "results/evaluation/metrics.json"
    if cnn_metrics.exists():
        data = json.loads(cnn_metrics.read_text())
        model_comparison.append(
            {
                "model": "Compact CNN (original-only training)",
                "train_source": "original",
                "train_rows": None,
                "accuracy": data["accuracy"],
                "balanced_accuracy": data["balanced_accuracy"],
                "macro_f1": data["macro_f1"],
            }
        )

    common_definitions = [
        {"label": "Image count", "definition": "Count of decoded image files at the stated source, class, and split grain."},
        {"label": "Macro-F1", "definition": "Unweighted mean of the four class F1 scores."},
        {"label": "Balanced accuracy", "definition": "Unweighted mean of per-class recall."},
    ]
    return {
        "surface": surface,
        "title": "Alzheimer MRI - data quality and model evaluation" if surface == "dashboard" else "Alzheimer MRI technical evaluation",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "status": "reviewed",
        "filters": [{"id": "class", "label": "MRI class", "field": "label", "defaultValue": "all"}],
        "queries": {
            "dataset_inventory": {
                "rows": audit["class_counts"],
                "source": source("Audited image manifest - class inventory", ["data/processed/image_manifest.csv"], common_definitions),
            },
            "quality_checks": {
                "rows": quality_rows,
                "source": source("Audited integrity and duplicate summary", ["results/data_quality/audit_summary.json"], common_definitions),
            },
            "split_distribution": {
                "rows": split_counts,
                "source": source("Leakage-aware split manifest", ["data/processed/dataset_splits.csv"], common_definitions),
            },
            "image_characteristics": {
                "rows": audit["intensity_summary"],
                "source": source("Image intensity, entropy, and sharpness summary", ["results/data_quality/audit_summary.json"], common_definitions),
            },
            "validation_search": {
                "rows": search.to_dict(orient="records"),
                "source": source("HOG baseline validation search", ["results/baseline/validation_search.csv"], common_definitions),
            },
            "test_metrics": {
                "rows": metric_rows,
                "source": source("Held-out original-only test evaluation", ["results/baseline/metrics.json"], common_definitions),
            },
            "class_metrics": {
                "rows": report.where(pd.notna(report), None).to_dict(orient="records"),
                "source": source("Held-out per-class evaluation", ["results/baseline/classification_report.csv"], common_definitions),
            },
            "model_comparison": {
                "rows": model_comparison,
                "source": source("Cross-model held-out comparison", ["results/baseline/metrics.json", "results/evaluation/metrics.json"], common_definitions),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", choices=["dashboard", "report"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_snapshot(args.surface), indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
