from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from skimage.feature import hog
from tqdm import tqdm

from .dataset import CLASS_NAMES, CLASS_TO_INDEX, load_split_rows
from .paths import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, RESULTS_DIR, ensure_output_dirs


def image_features(path: Path, image_size: int = 64) -> np.ndarray:
    """Extract contrast-normalized HOG plus a coarse intensity histogram."""
    with Image.open(path) as source:
        gray = source.convert("L").resize((image_size, image_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(gray, dtype=np.float32) / 255.0
    gradients = hog(
        pixels,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)
    histogram, _ = np.histogram(pixels, bins=16, range=(0.0, 1.0), density=True)
    histogram = (histogram / (histogram.sum() + 1e-8)).astype(np.float32)
    return np.concatenate([gradients, histogram])


def featurize(
    rows: pd.DataFrame, image_size: int, workers: int, cache_name: str
) -> tuple[np.ndarray, np.ndarray]:
    paths = [RAW_DATA_DIR / path for path in rows["path"]]
    fingerprint = hashlib.sha256(
        "\n".join(rows["path"].astype(str)).encode("utf-8")
    ).hexdigest()[:12]
    cache_dir = PROCESSED_DATA_DIR / "features"
    cache_path = cache_dir / f"{cache_name}_{image_size}_{fingerprint}.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        return cached["features"], cached["labels"]

    def extract(path: Path) -> np.ndarray:
        return image_features(path, image_size=image_size)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        features = list(
            tqdm(
                executor.map(extract, paths),
                total=len(paths),
                desc="Extracting HOG",
                unit="image",
            )
        )
    labels = rows["label"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    feature_matrix = np.stack(features)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, features=feature_matrix, labels=labels)
    return feature_matrix, labels


def metric_summary(labels: np.ndarray, predictions: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the CPU HOG baseline.")
    parser.add_argument("--splits", type=Path, default=PROCESSED_DATA_DIR / "dataset_splits.csv")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--train-source",
        choices=["all", "original"],
        default="all",
        help="Whether training may use the augmented collection or original images only.",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "baseline")
    parser.add_argument("--model-name", type=str, default="hog_sgd_classifier.joblib")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    started = time.time()
    train_rows = load_split_rows(args.splits, "train")
    if args.train_source == "original":
        train_rows = train_rows[train_rows["dataset"] == "original"].copy()
    validation_rows = load_split_rows(args.splits, "validation")
    test_rows = load_split_rows(args.splits, "test")

    train_x, train_y = featurize(
        train_rows, args.image_size, args.workers, f"train_{args.train_source}"
    )
    validation_x, validation_y = featurize(validation_rows, args.image_size, args.workers, "validation")
    test_x, test_y = featurize(test_rows, args.image_size, args.workers, "test")

    candidates = (3e-5, 1e-4, 3e-4)
    validation_results = []
    best_model = None
    best_score = -1.0
    for alpha in candidates:
        model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=alpha,
            class_weight="balanced",
            max_iter=args.max_iterations,
            tol=args.tolerance,
            average=True,
            n_jobs=-1,
            random_state=args.seed,
        )
        model.fit(train_x, train_y)
        predictions = model.predict(validation_x)
        metrics = {"alpha": alpha, **metric_summary(validation_y, predictions)}
        validation_results.append(metrics)
        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            best_model = model

    assert best_model is not None
    test_predictions = best_model.predict(test_x)
    probabilities = best_model.predict_proba(test_x)
    test_metrics = metric_summary(test_y, test_predictions)
    report = classification_report(
        test_y,
        test_predictions,
        labels=range(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    model_path = MODELS_DIR / args.model_name
    joblib.dump(
        {
            "model": best_model,
            "class_names": list(CLASS_NAMES),
            "image_size": args.image_size,
            "feature_method": "HOG + 16-bin normalized intensity histogram",
            "train_source": args.train_source,
        },
        model_path,
    )

    result_dir = args.output_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(validation_results).to_csv(result_dir / "validation_search.csv", index=False)
    pd.DataFrame(report).T.to_csv(result_dir / "classification_report.csv")
    prediction_rows = test_rows[["path", "label"]].copy()
    prediction_rows["prediction"] = [CLASS_NAMES[index] for index in test_predictions]
    prediction_rows["confidence"] = probabilities.max(axis=1)
    for index, name in enumerate(CLASS_NAMES):
        prediction_rows[f"probability_{name}"] = probabilities[:, index]
    prediction_rows.to_csv(result_dir / "test_predictions.csv", index=False)

    matrix = confusion_matrix(test_y, test_predictions, labels=range(len(CLASS_NAMES)))
    fig, axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axis)
    axis.set(xlabel="Predicted", ylabel="Actual", title="HOG baseline - held-out original MRI test set")
    fig.tight_layout()
    fig.savefig(result_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    summary = {
        "model": "HOG features + class-balanced SGD logistic classifier",
        "train_source": args.train_source,
        "selection_metric": "validation macro-F1",
        "best_validation_macro_f1": best_score,
        "selected_alpha": best_model.alpha,
        "test_metrics": test_metrics,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "test_rows": len(test_rows),
        "duration_seconds": round(time.time() - started, 2),
        "model_path": model_path.relative_to(model_path.parents[1]).as_posix(),
    }
    (result_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
