from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .baseline import image_features
from .paths import MODELS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the trained HOG baseline on MRI images.")
    parser.add_argument("input", type=Path, help="Image file or directory of images.")
    parser.add_argument("--model", type=Path, default=MODELS_DIR / "hog_sgd_classifier.joblib")
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--review-threshold", type=float, default=0.70)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = joblib.load(args.model)
    paths = [args.input] if args.input.is_file() else sorted(
        path for path in args.input.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    features = [image_features(path, bundle["image_size"]) for path in paths]
    probabilities = bundle["model"].predict_proba(features)
    class_names = bundle["class_names"]
    rows = []
    for path, scores in zip(paths, probabilities, strict=False):
        best = int(scores.argmax())
        row = {
            "path": path.as_posix(),
            "prediction": class_names[best],
            "confidence": float(scores[best]),
            "review_recommended": bool(scores[best] < args.review_threshold),
        }
        row.update({f"probability_{name}": float(scores[index]) for index, name in enumerate(class_names)})
        rows.append(row)
    output = pd.DataFrame(rows)
    output.to_csv(args.output, index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
