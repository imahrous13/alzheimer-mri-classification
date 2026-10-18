from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import fftpack
from scipy.ndimage import laplace
from tqdm import tqdm

from .dataset import CLASS_NAMES
from .paths import PROCESSED_DATA_DIR, RAW_DATA_DIR, RESULTS_DIR, ensure_output_dirs


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def entropy(array: np.ndarray) -> float:
    """Shannon entropy (bits) of an 8-bit grayscale array."""
    histogram = np.bincount(
        np.asarray(array, dtype=np.uint8).ravel(), minlength=256
    ).astype(np.float64)
    probabilities = histogram / histogram.sum()
    nonzero = probabilities > 0
    return float(-(probabilities[nonzero] * np.log2(probabilities[nonzero])).sum())


def perceptual_hash(image: Image.Image, hash_size: int = 8, factor: int = 4) -> str:
    """Return a 64-bit DCT perceptual hash as a 16-character hex string."""
    size = hash_size * factor
    resized = image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    pixels = np.asarray(resized, dtype=np.float32)
    dct = fftpack.dct(fftpack.dct(pixels, axis=0, norm="ortho"), axis=1, norm="ortho")
    low_frequency = dct[:hash_size, :hash_size]
    median = float(np.median(low_frequency[low_frequency != low_frequency[0, 0]]))
    bits = (low_frequency > median).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return format(value, "016x")


def inspect_image(path: Path, dataset: str, label: str) -> dict:
    record = {
        "dataset": dataset,
        "label": label,
        "path": path.relative_to(RAW_DATA_DIR).as_posix(),
        "readable": False,
        "width": 0,
        "height": 0,
        "mode": "",
        "sha256": "",
        "phash": "",
        "mean_intensity": 0.0,
        "std_intensity": 0.0,
        "entropy": 0.0,
        "sharpness": 0.0,
        "error": "",
    }
    try:
        record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        with Image.open(path) as source:
            source.load()
            gray = source.convert("L")
            pixels = np.asarray(gray, dtype=np.float32)
            record.update(
                {
                    "readable": True,
                    "width": source.width,
                    "height": source.height,
                    "mode": source.mode,
                    "phash": perceptual_hash(source),
                    "mean_intensity": float(pixels.mean()),
                    "std_intensity": float(pixels.std()),
                    "entropy": entropy(pixels),
                    "sharpness": float(laplace(pixels).var()),
                }
            )
    except Exception as error:  # noqa: BLE001 - record any decoder failure
        record["error"] = f"{type(error).__name__}: {error}"
    return record


def discover(raw_dir: Path) -> list[tuple[Path, str, str]]:
    discovered: list[tuple[Path, str, str]] = []
    for dataset_dir in sorted(path for path in raw_dir.iterdir() if path.is_dir()):
        for label_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
            for image_path in sorted(label_dir.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_SUFFIXES:
                    discovered.append((image_path, dataset_dir.name, label_dir.name))
    return discovered


def summarize(manifest: pd.DataFrame) -> dict:
    readable = manifest[manifest["readable"]].copy()

    exact_groups = readable.groupby("sha256")
    exact_duplicate_rows = int(
        exact_groups["path"].transform("size").gt(1).sum()
    )
    perceptual_groups = readable.groupby(["label", "phash"])
    perceptual_duplicate_rows = int(
        perceptual_groups["path"].transform("size").gt(1).sum()
    )

    cross_exact = int((exact_groups["dataset"].nunique() > 1).sum())
    cross_perceptual = int((perceptual_groups["dataset"].nunique() > 1).sum())

    class_counts = (
        readable.groupby(["dataset", "label"]).size().rename("count").reset_index()
    )
    dimensions = (
        readable.groupby(["dataset", "width", "height"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
    )
    intensity = (
        readable.groupby(["dataset", "label"])[
            ["mean_intensity", "std_intensity", "entropy", "sharpness"]
        ]
        .agg(["mean", "median", "std"])
    )
    intensity.columns = [f"{metric}_{stat}" for metric, stat in intensity.columns]
    intensity = intensity.reset_index().round(4)

    observed_labels = set(readable["label"].unique())
    return {
        "total_files": int(len(manifest)),
        "readable_files": int(len(readable)),
        "unreadable_files": int((~manifest["readable"]).sum()),
        "datasets": sorted(readable["dataset"].unique().tolist()),
        "classes": sorted(observed_labels),
        "unexpected_labels": sorted(observed_labels - set(CLASS_NAMES)),
        "missing_labels": sorted(set(CLASS_NAMES) - observed_labels),
        "exact_duplicate_rows": exact_duplicate_rows,
        "perceptual_duplicate_rows": perceptual_duplicate_rows,
        "cross_dataset_exact_hash_groups": cross_exact,
        "cross_dataset_perceptual_hash_groups": cross_perceptual,
        "class_counts": class_counts.to_dict(orient="records"),
        "top_dimensions": dimensions.head(3).to_dict(orient="records"),
        "intensity_summary": intensity.to_dict(orient="records"),
        "_dimensions": dimensions,
        "_exact_duplicates": readable[
            readable["sha256"].isin(
                exact_groups.size()[exact_groups.size() > 1].index
            )
        ],
        "_perceptual_duplicates": readable[
            readable.set_index(["label", "phash"]).index.isin(
                perceptual_groups.size()[perceptual_groups.size() > 1].index
            )
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the Alzheimer MRI image collection.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    discovered = discover(args.raw_dir)

    def task(item: tuple[Path, str, str]) -> dict:
        path, dataset, label = item
        return inspect_image(path, dataset, label)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        records = list(
            tqdm(
                executor.map(task, discovered),
                total=len(discovered),
                desc="Auditing images",
                unit="image",
            )
        )
    manifest = pd.DataFrame(records)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(PROCESSED_DATA_DIR / "image_manifest.csv", index=False)

    summary = summarize(manifest)
    dimensions = summary.pop("_dimensions")
    exact_duplicates = summary.pop("_exact_duplicates")
    perceptual_duplicates = summary.pop("_perceptual_duplicates")

    output_dir = RESULTS_DIR / "data_quality"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    pd.DataFrame(summary["class_counts"]).to_csv(
        output_dir / "class_distribution.csv", index=False
    )
    dimensions.to_csv(output_dir / "dimension_distribution.csv", index=False)
    exact_duplicates.to_csv(output_dir / "exact_duplicates.csv", index=False)
    perceptual_duplicates.to_csv(output_dir / "perceptual_duplicates.csv", index=False)
    manifest[~manifest["readable"]].to_csv(
        output_dir / "unreadable_files.csv", index=False
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
