from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .paths import PROCESSED_DATA_DIR, ensure_output_dirs


def hamming_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def cluster_hashes(hashes: list[str], radius: int) -> dict[str, str]:
    """Cluster perceptual hashes using banded candidate lookup and union-find."""
    unique_hashes = sorted(set(hashes))
    parent = {value: value for value in unique_hashes}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    bands: dict[tuple[int, str], list[str]] = defaultdict(list)
    band_count = 8
    band_width = len(unique_hashes[0]) // band_count if unique_hashes else 2
    for value in unique_hashes:
        candidates: set[str] = set()
        for band_index in range(band_count):
            start = band_index * band_width
            key = (band_index, value[start : start + band_width])
            candidates.update(bands[key])
        for candidate in candidates:
            if hamming_distance(value, candidate) <= radius:
                union(value, candidate)
        for band_index in range(band_count):
            start = band_index * band_width
            bands[(band_index, value[start : start + band_width])].append(value)

    return {value: find(value) for value in unique_hashes}


def assign_holdout_groups(
    original: pd.DataFrame,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, str]:
    rng = random.Random(seed)
    assignments: dict[str, str] = {}
    for _label, label_rows in original.groupby("label"):
        groups = sorted(label_rows["leakage_group"].unique())
        rng.shuffle(groups)
        total = len(groups)
        test_count = max(1, round(total * test_fraction))
        validation_count = max(1, round(total * validation_fraction))
        for group in groups[:test_count]:
            assignments[group] = "test"
        for group in groups[test_count : test_count + validation_count]:
            assignments[group] = "validation"
        for group in groups[test_count + validation_count :]:
            assignments[group] = "train"
    return assignments


def create_splits(
    manifest: pd.DataFrame,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    phash_radius: int = 4,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    usable = manifest[manifest["readable"]].copy()
    usable["leakage_group"] = ""

    for label, indices in usable.groupby("label").groups.items():
        hashes = usable.loc[indices, "phash"].astype(str).tolist()
        clusters = cluster_hashes(hashes, radius=phash_radius)
        usable.loc[indices, "leakage_group"] = [f"{label}:{clusters[value]}" for value in hashes]

    original = usable[usable["dataset"] == "original"].copy()
    assignments = assign_holdout_groups(
        original,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )

    usable["split"] = "train"
    usable.loc[usable["dataset"] == "original", "split"] = usable.loc[
        usable["dataset"] == "original", "leakage_group"
    ].map(assignments)

    held_out_groups = {
        group for group, split in assignments.items() if split in {"validation", "test"}
    }
    augmented_holdout_matches = (
        (usable["dataset"] != "original") & usable["leakage_group"].isin(held_out_groups)
    )
    usable.loc[augmented_holdout_matches, "split"] = "excluded_leakage_guard"

    split_counts = (
        usable.groupby(["split", "dataset", "label"]).size().rename("count").reset_index()
    )
    summary = {
        "seed": seed,
        "phash_hamming_radius": phash_radius,
        "validation_fraction_of_original_groups": validation_fraction,
        "test_fraction_of_original_groups": test_fraction,
        "rows": int(len(usable)),
        "leakage_groups": int(usable["leakage_group"].nunique()),
        "augmented_rows_excluded_by_leakage_guard": int(augmented_holdout_matches.sum()),
        "split_counts": split_counts.to_dict(orient="records"),
    }
    return usable, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create leakage-aware MRI dataset splits.")
    parser.add_argument(
        "--manifest", type=Path, default=PROCESSED_DATA_DIR / "image_manifest.csv"
    )
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument(
        "--phash-radius",
        type=int,
        default=0,
        help=(
            "Perceptual-hash Hamming radius. The audited dataset uses 0 by "
            "default because larger radii create transitive chains that merge "
            "many distinct scans into oversized groups."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    manifest = pd.read_csv(args.manifest)
    splits, summary = create_splits(
        manifest,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        phash_radius=args.phash_radius,
        seed=args.seed,
    )
    splits.to_csv(PROCESSED_DATA_DIR / "dataset_splits.csv", index=False)
    (PROCESSED_DATA_DIR / "split_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
