import pandas as pd

from alzheimer_mri.splits import create_splits, hamming_distance


def test_hamming_distance():
    assert hamming_distance("0", "0") == 0
    assert hamming_distance("0", "f") == 4


def test_augmented_matches_to_holdout_are_excluded():
    rows = []
    labels = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
    for label_index, label in enumerate(labels):
        for item in range(12):
            value = f"{label_index * 100 + item:016x}"
            rows.append(
                {
                    "dataset": "original",
                    "label": label,
                    "path": f"original/{label}/{item}.jpg",
                    "readable": True,
                    "phash": value,
                }
            )
            rows.append(
                {
                    "dataset": "augmented",
                    "label": label,
                    "path": f"augmented/{label}/{item}.jpg",
                    "readable": True,
                    "phash": value,
                }
            )

    split_rows, _ = create_splits(
        pd.DataFrame(rows),
        validation_fraction=0.25,
        test_fraction=0.25,
        phash_radius=0,
        seed=7,
    )
    holdout_groups = set(
        split_rows.loc[
            split_rows["split"].isin(["validation", "test"]), "leakage_group"
        ]
    )
    augmented_matches = split_rows[
        (split_rows["dataset"] == "augmented")
        & split_rows["leakage_group"].isin(holdout_groups)
    ]
    assert not augmented_matches.empty
    assert set(augmented_matches["split"]) == {"excluded_leakage_guard"}
