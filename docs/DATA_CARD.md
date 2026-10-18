# Dataset card

## Scope

This repository uses the **Augmented Alzheimer MRI Dataset** distributed on Kaggle. The local
copy contains an original image collection and a larger augmented collection, each organized
into four directory labels:

- `MildDemented`
- `ModerateDemented`
- `NonDemented`
- `VeryMildDemented`

Files are stored locally under `data/raw/` for reproducibility but are excluded from Git
because of size. Source: <https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset>.

## Audited inventory

| Dataset | Mild | Moderate | Non-demented | Very mild | Total |
|---|---:|---:|---:|---:|---:|
| Original | 896 | 64 | 3,200 | 2,240 | 6,400 |
| Augmented | 8,960 | 6,464 | 9,600 | 8,960 | 33,984 |
| **Total** | **9,856** | **6,528** | **12,800** | **11,200** | **40,384** |

All 40,384 files decoded successfully. Original images are 176x208 pixels; augmented images
are a mixture of 180x180 and 200x190 pixels.

## Quality and leakage findings

- 657 rows belong to an exact SHA-256 duplicate group.
- 14,981 rows belong to a same-label perceptual-hash duplicate group.
- No byte-identical hash group crosses the original and augmented directories.
- 3,626 same-label perceptual-hash groups cross those directories, indicating transformed or
  re-encoded variants of original scans.

The split generator groups matching perceptual hashes before assigning holdouts. Validation and
test contain original images only. Any augmented image that matches a held-out perceptual group
is labeled `excluded_leakage_guard` and is never used for training.

## Final split

| Split | Rows | Source policy |
|---|---:|---|
| Train | 35,969 | Original training groups plus safe augmented images |
| Validation | 961 | Original images only |
| Test | 963 | Original images only |
| Leakage guard | 2,491 | Augmented images matching validation/test groups |

The split is deterministic (`seed=42`) with a perceptual-hash Hamming radius of 0 (exact
64-bit match). A radius of 4 was tested but rejected: transitive single-link clustering merged
many distinct scans into oversized groups and produced an unusable class-stratified test set.

## Known limitations

- Image-level class folders but no patient identifiers. Perceptual grouping reduces image-copy
  leakage; it cannot guarantee patient-independent splitting.
- The original `ModerateDemented` class has only 64 images, so its validation/test estimates
  have high uncertainty (test n = 10).
- Provenance fields (scanner, acquisition protocol, institution, demographics, clinical
  adjudication) are unavailable.
- Augmentation recipes and lineage identifiers are not encoded per image.
- Directory labels are treated as supplied ground truth; diagnoses are not independently verified.

This dataset is appropriate for educational benchmarking, not clinical validation or diagnostic use.
