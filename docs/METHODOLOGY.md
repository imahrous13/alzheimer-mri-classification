# Methodology

## Reproducible workflow

1. `alzheimer_mri.audit` inventories every image, verifies decoding, calculates image statistics,
   and generates SHA-256 and DCT perceptual hashes.
2. `alzheimer_mri.splits` groups same-label matching perceptual hashes and creates deterministic
   train, validation, test, and leakage-guard partitions.
3. `alzheimer_mri.baseline` extracts HOG and intensity-histogram features, selects regularization
   on validation macro-F1, and evaluates once on the held-out test set.
4. `alzheimer_mri.train` provides a compact grayscale CNN with class-weighted loss, augmentation,
   early stopping, and checkpointing.
5. `alzheimer_mri.evaluate` produces CNN metrics, bootstrap intervals, a classification report,
   confusion matrix, one-vs-rest ROC curves, calibration figures, and row-level predictions.
6. `alzheimer_mri.predict_baseline` and `alzheimer_mri.predict` support batch inference; low-confidence
   outputs are explicitly flagged for review.
7. `alzheimer_mri.explain` creates Grad-CAM overlays for CNN predictions.

## Leakage control

The supplied augmented directory contains many transformed copies of original images. Randomly
splitting all images would let variants of the same underlying scan appear on both sides of
evaluation. This repository therefore:

- creates validation and test sets from the 6,400 original images only;
- assigns perceptual-hash groups atomically;
- excludes every augmented image whose group occurs in validation or test; and
- retains the excluded rows in the split manifest so the decision is auditable.

This is stronger than a naive image-level split but not equivalent to a patient-level split,
because patient identifiers are unavailable.

## Metrics

Primary model selection uses **macro-F1**, giving each class equal weight despite class imbalance.
The evaluation also reports accuracy, balanced accuracy, macro ROC-AUC, expected calibration
error, and per-class precision/recall/F1 with support, so the small `ModerateDemented` class is
not hidden by aggregate performance. Percentile bootstrap intervals (1,000 resamples) quantify
row-level uncertainty; they do not model unknown patient clustering.

## Model paths

The measured baseline is intentionally CPU-friendly:

- grayscale resize to 64x64;
- 9-orientation HOG with 8x8 cells and 2x2 blocks (1,764 features);
- 16-bin normalized intensity histogram (16 features);
- class-balanced SGD logistic classifier;
- L2 regularization selected from three `alpha` values on validation macro-F1.

The CNN path uses four convolutional blocks, batch normalization, global average pooling,
dropout, AdamW, a class-weighted cross-entropy objective, and validation macro-F1 early stopping.
The committed CNN run uses `image_size=64`, `batch_size=128`, and the original-only training
split, so it is a data-starved reference point rather than a tuned model.

## Interpretation boundary

Outputs are experimental class predictions, not diagnoses. Confidence scores are model scores,
not clinical certainty. External validation, patient-level provenance, calibration on a
representative clinical population, bias analysis, and medical-device governance would all be
required before any clinical use.
