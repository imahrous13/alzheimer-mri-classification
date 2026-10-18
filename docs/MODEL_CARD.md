# Model card — Alzheimer MRI classifiers

## Overview

Two models are trained and evaluated on the same leakage-controlled, original-only held-out test
set (`n = 963`). Both are educational research prototypes, **not** medical devices.

| Field | Value |
|---|---|
| Task | 4-class MRI classification (`MildDemented`, `ModerateDemented`, `NonDemented`, `VeryMildDemented`) |
| Input | single grayscale MRI slice, resized to 64x64 |
| Models | (a) HOG + intensity histogram + class-balanced linear classifier; (b) compact CNN |
| Framework | scikit-learn (baseline), PyTorch (CNN) |
| Selection metric | validation macro-F1 |

## Intended use

Research and portfolio demonstration: reproducible benchmarking, data-quality auditing, and
leakage-controlled evaluation. Explaining how dataset construction affects reported metrics.

## Out-of-scope use

Clinical diagnosis, triage, or any patient-facing decision. The models have no external validation,
no patient-level holdout, and no regulatory review.

## Training data

Kaggle *Augmented Alzheimer MRI Dataset* (6,400 original + 33,984 augmented images). Validation and
test use original images only. See [DATA_CARD.md](DATA_CARD.md).

## Evaluation (held-out original-only test set, n = 963)

| Model | Accuracy | Balanced acc. | Macro-F1 | Macro ROC-AUC | ECE |
|---|---:|---:|---:|---:|---:|
| HOG + linear (all data) | 0.652 | 0.701 | 0.598 | 0.868 | 0.224 |
| HOG + linear (original only) | 0.626 | 0.648 | 0.593 | 0.831 | 0.351 |
| Compact CNN (original only) | 0.580 | 0.586 | 0.484 | 0.851 | 0.140 |

Per-class metrics, confusion matrices, ROC curves, and reliability curves are in `results/`.

## Ethical considerations

- **Do not** present outputs as diagnoses. Confidence is a model score, not clinical certainty.
- The dataset lacks patient identifiers and acquisition metadata, so performance gaps across
  populations, scanners, and sites are unmeasured.
- `ModerateDemented` has only 64 original images (test `n = 10`); treat its metrics as preliminary.
- Risk of over-reliance: the interface flags low-confidence predictions for human review, but no
  automated decision should be made from these outputs.

## Maintenance

Re-running `scripts/run_pipeline.ps1` / `.sh` reproduces every number. Intermediate artifacts are
versioned under `results/`; large inputs (images, feature caches, CNN checkpoints) are gitignored.
