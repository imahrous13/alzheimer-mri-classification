# Alzheimer MRI classification — project report

This report summarizes an end-to-end, leakage-aware computer-vision study on the Kaggle
*Augmented Alzheimer MRI* dataset. Every number below is produced by the committed pipeline and
stored under `results/`.

> Educational research prototype. Not a medical diagnostic system. Directory labels are treated
> as supplied ground truth and are not clinically verified.

## 1. Executive summary

- **40,384** MRI images were audited; all decoded successfully.
- A leakage trap was quantified: **3,626** same-label perceptual-hash groups span the original and
  augmented collections, so a naive random split would leak near-duplicates across train/test.
- Leakage-controlled splits put **961** original images in validation and **963** in test, and
  quarantined **2,491** augmented images whose perceptual group matched a held-out scan.
- A CPU-friendly **HOG + linear** baseline reaches **65.2% accuracy / 0.598 macro-F1 / 0.868 ROC-AUC**
  on the held-out original test set.
- A from-scratch **CNN** trained on the clean original-only split reaches 58.0% accuracy /
  0.484 macro-F1 — an honest reference showing where deep learning needs more data/compute.

![Pipeline](../docs/figures/pipeline.png)

## 2. Data audit

![Class distribution and split composition](../docs/figures/distributions.png)
| Check | Value |
|---|---:|
| Images discovered | 40,384 |
| Images decoded | 40,384 (100%) |
| Exact SHA-256 duplicate rows | 657 |
| Same-label perceptual duplicate rows | 14,981 |
| Cross-source exact-hash groups | 0 |
| **Cross-source perceptual-hash groups** | **3,626** |

![Leakage example](../docs/figures/leakage_example.png)

Class counts (original): MildDemented 896, ModerateDemented 64, NonDemented 3,200,
VeryMildDemented 2,240. The moderate class is extremely under-represented.
Image statistics (mean intensity, contrast, entropy, Laplacian sharpness) are summarized per
dataset/class in `results/data_quality/audit_summary.json`; sharpness is markedly lower for
augmented images, a potential source shortcut.

## 3. Split design

Validation and test are drawn **only from original images**, grouped by perceptual hash so that
copy variants cannot straddle the split. Each augmented image is admitted to training only if its
perceptual group is absent from the held-out groups.

| Split | Rows |
|---|---:|
| Train (augmented + original) | 35,969 |
| Validation (original only) | 961 |
| Test (original only) | 963 |
| Quarantined (augmented leakage guard) | 2,491 |

Test-set class support: MildDemented 134, ModerateDemented 10, NonDemented 485, VeryMildDemented 334.

## 4. Models

**Baseline (HOG + linear).** Grayscale 64x64; 9-orientation HOG (1,764 dims) + 16-bin intensity
histogram; class-balanced SGD logistic classifier; L2 regularization selected on validation macro-F1.

**Compact CNN.** Four Conv-BN-ReLU blocks with global average pooling, dropout 0.35, AdamW,
class-weighted cross-entropy, ReduceLROnPlateau, and early stopping on validation macro-F1.
Committed run: `image_size=64`, `batch_size=128`, original-only training split (~4.5k images), CPU.

## 5. Results (held-out original test set, n = 963)

| Model | Training data | Accuracy | Balanced acc. | Macro-F1 | Macro ROC-AUC | ECE |
|---|---|---:|---:|---:|---:|---:|
| HOG + linear | augmented + original | **0.652** | **0.701** | **0.598** | 0.868 | 0.224 |
| HOG + linear | original only | 0.626 | 0.648 | 0.593 | 0.831 | 0.351 |
| Compact CNN | original only | 0.580 | 0.586 | 0.484 | 0.851 | **0.140** |

Bootstrap 95% intervals (1,000 resamples) for the headline baseline:

- accuracy [0.622, 0.682]; balanced accuracy [0.641, 0.746]; macro-F1 [0.530, 0.652].
- CNN balanced accuracy [0.528, 0.624].

![Model comparison](../docs/figures/model_comparison.png)

| ROC (ranking quality) | Reliability (calibration) |
|---|---|
| ![ROC](../docs/figures/roc_curves.png) | ![Calibration](../docs/figures/calibration.png) |

### Per-class performance (HOG baseline)

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| MildDemented | 0.439 | 0.642 | 0.521 | 134 |
| ModerateDemented | 0.391 | 0.900 | 0.545 | 10 |
| NonDemented | 0.761 | 0.742 | 0.752 | 485 |
| VeryMildDemented | 0.638 | 0.518 | 0.572 | 334 |

### Per-class performance (Compact CNN)

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| MildDemented | 1.000 | 0.060 | 0.113 | 134 |
| ModerateDemented | 0.429 | 0.900 | 0.581 | 10 |
| NonDemented | 0.856 | 0.526 | 0.651 | 485 |
| VeryMildDemented | 0.451 | 0.859 | 0.592 | 334 |

### Interpretation

- **Balanced accuracy > accuracy** for the baseline: the class-balanced objective trades some
  majority-class accuracy for better rare-class recall.
- **The CNN underperforms the baseline here.** Trained from scratch on ~4.5k images at 64x64 on
  CPU, it lacks the data and resolution to beat a strong classical descriptor. Its confusion
  matrix shows it collapses most `MildDemented` images into other classes.
- **Calibration differs:** the CNN is better calibrated (ECE 0.140) than the ensemble-free linear
  model (ECE 0.224), even though its accuracy is lower.

| HOG feature space (PCA) | Most-confident errors |
|---|---|
| ![Embedding](../docs/figures/hog_embedding.png) | ![Errors](../docs/figures/error_gallery.png) |

## 6. Reproduce

```bash
python -m alzheimer_mri.audit --workers 12
python -m alzheimer_mri.splits --phash-radius 0 --seed 42
python -m alzheimer_mri.baseline --workers 12
python -m alzheimer_mri.baseline --train-source original --output-dir results/baseline_original --model-name hog_sgd_classifier_original.joblib
python -m alzheimer_mri.train --train-source original --image-size 64 --batch-size 128 --epochs 12 --patience 3 --checkpoint-name best_model.pt
python -m alzheimer_mri.evaluate --checkpoint models/best_model.pt --output-dir results/evaluation
python scripts/enhance_baseline_results.py
python scripts/build_notebook.py
python scripts/build_dashboard.py
```

## 7. Limitations and next steps

- No patient identifiers → perceptual grouping is a proxy, not patient-level isolation.
- Tiny `ModerateDemented` holdout (n = 10) → wide uncertainty on that class.
- Single dataset, no external cohort, no scanner/metadata controls.
- Possible source shortcut from augmentation-induced sharpness/resolution differences.
- Next: patient-level splits, transfer learning on GPU, calibration study, and external validation.
