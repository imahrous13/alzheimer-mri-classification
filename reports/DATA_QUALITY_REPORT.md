# Alzheimer MRI data-quality report

## Executive finding

The collection is technically readable and complete at the file level, but it is not safe for a
naive random image split. The audit found **3,626 perceptual-hash groups shared between the
original and augmented directories**. A model evaluated after mixing these variants could score
well by recognizing transformations of images already seen during training.

The implemented split quarantines **2,491 augmented images** that match held-out original scans.
The resulting evaluation uses **961 original validation images** and **963 original test images**,
while **35,969 images** remain available for training.

## What was checked

- file discovery and supported extensions;
- decoder integrity (every file opened and decoded);
- expected and unexpected class folders;
- class balance by source collection;
- dimensions and image modes;
- brightness, contrast, entropy, and Laplacian sharpness summaries;
- byte-level duplicate hashes; and
- perceptual-hash overlap within and across datasets.

## Material observations

1. **Integrity is strong.** Every one of the 40,384 discovered images decoded successfully and
   all four expected labels were present.
2. **The original data are highly imbalanced.** Moderate dementia has 64 original images versus
   3,200 non-demented images. Aggregate accuracy alone would be misleading.
3. **Augmentation changes the apparent balance.** The augmented collection is larger and more
   even, but augmented volume is not independent clinical evidence.
4. **Cross-source perceptual overlap is extensive.** No byte-identical group crosses sources,
   yet 3,626 same-label perceptual groups do. Re-encoding and transformations explain why byte
   hashes alone are insufficient.
5. **Resolution is source-dependent.** Original scans are 176x208; augmented files are 180x180 or
   200x190. The pipeline normalizes dimensions at load time.
6. **Sharpness differs by source.** Mean Laplacian variance is substantially lower in the
   augmented collection (e.g. ~300 vs ~650), consistent with resizing/interpolation. This creates
   a possible source cue a model could exploit.

![Source statistics](../docs/figures/source_statistics.png)
![Leakage example](../docs/figures/leakage_example.png)

## Trust decision

The data are suitable for a portfolio-grade educational experiment with explicit leakage controls
and limitations. They are not sufficient for claims of patient-level generalization or clinical
performance because patient identifiers, acquisition metadata, and external cohorts are absent.

## Reproduce

```bash
python -m alzheimer_mri.audit --workers 12
python -m alzheimer_mri.splits --phash-radius 0 --seed 42
```

Machine-readable evidence is saved under `results/data_quality/` and `data/processed/`.
