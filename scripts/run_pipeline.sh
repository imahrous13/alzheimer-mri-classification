#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/src"
cd "$PROJECT_ROOT"

echo "1/9  Auditing images..."
python -m alzheimer_mri.audit --workers 12
echo "2/9  Building leakage-aware splits..."
python -m alzheimer_mri.splits --phash-radius 0 --seed 42
echo "3/9  Training HOG baseline (all data)..."
python -m alzheimer_mri.baseline --workers 12
echo "4/9  Training HOG baseline (original-only)..."
python -m alzheimer_mri.baseline --train-source original --output-dir results/baseline_original --model-name hog_sgd_classifier_original.joblib --workers 12
echo "5/9  Training compact CNN (original-only, CPU)..."
python -m alzheimer_mri.train --train-source original --image-size 64 --batch-size 128 --epochs 12 --patience 3 --checkpoint-name best_model.pt
echo "6/9  Evaluating CNN on held-out test set..."
python -m alzheimer_mri.evaluate --checkpoint models/best_model.pt --batch-size 128 --output-dir results/evaluation
echo "7/9  Enriching metrics with bootstrap intervals..."
python scripts/enhance_baseline_results.py
echo "8/9  Building and executing analysis notebook..."
python scripts/build_notebook.py
jupyter nbconvert --to notebook --execute notebooks/alzheimer_mri_audit_and_results.ipynb \
  --output alzheimer_mri_audit_and_results.ipynb \
  --output-dir notebooks \
  --ExecutePreprocessor.timeout=600
echo "9/9  Rendering figures and dashboard..."
python scripts/make_figures.py
python scripts/build_dashboard.py
echo "Done."
