$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
Set-Location $ProjectRoot

Write-Host "1/9  Auditing images..."        -ForegroundColor Cyan
python -m alzheimer_mri.audit --workers 12
Write-Host "2/9  Building leakage-aware splits..." -ForegroundColor Cyan
python -m alzheimer_mri.splits --phash-radius 0 --seed 42
Write-Host "3/9  Training HOG baseline (all data)..." -ForegroundColor Cyan
python -m alzheimer_mri.baseline --workers 12
Write-Host "4/9  Training HOG baseline (original-only)..." -ForegroundColor Cyan
python -m alzheimer_mri.baseline --train-source original --output-dir results/baseline_original --model-name hog_sgd_classifier_original.joblib --workers 12
Write-Host "5/9  Training compact CNN (original-only, CPU)..." -ForegroundColor Cyan
python -m alzheimer_mri.train --train-source original --image-size 64 --batch-size 128 --epochs 12 --patience 3 --checkpoint-name best_model.pt
Write-Host "6/9  Evaluating CNN on held-out test set..." -ForegroundColor Cyan
python -m alzheimer_mri.evaluate --checkpoint models/best_model.pt --batch-size 128 --output-dir results/evaluation
Write-Host "7/9  Enriching metrics with bootstrap intervals..." -ForegroundColor Cyan
python scripts/enhance_baseline_results.py
Write-Host "8/9  Building and executing analysis notebook..." -ForegroundColor Cyan
python scripts/build_notebook.py
python -m jupyter nbconvert --to notebook --execute notebooks/alzheimer_mri_audit_and_results.ipynb `
    --output alzheimer_mri_audit_and_results.ipynb `
    --output-dir notebooks `
    --ExecutePreprocessor.timeout=600
Write-Host "9/9  Rendering figures and dashboard..." -ForegroundColor Cyan
python scripts/make_figures.py
python scripts/build_dashboard.py
Write-Host "Done." -ForegroundColor Green
