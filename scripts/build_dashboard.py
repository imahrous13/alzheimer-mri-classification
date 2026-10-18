"""Build a self-contained interactive HTML dashboard from reviewed project outputs.

The dashboard is a single static file (``dashboard/index.html``) with the evidence
embedded as JSON, so it renders from disk or GitHub Pages without a build step.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from build_data_snapshot import build_snapshot
from sklearn.metrics import auc, roc_curve

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "dashboard" / "index.html"

CLASS_NAMES = ("MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented")


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Alzheimer MRI - Data Quality &amp; Model Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  :root {
    --bg: #0f172a; --panel: #1e293b; --ink: #e2e8f0; --muted: #94a3b8;
    --accent: #38bdf8; --accent2: #f59e0b; --good: #34d399; --warn: #fbbf24; --risk: #f87171;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
    font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
  header { padding: 28px 32px 12px; border-bottom: 1px solid #334155; }
  h1 { margin: 0 0 6px; font-size: 23px; }
  .sub { color: var(--muted); font-size: 13px; }
  .disclaimer { margin-top: 12px; display: inline-block; padding: 6px 12px; border-radius: 999px;
    background: rgba(248,113,113,.12); color: var(--risk); font-size: 12px; font-weight: 600; }
  main { padding: 22px 32px 60px; max-width: 1240px; margin: 0 auto; }
  .kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 14px; margin-bottom: 22px; }
  .kpi { background: var(--panel); border-radius: 14px; padding: 16px 18px; border: 1px solid #334155; }
  .kpi .v { font-size: 26px; font-weight: 700; }
  .kpi .l { color: var(--muted); font-size: 12px; margin-top: 4px; text-transform: uppercase; letter-spacing: .04em; }
  .kpi.pass .v { color: var(--good); } .kpi.review .v { color: var(--warn); }
  .kpi.risk .v { color: var(--risk); } .kpi.guarded .v { color: var(--accent); }
  section { margin: 8px 0 26px; }
  h2 { font-size: 15px; color: var(--accent); text-transform: uppercase; letter-spacing: .06em;
    border-left: 3px solid var(--accent); padding-left: 10px; }
  .grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; }
  .card { background: var(--panel); border: 1px solid #334155; border-radius: 14px; padding: 10px; }
  .note { color: var(--muted); font-size: 12px; line-height: 1.5; margin-top: 6px; }
  footer { color: var(--muted); font-size: 12px; padding: 0 32px 40px; max-width: 1240px; margin: 0 auto; }
</style>
</head>
<body>
<header>
  <h1>Alzheimer MRI &mdash; Data Quality &amp; Model Dashboard</h1>
  <div class="sub">Leakage-aware computer-vision study &middot; generated __GENERATED__ &middot; evidence: local audited outputs</div>
  <div class="disclaimer">Educational research prototype &mdash; not a medical diagnostic system</div>
</header>
<main>
  <div class="kpis" id="kpis"></div>

  <section>
    <h2>1. Cohort &amp; data quality</h2>
    <div class="grid2">
      <div class="card"><div id="classDist"></div>
        <div class="note">Original collection is strongly imbalanced (64 moderate / 3,200 non-demented).
        The augmented collection is larger and more even but not independent clinical evidence.</div></div>
      <div class="card"><div id="splitDist"></div>
        <div class="note">Validation and test come only from the original images. Augmented copies whose
        perceptual hash matches a held-out scan are quarantined, never trained on.</div></div>
    </div>
    <div class="grid2" style="margin-top:16px">
      <div class="card"><div id="sharpness"></div>
        <div class="note">Mean Laplacian variance by dataset/class. Augmented images are systematically
        smoother, a source cue a naive model could exploit.</div></div>
      <div class="card"><div id="quality"></div>
        <div class="note">Integrity and duplicate checks from the full 40k-image audit.</div></div>
    </div>
  </section>

  <section>
    <h2>2. Held-out evaluation</h2>
    <div class="grid2">
      <div class="card"><div id="modelCompare"></div>
        <div class="note">Original-only held-out test set (960 images). Higher is better.</div></div>
      <div class="card"><div id="perClass"></div>
        <div class="note">HOG baseline per-class precision / recall / F1 on the held-out test set.</div></div>
    </div>
    <div class="grid2" style="margin-top:16px">
      <div class="card"><div id="confusion"></div>
        <div class="note">HOG baseline confusion matrix (held-out original test set).</div></div>
      <div class="card"><div id="search"></div>
        <div class="note">Regularization search: validation macro-F1 by L2 strength (alpha).</div></div>
    </div>
  </section>

  <section>
    <h2>3. Diagnostics &amp; error analysis</h2>
    <div class="grid2">
      <div class="card"><div id="roc"></div>
        <div class="note">One-vs-rest ROC curves for the HOG baseline on the held-out set.</div></div>
      <div class="card"><div id="calibration"></div>
        <div class="note">Reliability curve; points annotated with sample counts. Lower ECE is better.</div></div>
    </div>
    <div class="grid2" style="margin-top:16px">
      <div class="card"><div id="sharpnessHist"></div>
        <div class="note">Laplacian sharpness by source collection; augmented images are systematically smoother.</div></div>
      <div class="card"><div id="errors"></div>
        <div class="note">Most-confident baseline errors &mdash; the dementia spectrum is genuinely ambiguous.</div></div>
    </div>
  </section>

  <footer>
    Reproduce: <code>python -m alzheimer_mri.audit --workers 12</code> &rarr;
    <code>python -m alzheimer_mri.splits --phash-radius 0 --seed 42</code> &rarr;
    <code>python -m alzheimer_mri.baseline --workers 12</code>.
    Metrics are model scores, not clinical certainty.
  </footer>
</main>
<script>
const DATA = __DATA__;
const ACCENT = '#38bdf8', ACCENT2 = '#f59e0b';
const layout = (title) => ({
  title: {text: title, font: {size: 14, color: '#e2e8f0'}},
  paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
  font: {color: '#cbd5e1', size: 12}, margin: {t: 40, r: 12, b: 50, l: 55},
  legend: {orientation: 'h', y: -0.2}
});
const config = {displayModeBar: false, responsive: true};

function kpis() {
  const host = document.getElementById('kpis');
  const checks = DATA.quality_checks.rows;
  const total = checks.find(c => c.check === 'Readable files');
  const cards = [
    {v: total.value.toLocaleString(), l: 'Images audited', cls: 'pass'},
    {v: '100%', l: 'Decodable', cls: 'pass'},
    {v: checks.find(c => c.check === 'Cross-source perceptual groups').value.toLocaleString(),
      l: 'Leakage groups', cls: 'risk'},
    {v: checks.find(c => c.check === 'Augmented rows quarantined').value.toLocaleString(),
      l: 'Augmented quarantined', cls: 'guarded'},
    {v: checks.find(c => c.check === 'Exact-duplicate rows').value.toLocaleString(),
      l: 'Exact-duplicate rows', cls: 'review'},
  ];
  host.innerHTML = cards.map(c =>
    `<div class="kpi ${c.cls}"><div class="v">${c.v}</div><div class="l">${c.l}</div></div>`).join('');
}

function classDist() {
  const rows = DATA.dataset_inventory.rows;
  const labels = [...new Set(rows.map(r => r.label))];
  const sets = [...new Set(rows.map(r => r.dataset))];
  const colors = {original: ACCENT, augmented: ACCENT2};
  const traces = sets.map(s => ({
    type: 'bar', name: s,
    x: labels,
    y: labels.map(l => (rows.find(r => r.label === l && r.dataset === s) || {}).count || 0),
    marker: {color: colors[s] || ACCENT2}
  }));
  Plotly.newPlot('classDist', traces, layout('Class counts by source collection'), config);
}

function splitDist() {
  const rows = DATA.split_distribution.rows;
  const splits = [...new Set(rows.map(r => r.split))];
  const sets = [...new Set(rows.map(r => r.dataset))];
  const colors = {original: ACCENT, augmented: ACCENT2};
  const traces = sets.map(s => ({
    type: 'bar', name: s,
    x: splits,
    y: splits.map(sp => rows.filter(r => r.split === sp && r.dataset === s)
      .reduce((a, b) => a + b.count, 0)),
    marker: {color: colors[s] || ACCENT2}
  }));
  const l = layout('Split composition after the leakage guard');
  l.barmode = 'stack';
  Plotly.newPlot('splitDist', traces, l, config);
}

function sharpness() {
  const rows = DATA.image_characteristics.rows;
  const labels = [...new Set(rows.map(r => r.label))];
  const traces = [
    {type: 'bar', name: 'original', x: labels,
      y: labels.map(l => (rows.find(r => r.label === l && r.dataset === 'original') || {}).sharpness_mean),
      marker: {color: ACCENT}},
    {type: 'bar', name: 'augmented', x: labels,
      y: labels.map(l => (rows.find(r => r.label === l && r.dataset === 'augmented') || {}).sharpness_mean),
      marker: {color: ACCENT2}}
  ];
  Plotly.newPlot('sharpness', traces, layout('Mean Laplacian sharpness by dataset/class'), config);
}

function quality() {
  const rows = DATA.quality_checks.rows;
  const colors = {pass: '#34d399', review: '#fbbf24', risk: '#f87171', guarded: '#38bdf8'};
  Plotly.newPlot('quality', [{
    type: 'bar', orientation: 'h',
    x: rows.map(r => r.value), y: rows.map(r => r.check),
    marker: {color: rows.map(r => colors[r.severity] || ACCENT)}
  }], Object.assign(layout('Integrity / duplicate checks'), {margin: {t: 30, r: 20, b: 40, l: 200}}), config);
}

function modelCompare() {
  const rows = DATA.model_comparison.rows;
  if (!rows.length) { document.getElementById('modelCompare').innerHTML =
    '<div class="note">No model metrics found.</div>'; return; }
  const metrics = ['accuracy', 'balanced_accuracy', 'macro_f1'];
  const traces = rows.map((r, i) => ({
    type: 'bar', name: r.model,
    x: ['Accuracy', 'Balanced acc.', 'Macro-F1'],
    y: metrics.map(m => r[m]),
    text: metrics.map(m => (r[m] * 100).toFixed(1) + '%'),
    textposition: 'outside',
    marker: {color: i === 0 ? ACCENT : (i === 1 ? ACCENT2 : '#a78bfa')}
  }));
  const l = layout('Held-out performance by model');
  l.barmode = 'group'; l.yaxis = {range: [0, 1], tickformat: '.0%'};
  Plotly.newPlot('modelCompare', traces, l, config);
}

function perClass() {
  const rows = DATA.class_metrics.rows.filter(r => ['precision', 'recall', 'f1-score'].includes(r.class) === false);
  const classes = DATA.class_metrics.rows
    .map(r => r['class']).filter(c => ['accuracy', 'macro avg', 'weighted avg', 'macro_precision'].includes(c) === false && c);
  const lookup = DATA.class_metrics.rows;
  const traces = ['precision', 'recall', 'f1-score'].map((m, i) => ({
    type: 'bar', name: m,
    x: classes,
    y: classes.map(c => (lookup.find(r => r['class'] === c) || {})[m]),
    marker: {color: [ACCENT, ACCENT2, '#a78bfa'][i]}
  }));
  const l = layout('Per-class precision / recall / F1 (HOG baseline)');
  l.barmode = 'group'; l.yaxis = {range: [0, 1], tickformat: '.0%'};
  Plotly.newPlot('perClass', traces, l, config);
}

function confusion() {
  const cm = DATA.confusion_matrix;
  if (!cm) { document.getElementById('confusion').innerHTML = '<div class="note">No confusion matrix.</div>'; return; }
  Plotly.newPlot('confusion', [{
    type: 'heatmap', z: cm.matrix, x: cm.labels, y: cm.labels,
    colorscale: 'Blues', showscale: true,
    hovertemplate: 'actual %{y}<br>predicted %{x}<br>count %{z}<extra></extra>'
  }], Object.assign(layout('Confusion matrix'), {xaxis: {title: 'Predicted'}, yaxis: {title: 'Actual', autorange: 'reversed'}}), config);
}

function search() {
  const rows = DATA.validation_search.rows.slice().sort((a, b) => a.alpha - b.alpha);
  Plotly.newPlot('search', [{
    type: 'bar', x: rows.map(r => r.alpha), y: rows.map(r => r.macro_f1),
    marker: {color: ACCENT}, text: rows.map(r => (r.macro_f1 * 100).toFixed(1) + '%'), textposition: 'outside'
  }], Object.assign(layout('Validation macro-F1 by alpha'), {yaxis: {range: [0, 1], tickformat: '.0%'}, xaxis: {title: 'alpha'}}), config);
}

function roc() {
  const roc = DATA.roc;
  const el = document.getElementById('roc');
  if (!roc) { el.innerHTML = '<div class="note">No ROC data.</div>'; return; }
  const colors = {MildDemented: '#4C72B0', ModerateDemented: '#C44E52', NonDemented: '#55A868', VeryMildDemented: '#CCB974'};
  const traces = Object.keys(roc).map(name => ({
    type: 'scatter', mode: 'lines', name: `${name} (AUC ${roc[name].auc.toFixed(3)})`,
    x: roc[name].fpr, y: roc[name].tpr, line: {color: colors[name], width: 2}
  }));
  traces.push({type: 'scatter', mode: 'lines', name: 'Chance', x: [0, 1], y: [0, 1],
    line: {color: '#64748b', dash: 'dash'}});
  const l = layout('One-vs-rest ROC (HOG baseline)');
  l.xaxis = {title: 'False-positive rate', range: [0, 1]};
  l.yaxis = {title: 'True-positive rate', range: [0, 1]};
  Plotly.newPlot('roc', traces, l, config);
}

function calibration() {
  const c = DATA.calibration;
  const el = document.getElementById('calibration');
  if (!c) { el.innerHTML = '<div class="note">No calibration data.</div>'; return; }
  const traces = [
    {type: 'scatter', mode: 'lines', name: 'Perfect', x: [0, 1], y: [0, 1],
      line: {color: '#64748b', dash: 'dash'}},
    {type: 'scatter', mode: 'lines+markers+text', name: 'Model', x: c.centers, y: c.accuracies,
      text: c.counts.map(String), textposition: 'top center',
      marker: {size: 8, color: ACCENT}, line: {color: ACCENT}}
  ];
  const l = layout('Reliability curve');
  l.xaxis = {title: 'Mean predicted confidence', range: [0, 1]};
  l.yaxis = {title: 'Observed accuracy', range: [0, 1]};
  Plotly.newPlot('calibration', traces, l, config);
}

function sharpnessHist() {
  const h = DATA.sharpness_hist;
  const el = document.getElementById('sharpnessHist');
  if (!h) { el.innerHTML = '<div class="note">No sharpness data.</div>'; return; }
  const centers = h.bins.slice(0, -1).map((b, i) => (b + h.bins[i + 1]) / 2);
  const traces = ['original', 'augmented'].map((s, i) => ({
    type: 'bar', name: s, x: centers, y: h.series[s],
    marker: {color: i === 0 ? ACCENT : ACCENT2}, opacity: 0.75
  }));
  const l = layout('Sharpness distribution by source');
  l.barmode = 'overlay';
  l.xaxis = {title: 'Laplacian variance'};
  l.yaxis = {title: 'Images'};
  Plotly.newPlot('sharpnessHist', traces, l, config);
}

function errors() {
  const rows = DATA.errors || [];
  const el = document.getElementById('errors');
  if (!rows.length) { el.innerHTML = '<div class="note">No errors.</div>'; return; }
  const header = '<tr style="text-align:left;color:#94a3b8"><th>Image</th><th>True</th><th>Predicted</th><th>Conf.</th></tr>';
  const body = rows.map(r => `<tr style="border-top:1px solid #334155">`
    + `<td style="font-size:11px">${r.path.split('/').slice(-2).join('/')}</td>`
    + `<td>${r.label}</td><td style="color:#f87171">${r.prediction}</td>`
    + `<td>${r.confidence.toFixed(2)}</td></tr>`).join('');
  el.innerHTML = `<table style="width:100%;border-collapse:collapse;font-size:12px">${header}${body}</table>`;
}

kpis(); classDist(); splitDist(); sharpness(); quality();
modelCompare(); perClass(); confusion(); search();
roc(); calibration(); sharpnessHist(); errors();
</script>
</body>
</html>
"""


def add_analytics(snapshot: dict) -> None:
    predictions_path = ROOT / "results/baseline/test_predictions.csv"
    if not predictions_path.exists():
        snapshot["confusion_matrix"] = None
        snapshot["roc"] = None
        snapshot["calibration"] = None
        snapshot["errors"] = []
        snapshot["sharpness_hist"] = None
        return

    predictions = pd.read_csv(predictions_path)
    labels = predictions["label"].to_numpy()

    matrix = pd.crosstab(predictions["label"], predictions["prediction"]).reindex(
        index=list(CLASS_NAMES), columns=list(CLASS_NAMES), fill_value=0
    )
    snapshot["confusion_matrix"] = {"labels": list(CLASS_NAMES), "matrix": matrix.values.tolist()}

    roc = {}
    for name in CLASS_NAMES:
        scores = predictions[f"probability_{name}"].to_numpy()
        binary = (labels == name).astype(int)
        fpr, tpr, _ = roc_curve(binary, scores)
        roc[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc(fpr, tpr))}
    snapshot["roc"] = roc

    confidence = predictions["confidence"].to_numpy()
    correct = predictions["prediction"].to_numpy() == labels
    edges = np.linspace(0, 1, 11)
    centers, accuracies, counts = [], [], []
    for lower, upper in zip(edges[:-1], edges[1:], strict=False):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            centers.append(float(confidence[mask].mean()))
            accuracies.append(float(correct[mask].mean()))
            counts.append(int(mask.sum()))
    snapshot["calibration"] = {"centers": centers, "accuracies": accuracies, "counts": counts}

    wrong = predictions[predictions["prediction"] != predictions["label"]]
    wrong = wrong.sort_values("confidence", ascending=False).head(12)
    snapshot["errors"] = [
        {
            "path": row["path"],
            "label": row["label"],
            "prediction": row["prediction"],
            "confidence": round(float(row["confidence"]), 3),
        }
        for _, row in wrong.iterrows()
    ]

    manifest_path = ROOT / "data/processed/image_manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path, usecols=["dataset", "sharpness"])
        bins = np.linspace(0, 1200, 25)
        histogram = {}
        for dataset in ("original", "augmented"):
            values = manifest.loc[manifest["dataset"] == dataset, "sharpness"].to_numpy()
            histogram[dataset] = np.histogram(values, bins=bins)[0].tolist()
        snapshot["sharpness_hist"] = {"bins": bins.tolist(), "series": histogram}


def main() -> None:
    snapshot = build_snapshot("dashboard")
    add_analytics(snapshot)
    snapshot["generatedAt"] = datetime.now(timezone.utc).isoformat()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    html = TEMPLATE.replace("__DATA__", json.dumps(snapshot)).replace(
        "__GENERATED__", datetime.now(timezone.utc).strftime("%Y-%m-%d")
    )
    OUTPUT.write_text(html, encoding="utf-8")

    data_path = ROOT / "dashboard" / "data.json"
    data_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(OUTPUT)
    print(data_path)


if __name__ == "__main__":
    main()
