import numpy as np

from alzheimer_mri.metrics import bootstrap_interval, classification_metrics

CLASS_NAMES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]


def test_perfect_predictions_give_unit_scores():
    labels = np.array([0, 1, 2, 3] * 5)
    probabilities = np.eye(4)[labels]
    metrics = classification_metrics(labels, probabilities, CLASS_NAMES)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["expected_calibration_error"] < 1e-9
    assert metrics["confusion_matrix"] == [
        [5, 0, 0, 0],
        [0, 5, 0, 0],
        [0, 0, 5, 0],
        [0, 0, 0, 5],
    ]


def test_metrics_contains_expected_keys():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, 40)
    probabilities = rng.dirichlet(np.ones(4), size=40)
    metrics = classification_metrics(labels, probabilities, CLASS_NAMES)
    for key in ("accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1"):
        assert key in metrics
    assert len(metrics["confusion_matrix"]) == 4


def test_bootstrap_interval_brackets_point_estimate():
    rng = np.random.default_rng(42)
    labels = rng.integers(0, 2, 200)
    predictions = rng.integers(0, 2, 200)
    low, high = bootstrap_interval(
        labels, predictions, lambda a, b: float((a == b).mean()), iterations=200
    )
    point = float((labels == predictions).mean())
    assert low <= point <= high
