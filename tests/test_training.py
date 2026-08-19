import numpy as np

from app.ml.train import _evaluate, select_f1_threshold


def test_validation_threshold_maximizes_illicit_f1():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.10, 0.40, 0.45, 0.90])

    threshold = select_f1_threshold(labels, probabilities)
    metrics = _evaluate(labels, probabilities, threshold)

    assert threshold == 0.45
    assert metrics["illicit_f1"] == 1.0


def test_selected_threshold_is_bounded_for_serving():
    labels = np.array([0, 1])
    probabilities = np.array([0.0, 1.0])

    assert 0 < select_f1_threshold(labels, probabilities) < 1
