"""
Unit tests for Baseline Model and Evaluator
"""

from ml.models.baseline_lr import LogisticRegressionBaseline
from ml.evaluation.evaluator import ModelEvaluator
from tests.conftest import make_sample_states


def test_baseline_lr_fit_predict():
    states = make_sample_states(8)
    labels = ["Normal", "Normal", "Suspicious", "Suspicious", "Reconnaissance", "Reconnaissance", "Initial Access", "Initial Access"]

    baseline = LogisticRegressionBaseline()
    baseline.fit(states, labels)
    assert baseline.is_fitted is True

    pred_stage = baseline.predict_state(states[0])
    assert isinstance(pred_stage, str)

    probs = baseline.predict_proba(states[0])
    assert isinstance(probs, dict)
    assert len(probs) > 1
    assert abs(sum(probs.values()) - 1.0) < 0.05


def test_evaluator_pending_and_empirical():
    # Test 1: Pending status when uncomputed
    pending_rep = ModelEvaluator.get_pending_report()
    assert pending_rep.status == "Model evaluation pending."
    assert pending_rep.is_empirical is False
    assert pending_rep.temporal_lstm_world_model.precision is None

    # Test 2: Empirical calculation from ground truth vs predictions
    y_true = [0, 0, 1, 1, 2, 2, 3, 3]
    y_pred = [0, 0, 1, 0, 2, 2, 3, 3]  # 1 mistake

    metrics = ModelEvaluator.evaluate_predictions(y_true, y_pred, dataset_name="CSE-CIC-IDS2018-Sample")
    assert metrics.precision is not None
    assert metrics.recall is not None
    assert metrics.f1_score is not None
    assert metrics.sample_count == 8
    assert metrics.f1_score > 0.8
