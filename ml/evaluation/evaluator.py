"""
NetForeSight AI - Empirical Model Evaluator
Computes mathematical evaluation metrics (Precision, Recall, F1, FPR, ROC-AUC) strictly from empirical test sets.
STRICT RULE: Never fabricate or hardcode performance metrics.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix

from backend.app.schemas.state import NetworkState
from backend.app.api.evaluation import EvaluationReport, ModelMetrics
from ml.models.lstm_world_model import LSTMWorldModel
from ml.models.baseline_lr import LogisticRegressionBaseline
from ml.preprocessing.dataset import STAGE_TO_LABEL


class ModelEvaluator:
    """Computes empirical validation metrics on real test splits without fabrication."""

    @staticmethod
    def get_pending_report(reason: str = "Model training and benchmark runs pending.") -> EvaluationReport:
        """Returns honest uncomputed report when evaluation experiments have not yet completed."""
        return EvaluationReport(
            status="Model evaluation pending.",
            temporal_lstm_world_model=ModelMetrics(
                precision=None,
                recall=None,
                f1_score=None,
                false_positive_rate=None,
                roc_auc=None,
                evaluation_dataset="CSE-CIC-IDS2018 (Pending training)",
                sample_count=None
            ),
            logistic_regression_baseline=ModelMetrics(
                precision=None,
                recall=None,
                f1_score=None,
                false_positive_rate=None,
                roc_auc=None,
                evaluation_dataset="CSE-CIC-IDS2018 (Pending training)",
                sample_count=None
            ),
            comparison_summary=reason,
            is_empirical=False
        )

    @staticmethod
    def evaluate_predictions(
        y_true: List[int],
        y_pred: List[int],
        y_scores: Optional[np.ndarray] = None,
        dataset_name: str = "Test Set"
    ) -> ModelMetrics:
        """Calculates exact empirical performance metrics from ground truth and predicted labels."""
        if len(y_true) == 0 or len(y_pred) == 0:
            return ModelMetrics(evaluation_dataset=dataset_name, sample_count=0)

        # Precision, Recall, F1
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, 
            y_pred, 
            average="weighted", 
            zero_division=0
        )

        # False Positive Rate from confusion matrix (binary/normal vs malicious)
        y_true_binary = [1 if y > 0 else 0 for y in y_true]
        y_pred_binary = [1 if y > 0 else 0 for y in y_pred]
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).ravel()
        fpr = float(fp) / float(max(1, fp + tn))

        # ROC-AUC if probability scores are provided
        roc_auc_val: Optional[float] = None
        if y_scores is not None and len(np.unique(y_true)) > 1:
            try:
                if y_scores.ndim == 1 or y_scores.shape[1] == 2:
                    roc_auc_val = float(roc_auc_score(y_true_binary, y_scores))
                else:
                    roc_auc_val = float(roc_auc_score(y_true, y_scores, multi_class="ovr", average="weighted"))
            except Exception:
                roc_auc_val = None

        return ModelMetrics(
            precision=round(float(p), 4),
            recall=round(float(r), 4),
            f1_score=round(float(f1), 4),
            false_positive_rate=round(float(fpr), 4),
            roc_auc=round(roc_auc_val, 4) if roc_auc_val is not None else None,
            evaluation_dataset=dataset_name,
            sample_count=len(y_true)
        )
