"""
NetForeSight AI - Model Evaluation & Benchmarking Router
Reports empirical benchmark comparisons between LSTM Temporal World Model and Logistic Regression Baseline.
STRICT RULE: Never fabricate or hardcode ML performance metrics.
If evaluation experiments have not been executed, transparently report 'Model evaluation pending.'
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, status
from pydantic import BaseModel, Field

router = APIRouter(prefix="/evaluation", tags=["Model Evaluation & Benchmarks"])


class ModelMetrics(BaseModel):
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    false_positive_rate: Optional[float] = None
    roc_auc: Optional[float] = None
    evaluation_dataset: Optional[str] = None
    sample_count: Optional[int] = None


class EvaluationReport(BaseModel):
    status: str = Field(..., description="Status of empirical model evaluation")
    temporal_lstm_world_model: ModelMetrics
    logistic_regression_baseline: ModelMetrics
    comparison_summary: str
    is_empirical: bool = Field(False, description="True only when calculated from real experimental test sets")


import json
import pathlib

METRICS_FILE = pathlib.Path(__file__).resolve().parents[3] / "ml" / "evaluation" / "metrics.json"


@router.get("/metrics", response_model=EvaluationReport, status_code=status.HTTP_200_OK)
async def get_evaluation_metrics():
    """
    Returns empirical evaluation metrics comparing the PyTorch LSTM Temporal World Model against the Baseline.
    Reads directly from ml/evaluation/metrics.json when experiments have completed.
    """
    if METRICS_FILE.exists():
        try:
            with open(METRICS_FILE, "r") as f:
                data = json.load(f)
            models = data.get("models", {})
            lstm = models.get("lstm_world_model", {})
            lr = models.get("logistic_regression_baseline", {})
            ds = data.get("dataset", {})

            return EvaluationReport(
                status="Empirical evaluation completed on held-out test split.",
                temporal_lstm_world_model=ModelMetrics(
                    precision=lstm.get("stage_precision"),
                    recall=lstm.get("stage_recall"),
                    f1_score=lstm.get("stage_f1"),
                    false_positive_rate=None,
                    roc_auc=None,
                    evaluation_dataset=f"{ds.get('path', 'CSE-CIC-IDS2018')} (Test Split)",
                    sample_count=ds.get("test_states")
                ),
                logistic_regression_baseline=ModelMetrics(
                    precision=lr.get("stage_precision"),
                    recall=lr.get("stage_recall"),
                    f1_score=lr.get("stage_f1"),
                    false_positive_rate=None,
                    roc_auc=None,
                    evaluation_dataset=f"{ds.get('path', 'CSE-CIC-IDS2018')} (Test Split)",
                    sample_count=ds.get("test_states")
                ),
                comparison_summary=(
                    f"LSTM Next-State MAE: {lstm.get('next_state_mae')} (vs Baseline {models.get('persistence_baseline', {}).get('next_state_mae')}, "
                    f"-{lstm.get('mae_reduction_vs_baseline_pct')}% error). "
                    f"LSTM Next-State RMSE: {lstm.get('next_state_rmse')} (vs Baseline {models.get('persistence_baseline', {}).get('next_state_rmse')}, "
                    f"-{lstm.get('rmse_reduction_vs_baseline_pct')}% error)."
                ),
                is_empirical=True
            )
        except Exception:
            pass

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
        comparison_summary="Empirical benchmarking pending model training.",
        is_empirical=False
    )
