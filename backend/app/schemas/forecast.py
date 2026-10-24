"""
NetForeSight AI - Forecast Schemas
Contract definitions for K-step future state forecasting and attack progression prediction.
"""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from backend.app.schemas.state import NetworkState
from backend.app.schemas.explainability import ExplainabilityReport
from backend.app.schemas.mitre import MitreReport


AttackStage = Literal[
    "Normal",
    "Suspicious",
    "Reconnaissance",
    "Initial Access",
    "Lateral Movement",
    "Exfiltration",
    "Unknown"
]


class ForecastRequest(BaseModel):
    """Request payload for state-transition and attack-progression forecasting."""
    history_states: List[NetworkState] = Field(
        ..., 
        min_length=1, 
        description="Temporal sequence of historical network states [S(t-W), ..., S(t)]"
    )
    window_size: float = Field(5.0, gt=0.0, description="Time window in seconds used for states")
    forecast_steps: int = Field(3, ge=1, le=10, description="Number of future windows K to forecast (default K=3)")
    scenario_id: Optional[str] = Field(None, description="Optional scenario identifier if executing demo replay")


class ForecastStep(BaseModel):
    """Represents a single future step S(t+k) in the forecast trajectory."""
    step_index: int = Field(..., description="Step offset k (e.g. 1 for S(t+1), 2 for S(t+2), 3 for S(t+3))")
    time_offset_seconds: float = Field(..., description="Future time offset in seconds from S(t)")
    forecasted_probability: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Forecasted probability of malicious attack progression at this future window"
    )
    predicted_stage: AttackStage = Field(..., description="Predicted cyber kill-chain progression stage")
    confidence_interval: Dict[str, float] = Field(
        default_factory=dict, 
        description="Lower and upper bounds for forecasted probability (e.g., {'lower': 0.72, 'upper': 0.88})"
    )
    predicted_state: Optional[NetworkState] = Field(None, description="Forecasted network state vector S(t+k)")


class ForecastResponse(BaseModel):
    """Complete forecast response payload."""
    forecast_horizon: int = Field(..., description="Total forecasted steps K")
    window_size: float = Field(..., description="Window duration in seconds")
    current_stage: AttackStage = Field(..., description="Estimated stage of the most recent observed state S(t)")
    current_infiltration_probability: float = Field(..., ge=0.0, le=1.0, description="Current infiltration/attack probability")
    forecast_steps: List[ForecastStep] = Field(..., description="Array of K future predicted steps")
    mitre_report: MitreReport = Field(..., description="Associated MITRE ATT&CK technique mappings")
    explainability: ExplainabilityReport = Field(..., description="Feature attribution report for the forecast trajectory")
    is_demo_mode: bool = Field(False, description="Flag indicating if output is from preprocessed/demo scenario")
    model_version: str = Field("WorldModel-LSTM-v1.0", description="Identifier of the inference backend used")
