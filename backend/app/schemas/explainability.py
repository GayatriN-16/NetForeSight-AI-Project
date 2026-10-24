"""
NetForeSight AI - Explainability Schemas
Data models for feature attribution and evidence grounding.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class FeatureContribution(BaseModel):
    """Represents the attribution of an individual state feature to the forecast risk."""
    feature_name: str = Field(..., description="Name of the network state feature (e.g., connection_fan_out, syn_ack_ratio)")
    display_name: str = Field(..., description="Human-readable SOC name for UI display")
    contribution_score: float = Field(..., description="Attribution value (positive increases risk, negative decreases risk)")
    direction: Literal["increases_risk", "decreases_risk", "neutral"] = Field("increases_risk")
    observed_value: float = Field(..., description="Actual observed metric value in the network state")
    baseline_reference: float = Field(..., description="Reference normal value for comparison")
    unit: str = Field("", description="Measurement unit (e.g., 'fps', 'ratio', 'ms', 'ports')")


class ExplainabilityReport(BaseModel):
    """Aggregated explainability breakdown for a forecast decision."""
    method: str = Field("gradient_attribution", description="Explainability method used (e.g. gradient_attribution, shap, perturbation)")
    top_contributions: List[FeatureContribution] = Field(default_factory=list)
    narrative_summary: str = Field(..., description="Clear textual rationale explaining the primary drivers of the prediction")
    is_model_derived: bool = Field(True, description="True if mathematically derived from model inference")
