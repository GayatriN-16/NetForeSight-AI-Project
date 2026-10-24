"""
NetForeSight AI - MITRE ATT&CK Schemas
Strictly typed data models for MITRE ATT&CK tactics, techniques, and supporting evidence.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class MitreTechnique(BaseModel):
    """Specific MITRE ATT&CK Technique with grounded evidence."""
    technique_id: str = Field(..., description="MITRE Technique ID (e.g., T1046, T1595, T1190, T1021)")
    technique_name: str = Field(..., description="Technique name (e.g., 'Network Service Discovery')")
    tactic_id: str = Field(..., description="Parent tactic ID (e.g., TA0007 Reconnaissance, TA0001 Initial Access)")
    tactic_name: str = Field(..., description="Parent tactic name")
    description: str = Field(..., description="Brief description of the technique in SOC context")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Evidence confidence score")
    evidence_triggers: List[str] = Field(default_factory=list, description="Exact network state metrics that triggered this mapping")
    mitigation_guidance: Optional[str] = Field(None, description="Recommended SOC defense action")


class MitreReport(BaseModel):
    """Aggregated MITRE ATT&CK assessment for current and forecasted network states."""
    active_mappings: List[MitreTechnique]
    forecasted_mappings: List[MitreTechnique]
    has_verified_evidence: bool = Field(True, description="False if mappings are illustrative demo or unverified")
