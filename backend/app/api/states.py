"""
NetForeSight AI - Network State Builder Router
Transforms raw flow series into windowed Network State vectors S(t).
"""

from typing import List
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from backend.app.schemas.flow import FlowRecord
from backend.app.schemas.state import NetworkStateSeries
from backend.app.services.state_builder import NetworkStateBuilder

router = APIRouter(prefix="/states", tags=["Network State Modeling"])


class BuildStatesRequest(BaseModel):
    flows: List[FlowRecord] = Field(..., min_length=1, description="List of chronologically ordered flow records")
    window_size_seconds: float = Field(5.0, gt=0.1, le=60.0, description="Temporal window duration in seconds (default 5.0s)")


@router.post("/build", response_model=NetworkStateSeries, status_code=status.HTTP_200_OK)
async def build_network_states(payload: BuildStatesRequest):
    """
    Groups flow records into contiguous time windows of width `window_size_seconds`
    and computes the topological, protocol, timing, and port-scan features for every S(t).
    """
    if not payload.flows:
        raise HTTPException(status_code=400, detail="Flows list cannot be empty.")

    try:
        series = NetworkStateBuilder.build_states(
            flows=payload.flows,
            window_size_seconds=payload.window_size_seconds,
            is_demo=False
        )
        return series
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate network states: {str(e)}")
