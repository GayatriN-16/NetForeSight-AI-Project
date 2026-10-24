"""
NetForeSight AI - Abstract WorldModel Interface
Defines the formal contracts for state-transition world models, k-step forecasting, and explainability.
"""

from abc import ABC, abstractmethod
from typing import List

from backend.app.schemas.state import NetworkState
from backend.app.schemas.forecast import ForecastResponse
from backend.app.schemas.explainability import ExplainabilityReport


class BaseWorldModel(ABC):
    """
    Abstract Base Class for Network-State World Models.
    Ensures backend services and frontend remain completely decoupled from underlying PyTorch / Scikit-Learn backends.
    """

    @abstractmethod
    def predict_next_state(self, history: List[NetworkState]) -> NetworkState:
        """
        Models temporal state transition P(S(t+1) | S(t), S(t-1), ...).
        Given historical states, predicts the immediate next state vector S(t+1).
        """
        pass

    @abstractmethod
    def forecast(self, history: List[NetworkState], k_steps: int = 3, window_size: float = 5.0) -> ForecastResponse:
        """
        Forecasts K future network states [S(t+1), ..., S(t+K)] and computes forecasted attack progression probabilities.
        """
        pass

    @abstractmethod
    def explain(self, history: List[NetworkState], forecast: ForecastResponse) -> ExplainabilityReport:
        """
        Derives mathematical feature attributions and explainability evidence for the forecast trajectory.
        """
        pass
