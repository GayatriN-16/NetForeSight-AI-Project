"""
NetForeSight AI - Logistic Regression Baseline Model
Conventional static classification baseline operating on identical normalized 16-feature state vectors.
Used to benchmark the PyTorch LSTM Temporal World Model against standard static classifiers.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression

from backend.app.schemas.state import NetworkState
from ml.preprocessing.scaler import StateFeatureScaler
from ml.preprocessing.dataset import STAGE_TO_LABEL, LABEL_TO_STAGE


class LogisticRegressionBaseline:
    """
    Logistic Regression static classifier baseline.
    Classifies individual network states S(t) without temporal memory or state-transition world modeling.
    """

    def __init__(self, random_state: int = 42, max_iter: int = 1000):
        self.model = LogisticRegression(
            random_state=random_state, 
            max_iter=max_iter, 
            class_weight="balanced"
        )
        self.scaler = StateFeatureScaler()
        self.is_fitted: bool = False

    def fit(self, states: List[NetworkState], labels: List[str]) -> "LogisticRegressionBaseline":
        """Fits the logistic regression baseline on normalized network state feature vectors."""
        if not states or not labels:
            raise ValueError("States and labels cannot be empty.")

        self.scaler.fit(states)
        X = self.scaler.transform(states)
        y = np.array([STAGE_TO_LABEL.get(lbl, 0) for lbl in labels], dtype=np.int64)

        # Ensure at least two classes present to fit
        if len(np.unique(y)) > 1:
            self.model.fit(X, y)
            self.is_fitted = True
        return self

    def predict_state(self, state: NetworkState) -> str:
        """Predicts the attack stage for an individual NetworkState vector."""
        if not self.is_fitted:
            return "Normal"

        X = self.scaler.transform([state])
        y_pred = self.model.predict(X)[0]
        return LABEL_TO_STAGE.get(int(y_pred), "Unknown")

    def predict_proba(self, state: NetworkState) -> Dict[str, float]:
        """Returns the probability distribution across stages for an individual NetworkState."""
        if not self.is_fitted:
            return {"Normal": 1.0, "Suspicious": 0.0, "Reconnaissance": 0.0, "Initial Access": 0.0, "Lateral Movement": 0.0}

        X = self.scaler.transform([state])
        probs = self.model.predict_proba(X)[0]
        classes = self.model.classes_
        
        dist: Dict[str, float] = {}
        for cls_idx, p in zip(classes, probs):
            stage_name = LABEL_TO_STAGE.get(int(cls_idx), f"Class_{cls_idx}")
            dist[stage_name] = round(float(p), 4)
        return dist
