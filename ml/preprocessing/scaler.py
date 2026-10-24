"""
NetForeSight AI - Network State Feature Normalizer / Scaler
Handles standardization and scaling of 16-dimensional Network State feature vectors S(t).
"""

from typing import List, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler

from backend.app.schemas.state import NetworkState

FEATURE_NAMES: List[str] = [
    "active_flows",
    "unique_src_ips",
    "unique_dst_ips",
    "unique_dst_ports",
    "connection_fan_out",
    "syn_count",
    "ack_count",
    "rst_count",
    "syn_ack_ratio",
    "failed_connection_ratio",
    "packet_rate",
    "byte_rate",
    "mean_packet_size",
    "mean_iat",
    "iat_variance",
    "port_scan_score"
]


class StateFeatureScaler:
    """Standardizes 16-element network state feature vectors for ML model ingestion."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted: bool = False
        self.feature_names: List[str] = FEATURE_NAMES

    def extract_features(self, state: NetworkState) -> np.ndarray:
        """Extracts 1D numpy array of 16 numerical features from a NetworkState object."""
        if state.raw_feature_vector and len(state.raw_feature_vector) == len(FEATURE_NAMES):
            return np.array(state.raw_feature_vector, dtype=np.float32)
        
        return np.array([
            float(state.active_flows),
            float(state.unique_src_ips),
            float(state.unique_dst_ips),
            float(state.unique_dst_ports),
            float(state.connection_fan_out),
            float(state.syn_count),
            float(state.ack_count),
            float(state.rst_count),
            float(state.syn_ack_ratio),
            float(state.failed_connection_ratio),
            float(state.packet_rate),
            float(state.byte_rate),
            float(state.mean_packet_size),
            float(state.mean_iat),
            float(state.iat_variance),
            float(state.port_scan_score)
        ], dtype=np.float32)

    def extract_matrix(self, states: List[NetworkState]) -> np.ndarray:
        """Extracts (N, 16) 2D feature matrix from a sequence of NetworkState objects."""
        if not states:
            return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        return np.vstack([self.extract_features(s) for s in states])

    def fit(self, states: List[NetworkState]) -> "StateFeatureScaler":
        """Fits the StandardScaler on a series of network state vectors."""
        X = self.extract_matrix(states)
        if len(X) > 0:
            self.scaler.fit(X)
            self.is_fitted = True
        return self

    def transform(self, states: List[NetworkState]) -> np.ndarray:
        """Transforms network states into standardized numerical arrays."""
        X = self.extract_matrix(states)
        if not self.is_fitted:
            # Fallback to robust log1p / minmax normalization if not yet fitted
            return np.log1p(np.maximum(0.0, X)).astype(np.float32)
        return self.scaler.transform(X).astype(np.float32)

    def inverse_transform(self, X_norm: np.ndarray) -> np.ndarray:
        """Reconstructs original metric scale from standardized feature array."""
        if not self.is_fitted:
            return np.expm1(np.maximum(0.0, X_norm)).astype(np.float32)
        return self.scaler.inverse_transform(X_norm).astype(np.float32)
