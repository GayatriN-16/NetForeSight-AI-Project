"""
Unit tests for PyTorch LSTM Temporal World Model
"""

import pathlib
import sys
import torch
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.schemas.state import NetworkState
from ml.models.lstm_world_model import LSTMWorldModelNet, LSTMWorldModel
from ml.training.train_world_model import WorldModelTrainer
from ml.preprocessing.dataset import create_sliding_window_sequences, TemporalNetworkStateDataset
from torch.utils.data import DataLoader


def make_sample_states(n=10):
    states = []
    for i in range(n):
        states.append(NetworkState(
            window_index=i,
            timestamp_start=float(i * 5),
            timestamp_end=float((i + 1) * 5),
            window_duration=5.0,
            active_flows=10 + i * 2,
            unique_src_ips=2,
            unique_dst_ips=3 + (i % 3),
            unique_dst_ports=4 + i,
            connection_fan_out=float(1.5 + (i * 0.2)),
            syn_count=10 + i,
            ack_count=8,
            rst_count=i % 2,
            syn_ack_ratio=1.2,
            failed_connection_ratio=0.1,
            packet_rate=20.0 + i,
            byte_rate=5000.0,
            mean_packet_size=250.0,
            mean_iat=50.0,
            iat_variance=60.0,
            max_iat=100.0,
            port_scan_score=min(1.0, 0.1 * i)
        ))
    return states


def test_lstm_world_model_net_shapes():
    net = LSTMWorldModelNet(input_dim=16, hidden_dim=32, num_layers=1, num_classes=5)
    dummy_input = torch.randn(4, 3, 16)  # (batch=4, seq_len=3, dim=16)
    
    next_state, stage_logits, latent_z = net(dummy_input)
    assert next_state.shape == torch.Size([4, 16])
    assert stage_logits.shape == torch.Size([4, 5])
    assert latent_z.shape == torch.Size([4, 32])


def test_world_model_predict_and_forecast():
    model = LSTMWorldModel(input_dim=16, hidden_dim=32, num_layers=1, window_length=3)
    states = make_sample_states(6)

    # 1. Predict next state
    next_s = model.predict_next_state(states)
    assert isinstance(next_s, NetworkState)
    assert next_s.window_index == states[-1].window_index + 1
    assert next_s.window_duration == 5.0

    # 2. K-step forecast (K=3)
    forecast_resp = model.forecast(states, k_steps=3, window_size=5.0)
    assert forecast_resp.forecast_horizon == 3
    assert len(forecast_resp.forecast_steps) == 3
    assert forecast_resp.forecast_steps[0].forecasted_probability >= 0.0
    assert forecast_resp.forecast_steps[0].forecasted_probability <= 1.0
    assert forecast_resp.is_demo_mode is False

    # 3. Explainability
    explain_rep = model.explain(states, forecast_resp)
    assert explain_rep.is_model_derived is True
    assert len(explain_rep.top_contributions) > 0


def test_world_model_trainer():
    model = LSTMWorldModel(input_dim=16, hidden_dim=32, num_layers=1, window_length=3)
    states = make_sample_states(10)
    
    X_seqs, y_states, y_stages = create_sliding_window_sequences(states, window_length=3)
    ds = TemporalNetworkStateDataset(X_seqs, y_states, y_stages)
    dl = DataLoader(ds, batch_size=2, shuffle=True)

    trainer = WorldModelTrainer(model, lr=0.01)
    train_metrics = trainer.train_epoch(dl)
    
    assert "loss" in train_metrics
    assert "state_mse" in train_metrics
    assert "stage_accuracy" in train_metrics
    assert train_metrics["loss"] >= 0.0
