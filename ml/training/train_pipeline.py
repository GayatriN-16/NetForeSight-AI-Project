"""
NetForeSight AI - End-to-End Model Training & Empirical Evaluation Pipeline
===========================================================================
1. Ingests raw CSE-CIC-IDS2018 flows via CSVParserService.
2. Converts flows into 5-second NetworkState vectors S(t) and aligned labels via NetworkStateBuilder.
3. Chronologically splits states into Train (70%), Val (15%), Test (15%) BEFORE scaler fitting.
4. Fits StateFeatureScaler strictly on Train states (zero leakage).
5. Generates sliding-window sequences for next-state regression S(t+1) and next-stage classification.
6. Trains:
   - Baseline 1: Persistence / Last-State Baseline S(t+1) = S(t)
   - Baseline 2: Static Logistic Regression Classifier
   - Intended Model: PyTorch LSTM Temporal World Model (Multi-Task: Regression + Classification)
7. Evaluates on held-out chronological test data and computes actual MAE, RMSE, Accuracy, F1.
8. Saves trained weights, scaler params, and evaluation metrics.
"""

import os
import sys
import json
import time
import pathlib
from typing import Dict, Any, Tuple
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from backend.app.services.parser_csv import CSVParserService
from backend.app.services.state_builder import NetworkStateBuilder
from ml.preprocessing.scaler import StateFeatureScaler, FEATURE_NAMES
from ml.preprocessing.dataset import (
    prepare_chronological_datasets,
    TemporalNetworkStateDataset,
    STAGE_TO_LABEL,
    LABEL_TO_STAGE
)
from ml.models.lstm_world_model import LSTMWorldModelNet


def run_training_pipeline(
    csv_path: str = "data/sample/cse_cic_ids2018_infiltration_scenario.csv",
    window_length: int = 4,
    hidden_dim: int = 64,
    num_layers: int = 2,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 0.001,
    output_dir: str = "ml/weights",
    metrics_path: str = "ml/evaluation/metrics.json"
) -> Dict[str, Any]:
    print(f"=== NetForeSight AI: Training Pipeline ===")
    print(f"Dataset path: {csv_path}")
    print(f"Parameters: window_length={window_length}, hidden_dim={hidden_dim}, epochs={epochs}")

    # -------------------------------------------------------------------------
    # 1. Flow Ingestion via CSVParserService
    # -------------------------------------------------------------------------
    full_csv_path = PROJECT_ROOT / csv_path
    if not full_csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {full_csv_path}")

    print(f"\n[1/6] Ingesting CSV flows via CSVParserService...")
    t0 = time.time()
    with open(full_csv_path, "rb") as f:
        csv_bytes = f.read()

    flows, flow_summary = CSVParserService.parse_csv_content(csv_bytes, filename=full_csv_path.name)
    print(f"  Parsed {len(flows)} flows in {time.time() - t0:.2f}s")
    print(f"  Time span: {flow_summary.time_span_seconds:.1f} seconds")

    # -------------------------------------------------------------------------
    # 2. Network State Windowing via NetworkStateBuilder
    # -------------------------------------------------------------------------
    print(f"\n[2/6] Building 5.0s NetworkState vectors S(t) via NetworkStateBuilder...")
    t1 = time.time()
    states, labels = NetworkStateBuilder.build_states_with_labels(flows, window_size_seconds=5.0)
    print(f"  Created {len(states)} temporal states in {time.time() - t1:.2f}s")
    from collections import Counter
    label_counts = Counter(labels)
    print(f"  Window label distribution: {dict(label_counts)}")

    # -------------------------------------------------------------------------
    # 3. Chronological Split & Scaler Fitting (Zero Leakage)
    # -------------------------------------------------------------------------
    print(f"\n[3/6] Chronological splitting & train-only scaling (Zero Leakage)...")
    (X_tr, yn_tr, ys_tr), (X_v, yn_v, ys_v), (X_te, yn_te, ys_te), scaler = (
        prepare_chronological_datasets(
            states, labels,
            window_length=window_length,
            train_frac=0.70,
            val_frac=0.15
        )
    )

    print(f"  Train set:      X={X_tr.shape}, y_next={yn_tr.shape}, y_stage={ys_tr.shape}")
    print(f"  Validation set: X={X_v.shape},  y_next={yn_v.shape},  y_stage={ys_v.shape}")
    print(f"  Test set:       X={X_te.shape}, y_next={yn_te.shape}, y_stage={ys_te.shape}")

    # -------------------------------------------------------------------------
    # 4. Baseline Evaluation on Held-Out Test Set
    # -------------------------------------------------------------------------
    print(f"\n[4/6] Evaluating Baselines on Held-Out Test Set...")
    # Baseline 1: Persistence / Last-Observed-State Baseline for regression: S(t+1) = S(t)
    # S(t) is the last step in the input sequence X[:, -1, :]
    baseline_pred_next_state = X_te[:, -1, :]
    
    base_mae = float(np.mean(np.abs(yn_te - baseline_pred_next_state)))
    base_rmse = float(np.sqrt(np.mean((yn_te - baseline_pred_next_state) ** 2)))
    print(f"  [Baseline 1 - Persistence] Next-State MAE:  {base_mae:.4f}")
    print(f"  [Baseline 1 - Persistence] Next-State RMSE: {base_rmse:.4f}")

    # Baseline 2: Static Logistic Regression for attack stage classification
    # Train on last observed step of train input sequences
    X_tr_flat = X_tr[:, -1, :]
    X_te_flat = X_te[:, -1, :]
    
    if len(np.unique(ys_tr)) > 1:
        lr_baseline = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
        lr_baseline.fit(X_tr_flat, ys_tr)
        lr_stage_preds = lr_baseline.predict(X_te_flat)
    else:
        # Majority class baseline if train set contains only one class
        majority_class = int(ys_tr[0])
        lr_stage_preds = np.full_like(ys_te, majority_class)

    lr_acc = float(accuracy_score(ys_te, lr_stage_preds))
    p_lr, r_lr, f1_lr, _ = precision_recall_fscore_support(ys_te, lr_stage_preds, average="weighted", zero_division=0)
    print(f"  [Baseline 2 - Logistic Regression] Stage Accuracy: {lr_acc * 100:.2f}% | F1: {f1_lr:.4f}")

    # -------------------------------------------------------------------------
    # 5. Train PyTorch LSTM Temporal World Model
    # -------------------------------------------------------------------------
    print(f"\n[5/6] Training PyTorch LSTM Temporal World Model...")
    train_dataset = TemporalNetworkStateDataset(X_tr, yn_tr, ys_tr)
    val_dataset   = TemporalNetworkStateDataset(X_v, yn_v, ys_v)
    test_dataset  = TemporalNetworkStateDataset(X_te, yn_te, ys_te)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_classes = 5
    net = LSTMWorldModelNet(
        input_dim=16,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes
    )

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    state_criterion = nn.MSELoss()
    stage_criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(1, epochs + 1):
        net.train()
        train_loss = 0.0
        train_mse = 0.0

        for batch_X, batch_yn, batch_ys in train_loader:
            optimizer.zero_grad()
            pred_next, pred_stage, _ = net(batch_X)

            loss_state = state_criterion(pred_next, batch_yn)
            loss_stage = stage_criterion(pred_stage, batch_ys)
            total_loss = loss_state + 0.5 * loss_stage

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += total_loss.item() * batch_X.size(0)
            train_mse += loss_state.item() * batch_X.size(0)

        train_loss /= len(train_dataset)
        train_mse /= len(train_dataset)

        # Validation evaluation
        net.eval()
        val_loss = 0.0
        val_mse = 0.0
        with torch.no_grad():
            for batch_X, batch_yn, batch_ys in val_loader:
                pred_next, pred_stage, _ = net(batch_X)
                loss_state = state_criterion(pred_next, batch_yn)
                loss_stage = stage_criterion(pred_stage, batch_ys)
                loss = loss_state + 0.5 * loss_stage
                val_loss += loss.item() * batch_X.size(0)
                val_mse += loss_state.item() * batch_X.size(0)

        val_loss /= len(val_dataset)
        val_mse /= len(val_dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in net.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.4f} (MSE: {train_mse:.4f}) | Val Loss: {val_loss:.4f} (MSE: {val_mse:.4f})")

    # Load best validation checkpoint
    if best_weights is not None:
        net.load_state_dict(best_weights)

    # -------------------------------------------------------------------------
    # 6. Final Empirical Evaluation on Held-Out Test Set
    # -------------------------------------------------------------------------
    print(f"\n[6/6] Computing Final Test Metrics for LSTM World Model...")
    net.eval()
    all_pred_states = []
    all_pred_stages = []

    with torch.no_grad():
        for batch_X, _, _ in test_loader:
            pred_next, pred_stage, _ = net(batch_X)
            all_pred_states.append(pred_next.cpu().numpy())
            all_pred_stages.append(torch.argmax(pred_stage, dim=1).cpu().numpy())

    pred_states_arr = np.vstack(all_pred_states)
    pred_stages_arr = np.concatenate(all_pred_stages)

    # State Forecasting Metrics (Continuous Next-State Vector S(t+1))
    lstm_mae = float(np.mean(np.abs(yn_te - pred_states_arr)))
    lstm_rmse = float(np.sqrt(np.mean((yn_te - pred_states_arr) ** 2)))

    # Per-feature MAE and RMSE
    per_feat_mae = {feat: float(m) for feat, m in zip(FEATURE_NAMES, np.mean(np.abs(yn_te - pred_states_arr), axis=0))}
    per_feat_rmse = {feat: float(r) for feat, r in zip(FEATURE_NAMES, np.sqrt(np.mean((yn_te - pred_states_arr) ** 2, axis=0)))}

    # Attack Stage Forecasting Metrics
    lstm_acc = float(accuracy_score(ys_te, pred_stages_arr))
    p_lstm, r_lstm, f1_lstm, _ = precision_recall_fscore_support(ys_te, pred_stages_arr, average="weighted", zero_division=0)

    # Checkpoint output
    out_dir_path = PROJECT_ROOT / output_dir
    out_dir_path.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir_path / "lstm_world_model.pt"
    torch.save(net.state_dict(), weights_path)

    # Compile Evaluation Results
    results = {
        "status": "Completed",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {
            "path": csv_path,
            "total_flows_parsed": len(flows),
            "total_states_built": len(states),
            "state_labels": dict(label_counts),
            "train_states": len(X_tr),
            "val_states": len(X_v),
            "test_states": len(X_te),
            "window_length": window_length,
            "feature_dim": 16,
            "leakage_audit": "Zero leakage - Chronological split performed before scaler fit; disjoint state partitions."
        },
        "models": {
            "persistence_baseline": {
                "description": "Last-Observed-State Persistence Baseline: S(t+1) = S(t)",
                "next_state_mae": round(base_mae, 4),
                "next_state_rmse": round(base_rmse, 4)
            },
            "logistic_regression_baseline": {
                "description": "Static Logistic Regression on S(t) with class_weight='balanced'",
                "stage_accuracy": round(lr_acc, 4),
                "stage_precision": round(float(p_lr), 4),
                "stage_recall": round(float(r_lr), 4),
                "stage_f1": round(float(f1_lr), 4)
            },
            "lstm_world_model": {
                "description": "Multi-Task PyTorch LSTM Temporal World Model (2 layers, hidden=64)",
                "weights_saved_to": str(weights_path.relative_to(PROJECT_ROOT)),
                "next_state_mae": round(lstm_mae, 4),
                "next_state_rmse": round(lstm_rmse, 4),
                "stage_accuracy": round(lstm_acc, 4),
                "stage_precision": round(float(p_lstm), 4),
                "stage_recall": round(float(r_lstm), 4),
                "stage_f1": round(float(f1_lstm), 4),
                "mae_reduction_vs_baseline_pct": round(((base_mae - lstm_mae) / base_mae) * 100.0, 2) if base_mae > 0 else 0.0,
                "rmse_reduction_vs_baseline_pct": round(((base_rmse - lstm_rmse) / base_rmse) * 100.0, 2) if base_rmse > 0 else 0.0,
                "per_feature_mae": per_feat_mae,
                "per_feature_rmse": per_feat_rmse
            }
        }
    }

    # Save metrics JSON
    metrics_file = PROJECT_ROOT / metrics_path
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("FINAL EMPIRICAL RESULTS (HELD-OUT CHRONOLOGICAL TEST SET)")
    print("=" * 60)
    print(f"State Forecasting MAE:  Baseline = {base_mae:.4f}  |  LSTM = {lstm_mae:.4f} (Delta: {results['models']['lstm_world_model']['mae_reduction_vs_baseline_pct']:+.2f}%)")
    print(f"State Forecasting RMSE: Baseline = {base_rmse:.4f}  |  LSTM = {lstm_rmse:.4f} (Delta: {results['models']['lstm_world_model']['rmse_reduction_vs_baseline_pct']:+.2f}%)")
    print(f"Stage Classification:   LR Base  = {lr_acc*100:.2f}% (F1={f1_lr:.4f}) | LSTM = {lstm_acc*100:.2f}% (F1={f1_lstm:.4f})")
    print(f"Weights saved to:       {weights_path}")
    print(f"Metrics saved to:       {metrics_file}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_training_pipeline()
