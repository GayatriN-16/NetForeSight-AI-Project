"""
NetForeSight AI - PyTorch LSTM World Model Training Harness
Provides training, validation, multi-task loss computation (State Regression MSE + Stage CrossEntropy), and checkpointing.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ml.models.lstm_world_model import LSTMWorldModelNet, LSTMWorldModel
from ml.preprocessing.dataset import TemporalNetworkStateDataset


class WorldModelTrainer:
    """Orchestrates multi-task training for the LSTM Temporal World Model."""

    def __init__(
        self,
        model: LSTMWorldModel,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        state_loss_weight: float = 1.0,
        stage_loss_weight: float = 0.5
    ):
        self.model = model
        self.net = model.net
        self.lr = lr
        self.state_loss_weight = state_loss_weight
        self.stage_loss_weight = stage_loss_weight
        
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        self.state_criterion = nn.MSELoss()
        self.stage_criterion = nn.CrossEntropyLoss()

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Runs a single training epoch across batches."""
        self.net.train()
        total_loss = 0.0
        total_state_loss = 0.0
        total_stage_loss = 0.0
        correct_stages = 0
        total_samples = 0

        for X_batch, y_state_batch, y_stage_batch in dataloader:
            self.optimizer.zero_grad()
            
            pred_state, pred_stage, _ = self.net(X_batch)
            
            loss_state = self.state_criterion(pred_state, y_state_batch)
            loss_stage = self.stage_criterion(pred_stage, y_stage_batch)
            
            loss = (self.state_loss_weight * loss_state) + (self.stage_loss_weight * loss_stage)
            loss.backward()
            
            # Gradient clipping for LSTM stability
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.optimizer.step()

            batch_size = X_batch.size(0)
            total_loss += loss.item() * batch_size
            total_state_loss += loss_state.item() * batch_size
            total_stage_loss += loss_stage.item() * batch_size
            
            preds = torch.argmax(pred_stage, dim=1)
            correct_stages += (preds == y_stage_batch).sum().item()
            total_samples += batch_size

        n = max(1, total_samples)
        return {
            "loss": total_loss / n,
            "state_mse": total_state_loss / n,
            "stage_loss": total_stage_loss / n,
            "stage_accuracy": correct_stages / n
        }

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluates model performance on validation / test dataloader."""
        self.net.eval()
        total_loss = 0.0
        total_state_loss = 0.0
        correct_stages = 0
        total_samples = 0

        with torch.no_grad():
            for X_batch, y_state_batch, y_stage_batch in dataloader:
                pred_state, pred_stage, _ = self.net(X_batch)
                
                loss_state = self.state_criterion(pred_state, y_state_batch)
                loss_stage = self.stage_criterion(pred_stage, y_stage_batch)
                loss = (self.state_loss_weight * loss_state) + (self.stage_loss_weight * loss_stage)

                batch_size = X_batch.size(0)
                total_loss += loss.item() * batch_size
                total_state_loss += loss_state.item() * batch_size
                
                preds = torch.argmax(pred_stage, dim=1)
                correct_stages += (preds == y_stage_batch).sum().item()
                total_samples += batch_size

        n = max(1, total_samples)
        return {
            "val_loss": total_loss / n,
            "val_state_mse": total_state_loss / n,
            "val_stage_accuracy": correct_stages / n
        }
