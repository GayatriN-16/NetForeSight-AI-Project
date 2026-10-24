"""
NetForeSight AI — Temporal Network State Dataset & Sequence Generator
======================================================================
Transforms continuous state time-series into sliding-window sequences for
LSTM World Model training.

Changes vs. original
--------------------
* Labels are mapped via label_mapper.map_label_to_int() — NO silent fallback
  to 0 for unknown labels.  Unknown labels raise UnknownLabelError immediately.
* Temporal window labels are assigned by the label of the TARGET window
  (S(t+1)), not the input window, keeping input/target alignment correct.
* A chronological_split() helper enforces temporal ordering and prevents
  train/val/test leakage.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object

from backend.app.schemas.state import NetworkState
from ml.preprocessing.scaler import StateFeatureScaler
from ml.preprocessing.label_mapper import map_label_to_int, UnknownLabelError


# ---------------------------------------------------------------------------
# Stage integer ↔ name (kept for backward-compatibility with LSTM head)
# ---------------------------------------------------------------------------
STAGE_TO_LABEL: Dict[str, int] = {
    "Normal":           0,
    "Suspicious":       1,
    "Reconnaissance":   2,
    "Initial Access":   3,
    "Lateral Movement": 4,
}

LABEL_TO_STAGE: Dict[int, str] = {v: k for k, v in STAGE_TO_LABEL.items()}


# ---------------------------------------------------------------------------
# Core sequence builder
# ---------------------------------------------------------------------------

def create_sliding_window_sequences(
    states: List[NetworkState],
    labels: Optional[List[str]] = None,
    window_length: int = 4,
    scaler: Optional[StateFeatureScaler] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transforms state sequence [S(0), ..., S(T)] into sliding windows.

    X              : shape (N, window_length, 16)  — input history sequences
    Y_next_state   : shape (N, 16)                 — target next-state vector S(t+1)
    Y_stage        : shape (N,)  int64             — stage label of the TARGET S(t+1)

    Parameters
    ----------
    states        : Chronologically sorted list of NetworkState objects.
    labels        : Raw dataset label strings aligned 1-to-1 with states.
                    Must use EXPLICIT mappings via label_mapper.
                    Pass None to skip supervised stage labels (unsupervised mode).
    window_length : Number of past states in each input sequence (default 4).
    scaler        : Pre-fitted StateFeatureScaler.  If None a new scaler is
                    fitted on ALL provided states (use only for quick tests —
                    for proper training always pass a train-fitted scaler).

    Raises
    ------
    UnknownLabelError
        If any label in ``labels`` has no explicit mapping in label_mapper.
    ValueError
        If len(labels) != len(states) when labels is not None.
    """
    if len(states) <= window_length:
        return (
            np.empty((0, window_length, 16), dtype=np.float32),
            np.empty((0, 16), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    # ── Scaler ───────────────────────────────────────────────────────────
    if scaler is None:
        scaler = StateFeatureScaler()
        scaler.fit(states)

    X_mat = scaler.transform(states)   # shape: (T, 16)
    assert X_mat.shape[1] == 16, (
        f"StateFeatureScaler produced {X_mat.shape[1]}-d vectors; expected 16."
    )

    # ── Label vector ──────────────────────────────────────────────────────
    if labels is not None:
        if len(labels) != len(states):
            raise ValueError(
                f"len(labels)={len(labels)} must equal len(states)={len(states)}."
            )
        # Raises UnknownLabelError on first unknown label — no silent fallback.
        y_cat = np.array(
            [map_label_to_int(lbl) for lbl in labels], dtype=np.int64
        )
    else:
        # Unsupervised mode — stage targets default to 0 (Normal).
        y_cat = np.zeros(len(states), dtype=np.int64)

    # ── Sliding windows ───────────────────────────────────────────────────
    X_seqs:        List[np.ndarray] = []
    y_next_states: List[np.ndarray] = []
    y_stages:      List[int]        = []

    for i in range(len(states) - window_length):
        # Input: S(i) … S(i + window_length - 1)
        X_seqs.append(X_mat[i : i + window_length])
        # Target: S(i + window_length)  ← the true NEXT state
        target_idx = i + window_length
        y_next_states.append(X_mat[target_idx])
        y_stages.append(int(y_cat[target_idx]))

    return (
        np.array(X_seqs, dtype=np.float32),
        np.array(y_next_states, dtype=np.float32),
        np.array(y_stages, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Chronological split (no leakage)
# ---------------------------------------------------------------------------

def chronological_split(
    X: np.ndarray,
    y_next: np.ndarray,
    y_stage: np.ndarray,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Split sliding-window arrays into train / val / test sets
    **in chronological order** — no shuffling, no leakage.

    The scaler must be fitted on X_train ONLY before calling transform on
    X_val and X_test.  This function does NOT fit or transform the scaler;
    it is the caller's responsibility.

    Parameters
    ----------
    X         : (N, window_length, features) input sequences — already in
                chronological order (guaranteed by create_sliding_window_sequences).
    y_next    : (N, features) next-state targets.
    y_stage   : (N,) integer stage labels.
    train_frac: Fraction of windows assigned to training (default 0.70).
    val_frac  : Fraction of windows assigned to validation (default 0.15).
                test_frac = 1 - train_frac - val_frac.

    Returns
    -------
    (X_train, y_next_train, y_stage_train),
    (X_val,   y_next_val,   y_stage_val),
    (X_test,  y_next_test,  y_stage_test)
    """
    if not (0 < train_frac < 1 and 0 < val_frac < 1):
        raise ValueError("train_frac and val_frac must be in (0, 1).")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0.")

    N = len(X)
    n_train = int(N * train_frac)
    n_val   = int(N * val_frac)

    slices = {
        "train": slice(0,                  n_train),
        "val":   slice(n_train,            n_train + n_val),
        "test":  slice(n_train + n_val,    N),
    }

    def _split(arr):
        return arr[slices["train"]], arr[slices["val"]], arr[slices["test"]]

    X_tr, X_v, X_te         = _split(X)
    yn_tr, yn_v, yn_te      = _split(y_next)
    ys_tr, ys_v, ys_te      = _split(y_stage)

    return (X_tr, yn_tr, ys_tr), (X_v, yn_v, ys_v), (X_te, yn_te, ys_te)


def split_states_chronologically(
    states: List[NetworkState],
    labels: Optional[List[str]] = None,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
) -> Tuple[
    Tuple[List[NetworkState], Optional[List[str]]],
    Tuple[List[NetworkState], Optional[List[str]]],
    Tuple[List[NetworkState], Optional[List[str]]],
]:
    """
    Split NetworkState objects (and aligned labels) chronologically
    BEFORE fitting the scaler and BEFORE windowing.

    This ensures:
    1. Splitting occurs before any scaler parameters are computed.
    2. Zero overlapping-window leakage across split boundaries.
    """
    if not (0 < train_frac < 1 and 0 < val_frac < 1):
        raise ValueError("train_frac and val_frac must be in (0, 1).")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0.")

    n = len(states)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    states_train = states[:n_train]
    states_val   = states[n_train : n_train + n_val]
    states_test  = states[n_train + n_val :]

    if labels is not None:
        if len(labels) != len(states):
            raise ValueError(f"len(labels)={len(labels)} must equal len(states)={len(states)}.")
        labels_train = labels[:n_train]
        labels_val   = labels[n_train : n_train + n_val]
        labels_test  = labels[n_train + n_val :]
    else:
        labels_train = labels_val = labels_test = None

    return (
        (states_train, labels_train),
        (states_val,   labels_val),
        (states_test,  labels_test),
    )


def prepare_chronological_datasets(
    states: List[NetworkState],
    labels: Optional[List[str]] = None,
    window_length: int = 4,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    StateFeatureScaler,
]:
    """
    End-to-end leakage-free temporal dataset pipeline:
    1. Chronologically splits states & labels into train/val/test FIRST.
    2. Fits StateFeatureScaler strictly on train states ONLY.
    3. Transforms and builds sliding windows independently for each split using the
       train-fitted scaler. This guarantees zero overlapping-window leakage across
       split boundaries and zero scaler lookahead.
    """
    (s_tr, l_tr), (s_v, l_v), (s_te, l_te) = split_states_chronologically(
        states, labels, train_frac=train_frac, val_frac=val_frac
    )

    # Fit scaler ONLY on train states
    scaler = StateFeatureScaler()
    scaler.fit(s_tr)

    # Transform and create sliding window sequences using the train-fitted scaler
    X_tr, yn_tr, ys_tr = create_sliding_window_sequences(
        s_tr, l_tr, window_length=window_length, scaler=scaler
    )
    X_v, yn_v, ys_v = create_sliding_window_sequences(
        s_v, l_v, window_length=window_length, scaler=scaler
    )
    X_te, yn_te, ys_te = create_sliding_window_sequences(
        s_te, l_te, window_length=window_length, scaler=scaler
    )

    return (X_tr, yn_tr, ys_tr), (X_v, yn_v, ys_v), (X_te, yn_te, ys_te), scaler


class TemporalNetworkStateDataset(Dataset):
    """PyTorch Dataset wrapper for temporal network state sequences."""

    def __init__(
        self,
        X_seqs:       np.ndarray,
        y_next_states: np.ndarray,
        y_stages:      np.ndarray,
    ):
        self.X_seqs        = torch.from_numpy(X_seqs).float()       if TORCH_AVAILABLE else X_seqs
        self.y_next_states = torch.from_numpy(y_next_states).float() if TORCH_AVAILABLE else y_next_states
        self.y_stages      = torch.from_numpy(y_stages).long()       if TORCH_AVAILABLE else y_stages

    def __len__(self) -> int:
        return len(self.X_seqs)

    def __getitem__(self, idx: int):
        return self.X_seqs[idx], self.y_next_states[idx], self.y_stages[idx]
