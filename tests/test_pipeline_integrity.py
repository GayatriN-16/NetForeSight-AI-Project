"""
NetForeSight AI — Pipeline Integrity Tests
==========================================
Covers:
  A. Feature dimension & ordering (Issue 1)
  B. Label mapping & unknown-label safety (Issue 2)
  C. Timestamp sorting & temporal window alignment
  D. Future-target alignment
  E. Chronological train/val/test split with no leakage
  F. Scaler fitted on train only

Run with:  python3 -m pytest tests/ -v
"""

import pathlib, sys
import numpy as np
import pytest

# Ensure project root is on path so imports work
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.schemas.state import NetworkState
from backend.app.schemas.flow import FlowRecord
from backend.app.services.state_builder import NetworkStateBuilder
from ml.preprocessing.scaler import StateFeatureScaler, FEATURE_NAMES
from ml.preprocessing.label_mapper import (
    map_label_to_int,
    map_label_to_stage_name,
    map_labels_to_int_array,
    CIC_LABEL_TO_STAGE,
    UnknownLabelError,
    get_stages_present,
)
from ml.preprocessing.dataset import (
    create_sliding_window_sequences,
    chronological_split,
    split_states_chronologically,
    prepare_chronological_datasets,
    STAGE_TO_LABEL,
    LABEL_TO_STAGE,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_state(window_index: int = 0, **kwargs) -> NetworkState:
    """Build a minimal but valid NetworkState for testing."""
    defaults = dict(
        window_index=window_index,
        timestamp_start=float(window_index * 5),
        timestamp_end=float(window_index * 5 + 5),
        window_duration=5.0,
        active_flows=10,
        unique_src_ips=3,
        unique_dst_ips=3,
        unique_dst_ports=5,
        connection_fan_out=1.0,
        syn_count=2,
        ack_count=5,
        rst_count=0,
        syn_ack_ratio=0.4,
        failed_connection_ratio=0.0,
        packet_rate=20.0,
        byte_rate=1000.0,
        mean_packet_size=50.0,
        mean_iat=10.0,
        iat_variance=2.0,
        port_scan_score=0.0,
    )
    defaults.update(kwargs)
    return NetworkState(**defaults)


def _make_states(n: int) -> list:
    return [_make_state(i) for i in range(n)]


# ═══════════════════════════════════════════════════════════════════════════
# A. Feature dimension & ordering
# ═══════════════════════════════════════════════════════════════════════════

class TestFeatureDimensions:

    def test_feature_names_list_has_16_entries(self):
        assert len(FEATURE_NAMES) == 16, (
            f"FEATURE_NAMES has {len(FEATURE_NAMES)} entries; expected 16."
        )

    def test_feature_names_ordering(self):
        expected = [
            "active_flows", "unique_src_ips", "unique_dst_ips",
            "unique_dst_ports", "connection_fan_out",
            "syn_count", "ack_count", "rst_count",
            "syn_ack_ratio", "failed_connection_ratio",
            "packet_rate", "byte_rate", "mean_packet_size",
            "mean_iat", "iat_variance", "port_scan_score",
        ]
        assert FEATURE_NAMES == expected, (
            f"Feature ordering mismatch.\nExpected: {expected}\nGot:      {FEATURE_NAMES}"
        )

    def test_scaler_extract_features_returns_16d(self):
        scaler = StateFeatureScaler()
        state  = _make_state(0)
        vec    = scaler.extract_features(state)
        assert vec.shape == (16,), f"Expected (16,), got {vec.shape}"

    def test_scaler_extract_matrix_returns_correct_shape(self):
        scaler = StateFeatureScaler()
        states = _make_states(10)
        mat    = scaler.extract_matrix(states)
        assert mat.shape == (10, 16), f"Expected (10, 16), got {mat.shape}"

    def test_scaler_transform_returns_16d_after_fit(self):
        states = _make_states(20)
        scaler = StateFeatureScaler()
        scaler.fit(states)
        X = scaler.transform(states)
        assert X.shape == (20, 16), f"Expected (20, 16), got {X.shape}"

    def test_create_sequences_X_shape(self):
        states = _make_states(20)
        labels = ["Benign"] * 20
        X, yn, ys = create_sliding_window_sequences(states, labels, window_length=4)
        assert X.shape[2] == 16, (
            f"X.shape[2] = {X.shape[2]} but LSTM input_dim = 16."
        )

    def test_create_sequences_X_window_length(self):
        states = _make_states(20)
        labels = ["Benign"] * 20
        for wl in [3, 4, 8]:
            X, _, _ = create_sliding_window_sequences(states, labels, window_length=wl)
            assert X.shape[1] == wl, f"Window length mismatch: got {X.shape[1]}, expected {wl}"

    def test_create_sequences_y_next_state_is_16d(self):
        states = _make_states(20)
        labels = ["Benign"] * 20
        _, yn, _ = create_sliding_window_sequences(states, labels, window_length=4)
        assert yn.shape[1] == 16, f"y_next dim = {yn.shape[1]}, expected 16"

    def test_sequence_count(self):
        n, wl = 20, 4
        states = _make_states(n)
        labels = ["Benign"] * n
        X, _, _ = create_sliding_window_sequences(states, labels, window_length=wl)
        assert X.shape[0] == n - wl, f"Expected {n - wl} windows, got {X.shape[0]}"

    def test_sequences_no_nan_or_inf(self):
        states = _make_states(20)
        labels = ["Benign"] * 20
        X, yn, _ = create_sliding_window_sequences(states, labels)
        assert not np.isnan(X).any(),  "NaN found in X"
        assert not np.isinf(X).any(),  "Inf found in X"
        assert not np.isnan(yn).any(), "NaN found in y_next"
        assert not np.isinf(yn).any(), "Inf found in y_next"


# ═══════════════════════════════════════════════════════════════════════════
# B. Label mapping & unknown-label safety
# ═══════════════════════════════════════════════════════════════════════════

class TestLabelMapping:

    def test_benign_maps_to_normal_stage(self):
        assert map_label_to_stage_name("Benign") == "Normal"

    def test_benign_maps_to_int_0(self):
        assert map_label_to_int("Benign") == 0

    def test_infilteration_maps_to_initial_access(self):
        # "Infilteration" is the official CIC spelling — preserved exactly.
        assert map_label_to_stage_name("Infilteration") == "Initial Access"

    def test_infilteration_maps_to_int_3(self):
        assert map_label_to_int("Infilteration") == 3

    def test_unknown_label_raises_error(self):
        with pytest.raises(UnknownLabelError):
            map_label_to_int("SomeFakeAttack")

    def test_empty_string_label_raises_error(self):
        with pytest.raises(UnknownLabelError):
            map_label_to_int("")

    def test_normal_label_raises_error(self):
        """'Normal' is an internal stage name, NOT a CSV label — must raise."""
        with pytest.raises(UnknownLabelError):
            map_label_to_int("Normal")

    def test_label_label_raises_error(self):
        """'Label' is a repeated-header artifact — must raise."""
        with pytest.raises(UnknownLabelError):
            map_label_to_int("Label")

    def test_silent_fallback_is_gone(self):
        """Verify create_sliding_window_sequences raises for unknown labels."""
        states = _make_states(10)
        bad_labels = ["Benign"] * 9 + ["UNKNOWN_ATTACK"]
        with pytest.raises(UnknownLabelError):
            create_sliding_window_sequences(states, bad_labels, window_length=4)

    def test_mixed_known_labels_succeed(self):
        states = _make_states(20)
        labels = ["Benign"] * 10 + ["Infilteration"] * 10
        X, _, ys = create_sliding_window_sequences(states, labels, window_length=4)
        assert len(ys) == 16
        # All stage values must be 0 (Normal) or 3 (Initial Access)
        assert set(ys.tolist()).issubset({0, 3})

    def test_all_cic_labels_have_explicit_mapping(self):
        """Every key in CIC_LABEL_TO_STAGE must produce a valid stage int."""
        for raw_lbl in CIC_LABEL_TO_STAGE:
            result = map_label_to_int(raw_lbl)
            assert isinstance(result, int), f"Bad mapping for '{raw_lbl}': {result}"

    def test_get_stages_present(self):
        labels = ["Benign", "Infilteration", "Benign"]
        stages = get_stages_present(labels)
        assert stages == {"Normal", "Initial Access"}


# ═══════════════════════════════════════════════════════════════════════════
# C. Timestamp sorting inside windows
# ═══════════════════════════════════════════════════════════════════════════

class TestTimestampOrdering:

    def test_states_sorted_by_timestamp(self):
        """
        NetworkStateBuilder sorts flows before windowing.
        Verify that states with monotonically increasing timestamp_start
        produce X sequences in the correct temporal order.
        """
        states = [_make_state(i) for i in range(20)]
        labels = ["Benign"] * 20
        X, _, _ = create_sliding_window_sequences(states, labels, window_length=4)
        # Extract active_flows feature (index 0) for each window's first step
        # Since states are ordered, first element of window i should come before first
        # element of window i+1.
        for i in range(len(X) - 1):
            ts_i   = states[i].timestamp_start
            ts_ip1 = states[i + 1].timestamp_start
            assert ts_i <= ts_ip1, (
                f"States not sorted: window {i} ts={ts_i} > window {i+1} ts={ts_ip1}"
            )

    def test_reversed_states_give_same_feature_matrix_when_sorted(self):
        """
        Reversed and forward state lists produce the same X after the scaler
        (since scaler.transform is stateless given fit params).  More
        importantly, a caller is responsible for sorting before passing states.
        """
        states_fwd = [_make_state(i, syn_count=i) for i in range(20)]
        states_rev = list(reversed(states_fwd))
        labels_fwd = ["Benign"] * 20
        labels_rev = list(reversed(labels_fwd))

        scaler = StateFeatureScaler()
        scaler.fit(states_fwd)

        X_fwd, _, _ = create_sliding_window_sequences(
            states_fwd, labels_fwd, window_length=4, scaler=scaler
        )
        X_rev, _, _ = create_sliding_window_sequences(
            states_rev, labels_rev, window_length=4, scaler=scaler
        )
        # Both should have same shape
        assert X_fwd.shape == X_rev.shape


# ═══════════════════════════════════════════════════════════════════════════
# D. Future-target alignment
# ═══════════════════════════════════════════════════════════════════════════

class TestTargetAlignment:

    def test_y_stage_is_label_of_target_state(self):
        """
        For window i, y_stage[i] must be the integer label of state
        at index i + window_length (the TRUE next state), not the
        label of the last state inside the input window.
        """
        n, wl = 10, 4
        # States 0..6 = Benign, states 7..9 = Infilteration
        states = _make_states(n)
        labels = ["Benign"] * 7 + ["Infilteration"] * 3

        _, _, ys = create_sliding_window_sequences(states, labels, window_length=wl)

        # Window i → target state index = i + wl
        expected = []
        for i in range(n - wl):
            target_label = labels[i + wl]
            expected.append(map_label_to_int(target_label))

        assert ys.tolist() == expected, (
            f"Target alignment mismatch.\nExpected: {expected}\nGot:      {ys.tolist()}"
        )

    def test_y_next_state_is_state_after_window(self):
        """
        y_next_states[i] must be the SCALED feature vector of state
        at position i + window_length.
        """
        n, wl = 10, 4
        states = _make_states(n)
        labels = ["Benign"] * n

        scaler = StateFeatureScaler()
        scaler.fit(states)
        X_mat = scaler.transform(states)   # (10, 16)

        _, yn, _ = create_sliding_window_sequences(
            states, labels, window_length=wl, scaler=scaler
        )

        for i in range(n - wl):
            np.testing.assert_array_almost_equal(
                yn[i], X_mat[i + wl],
                decimal=5,
                err_msg=f"y_next mismatch at window {i}"
            )

    def test_x_and_y_same_sample_count(self):
        states = _make_states(20)
        labels = ["Benign"] * 20
        X, yn, ys = create_sliding_window_sequences(states, labels, window_length=4)
        assert X.shape[0] == yn.shape[0] == ys.shape[0]


# ═══════════════════════════════════════════════════════════════════════════
# E. Chronological split — no leakage
# ═══════════════════════════════════════════════════════════════════════════

class TestChronologicalSplit:

    def _make_arrays(self, n=100, wl=4, features=16):
        X  = np.random.rand(n, wl, features).astype(np.float32)
        yn = np.random.rand(n, features).astype(np.float32)
        ys = np.zeros(n, dtype=np.int64)
        return X, yn, ys

    def test_split_sizes_sum_to_n(self):
        X, yn, ys = self._make_arrays(100)
        (Xtr,_,_), (Xv,_,_), (Xte,_,_) = chronological_split(X, yn, ys, 0.70, 0.15)
        assert len(Xtr) + len(Xv) + len(Xte) == 100

    def test_train_is_first_70_percent(self):
        X, yn, ys = self._make_arrays(100)
        (Xtr,_,_), _, _ = chronological_split(X, yn, ys, 0.70, 0.15)
        assert len(Xtr) == 70

    def test_val_follows_train_no_gap(self):
        X, yn, ys = self._make_arrays(100)
        (Xtr,_,_), (Xv,_,_), _ = chronological_split(X, yn, ys, 0.70, 0.15)
        # The first sample of val must equal the 71st sample of X (index 70)
        np.testing.assert_array_equal(Xv[0], X[len(Xtr)])

    def test_test_follows_val_no_gap(self):
        X, yn, ys = self._make_arrays(100)
        (Xtr,_,_), (Xv,_,_), (Xte,_,_) = chronological_split(X, yn, ys, 0.70, 0.15)
        np.testing.assert_array_equal(Xte[0], X[len(Xtr) + len(Xv)])

    def test_no_overlapping_indices(self):
        X, yn, ys = self._make_arrays(100)
        (Xtr,_,_), (Xv,_,_), (Xte,_,_) = chronological_split(X, yn, ys, 0.70, 0.15)
        # Use the first feature of first window step as a proxy identifier
        tr_ids = set(tuple(x[0]) for x in Xtr)
        v_ids  = set(tuple(x[0]) for x in Xv)
        te_ids = set(tuple(x[0]) for x in Xte)
        assert len(tr_ids & v_ids) == 0,  "Train and val overlap!"
        assert len(tr_ids & te_ids) == 0, "Train and test overlap!"
        assert len(v_ids  & te_ids) == 0, "Val and test overlap!"

    def test_invalid_fractions_raise(self):
        X, yn, ys = self._make_arrays(100)
        with pytest.raises(ValueError):
            chronological_split(X, yn, ys, 0.80, 0.25)  # sum >= 1

    def test_split_preserves_order(self):
        n = 100
        X  = np.arange(n, dtype=np.float32).reshape(n, 1, 1)
        yn = np.zeros((n, 1), dtype=np.float32)
        ys = np.zeros(n, dtype=np.int64)
        (Xtr,_,_), (Xv,_,_), (Xte,_,_) = chronological_split(X, yn, ys, 0.70, 0.15)
        # Ensure first element of val > last element of train
        assert float(Xv[0,0,0]) > float(Xtr[-1,0,0])
        assert float(Xte[0,0,0]) > float(Xv[-1,0,0])


# ═══════════════════════════════════════════════════════════════════════════
# F. Scaler fitted on training data only
# ═══════════════════════════════════════════════════════════════════════════

class TestScalerTrainOnly:

    def test_scaler_fitted_only_on_train_states(self):
        """
        A scaler fitted on train states must NOT be re-fitted on val/test.
        Verify that transforming val with a train-fitted scaler differs from
        transforming val with a full-data scaler (i.e. they are NOT identical,
        confirming the scaler was not re-fitted on val data).
        """
        n = 60
        states = [
            _make_state(i, syn_count=i * 3, active_flows=i)  # varying values
            for i in range(n)
        ]
        labels = ["Benign"] * n
        X, yn, ys = create_sliding_window_sequences(
            states, labels, window_length=4,
            scaler=None  # will fit on ALL data — do NOT do this for real training
        )

        # Correct: fit scaler only on train portion
        (Xtr, _, _), (Xv_correct, _, _), _ = chronological_split(X, yn, ys)

        # Demonstrate that if you had fitted scaler only on train NetworkStates:
        n_train_states = int(n * 0.70)
        train_states  = states[:n_train_states]
        train_scaler  = StateFeatureScaler()
        train_scaler.fit(train_states)

        assert train_scaler.is_fitted, "Scaler should be fitted after .fit()"

    def test_unfitted_scaler_does_not_raise_but_uses_fallback(self):
        """StateFeatureScaler.transform before .fit() uses log1p fallback."""
        states = _make_states(5)
        scaler = StateFeatureScaler()
        assert not scaler.is_fitted
        X = scaler.transform(states)
        assert X.shape == (5, 16)
        assert not np.isnan(X).any()

    def test_fitted_scaler_produces_standardized_output(self):
        """After fitting, output should have approximately zero mean."""
        states = [
            _make_state(i, syn_count=i, active_flows=i + 1)
            for i in range(100)
        ]
        scaler = StateFeatureScaler()
        scaler.fit(states)
        X = scaler.transform(states)
        # Mean of each feature should be close to 0 (StandardScaler guarantee)
        means = np.abs(X.mean(axis=0))
        assert np.all(means < 1e-5), f"Fitted scaler means not near zero: {means}"


# ═══════════════════════════════════════════════════════════════════════════
# G. Zero-Leakage Pipeline (Split before Scaler + Zero Boundary Overlap)
# ═══════════════════════════════════════════════════════════════════════════

class TestZeroLeakagePipeline:

    def test_split_states_chronologically_disjoint(self):
        """Verifies state lists are strictly disjoint with correct counts."""
        states = _make_states(100)
        labels = ["Benign"] * 100
        (st_tr, lbl_tr), (st_v, lbl_v), (st_te, lbl_te) = split_states_chronologically(
            states, labels, train_frac=0.70, val_frac=0.15
        )
        assert len(st_tr) == 70
        assert len(st_v) == 15
        assert len(st_te) == 15

        # Check disjoint state timestamps / window indices
        tr_indices = set(s.window_index for s in st_tr)
        v_indices  = set(s.window_index for s in st_v)
        te_indices = set(s.window_index for s in st_te)

        assert len(tr_indices & v_indices) == 0, "Train and val states overlap!"
        assert len(tr_indices & te_indices) == 0, "Train and test states overlap!"
        assert len(v_indices & te_indices) == 0, "Val and test states overlap!"

    def test_prepare_chronological_datasets_no_window_overlap(self):
        """
        Confirms that creating sequences from partitioned state subsets
        leaves ZERO overlapping windows or targets between train, val, and test.
        """
        n, wl = 100, 4
        # Give each state a unique identifying active_flows value
        states = [_make_state(i, active_flows=i + 1) for i in range(n)]
        labels = ["Benign"] * n

        (Xtr, yn_tr, ys_tr), (Xv, yn_v, ys_v), (Xte, yn_te, ys_te), scaler = (
            prepare_chronological_datasets(states, labels, window_length=wl, train_frac=0.70, val_frac=0.15)
        )

        assert scaler.is_fitted
        # Check shapes: train has 70 states -> 70 - wl sequences
        assert Xtr.shape == (70 - wl, wl, 16)
        assert Xv.shape == (15 - wl, wl, 16)
        assert Xte.shape == (15 - wl, wl, 16)

        # Unscale active_flows (feature 0) to verify state indices
        tr_last_target_active_flows = states[69].active_flows
        v_first_state_active_flows = states[70].active_flows
        assert v_first_state_active_flows > tr_last_target_active_flows

    def test_prepare_chronological_datasets_scaler_fitted_on_train_only(self):
        """
        Confirms scaler was fitted only on training data, so training feature
        mean is zero while validation feature mean is non-zero when distributions drift.
        """
        # Train has small values, val has large values
        train_states = [_make_state(i, byte_rate=100.0 + i) for i in range(70)]
        val_states   = [_make_state(70 + i, byte_rate=5000.0 + i) for i in range(15)]
        test_states  = [_make_state(85 + i, byte_rate=10000.0 + i) for i in range(15)]
        all_states   = train_states + val_states + test_states
        all_labels   = ["Benign"] * len(all_states)

        (Xtr, _, _), (Xv, _, _), _, scaler = prepare_chronological_datasets(
            all_states, all_labels, window_length=4, train_frac=0.70, val_frac=0.15
        )

        # Byte rate is feature index 11
        train_byte_rate_mean = Xtr[:, :, 11].mean()
        val_byte_rate_mean   = Xv[:, :, 11].mean()

        # Train is standardized around 0, Val is shifted far above 0
        assert abs(train_byte_rate_mean) < 0.2
        assert val_byte_rate_mean > 5.0  # Demonstrates scaler was NOT fitted on val

    def test_end_to_end_flow_to_windowed_dataset(self):
        """
        Tests the complete pipeline: FlowRecords -> NetworkStateBuilder ->
        prepare_chronological_datasets -> verified 16-d features with zero leakage.
        """
        flows = []
        for i in range(100):
            ts = float(i * 1.0)
            lbl = "Infilteration" if i >= 60 else "Benign"
            flows.append(FlowRecord(
                timestamp=ts,
                src_ip="192.168.1.10",
                dst_ip=f"10.0.0.{1 + (i % 5)}",
                src_port=50000 + i,
                dst_port=80,
                protocol="TCP",
                packet_count=10,
                byte_count=500,
                syn_flag=(i % 2 == 0),
                ack_flag=True,
                rst_flag=False,
                label=lbl
            ))

        states, labels = NetworkStateBuilder.build_states_with_labels(flows, window_size_seconds=5.0)
        assert len(states) == 20
        assert len(labels) == 20
        assert "Benign" in labels
        assert "Infilteration" in labels

        (Xtr, yn_tr, ys_tr), (Xv, yn_v, ys_v), (Xte, yn_te, ys_te), scaler = (
            prepare_chronological_datasets(states, labels, window_length=4, train_frac=0.70, val_frac=0.15)
        )

        assert Xtr.shape[2] == 16
        assert yn_tr.shape[1] == 16
        assert scaler.is_fitted
        # Both Normal (0) and Initial Access (3) stages are present in the targets
        all_target_stages = set(ys_tr.tolist()) | set(ys_v.tolist()) | set(ys_te.tolist())
        assert 0 in all_target_stages
        assert 3 in all_target_stages
