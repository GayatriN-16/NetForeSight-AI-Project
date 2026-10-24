"""
Phase 4 Step 2 — CSV ingestion tests for the real CSE-CIC-IDS2018 sample.
Run with: python3 -m pytest tests/ -v
"""
import pathlib, json
import pandas as pd
import numpy as np
import pytest

SAMPLE_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "sample"
SLICE_CSV  = SAMPLE_DIR / "cse_cic_ids2018_infiltration_scenario.csv"
META_JSON  = SAMPLE_DIR / "METADATA.json"

REQUIRED_FIELDS = [
    "Timestamp", "Dst Port", "Protocol", "Flow Duration",
    "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max",
    "SYN Flag Cnt", "ACK Flag Cnt", "RST Flag Cnt",
    "Init Fwd Win Byts", "Init Bwd Win Byts", "Label",
]

@pytest.fixture(scope="module")
def df():
    assert SLICE_CSV.exists(), f"Slice CSV not found: {SLICE_CSV}"
    return pd.read_csv(SLICE_CSV, low_memory=False)

@pytest.fixture(scope="module")
def meta():
    assert META_JSON.exists(), f"METADATA.json not found: {META_JSON}"
    return json.loads(META_JSON.read_text())

# ── File existence ──────────────────────────────────────────────────────────
def test_slice_csv_exists():
    assert SLICE_CSV.exists()

def test_metadata_json_exists():
    assert META_JSON.exists()

# ── Row count ───────────────────────────────────────────────────────────────
def test_slice_row_count(df, meta):
    assert len(df) == meta["selected_row_range"]["total_rows"], \
        f"Expected {meta['selected_row_range']['total_rows']} rows, got {len(df)}"

# ── Required fields ─────────────────────────────────────────────────────────
def test_required_fields_present(df):
    missing = [f for f in REQUIRED_FIELDS if f not in df.columns]
    assert not missing, f"Missing required fields: {missing}"

# ── Labels not renamed ──────────────────────────────────────────────────────
def test_label_values_not_fabricated(df):
    allowed = {"Benign", "Infilteration"}
    actual  = set(df["Label"].unique())
    unknown = actual - allowed
    assert not unknown, f"Unexpected label values found: {unknown}"

def test_both_classes_present(df):
    labels = set(df["Label"].unique())
    assert "Benign" in labels
    assert "Infilteration" in labels

# ── Data quality ─────────────────────────────────────────────────────────────
def test_no_inf_in_required_numeric_cols(df):
    num_cols = [c for c in REQUIRED_FIELDS if c not in ("Timestamp", "Protocol", "Label")]
    for col in num_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        assert not np.isinf(series).any(), f"Inf found in column: {col}"

def test_no_nan_in_required_fields_after_fill(df):
    """Required numeric cols should have no NaN after coercion + fillna(0)."""
    num_cols = [c for c in REQUIRED_FIELDS if c not in ("Timestamp", "Protocol", "Label")]
    for col in num_cols:
        filled = pd.to_numeric(df[col], errors="coerce").fillna(0)
        assert not filled.isnull().any(), f"NaN remains in {col} after fillna(0)"

# ── Temporal ordering ─────────────────────────────────────────────────────
def test_timestamps_parseable(df):
    ts = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")
    assert ts.notnull().all(), "Some Timestamp values could not be parsed"

def test_timestamps_span_expected_range(df):
    """CICFlowMeter records flows in completion order, not start order — so rows
    are NOT monotonically sorted. This test verifies timestamps span the expected
    date range for 28/02/2018 only (no out-of-range dates)."""
    ts = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")
    assert ts.min().date().isoformat() == "2018-02-28", \
        f"Unexpected start date: {ts.min()}"
    assert ts.max().date().isoformat() == "2018-02-28", \
        f"Unexpected end date: {ts.max()}"
    # Confirm not all identical (sanity check)
    assert ts.nunique() > 100, "Too few unique timestamps"

# ── Metadata integrity ───────────────────────────────────────────────────
def test_metadata_has_required_keys(meta):
    for key in ["source", "original_filename", "selected_row_range",
                "timestamp_range", "label_distribution", "preprocessing"]:
        assert key in meta, f"METADATA.json missing key: {key}"

def test_metadata_label_distribution_matches_csv(df, meta):
    actual = df["Label"].value_counts().to_dict()
    for lbl, cnt in meta["label_distribution"].items():
        assert actual.get(lbl, 0) == cnt, \
            f"Label '{lbl}': metadata says {cnt}, CSV has {actual.get(lbl, 0)}"

# ── Windowing smoke test ────────────────────────────────────────────────────
def test_5sec_windows_non_empty(df):
    """Verify at least 100 non-empty 5-second windows can be built."""
    ts = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")
    df2 = df.copy()
    df2["_ts"] = ts
    df2 = df2.sort_values("_ts").reset_index(drop=True)
    t = df2["_ts"].min()
    t_end = df2["_ts"].max()
    count = 0
    while t < t_end:
        mask = (df2["_ts"] >= t) & (df2["_ts"] < t + pd.Timedelta(seconds=5))
        if mask.sum() > 0:
            count += 1
        t += pd.Timedelta(seconds=5)
        if count >= 100:
            break
    assert count >= 100, f"Only {count} non-empty windows found"
