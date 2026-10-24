"""
NetForeSight AI — CIC-IDS2018 Label Mapper
==========================================
Maps raw dataset labels from the CSE-CIC-IDS2018 corpus to the internal
attack-stage integer targets used by the LSTM classifier head.

Design principles
-----------------
1. Every supported raw label is mapped EXPLICITLY — no silent fallback to 0.
2. Unknown labels raise a clear ValueError so the engineer knows immediately.
3. The official CIC spelling "Infilteration" (one 'l') is preserved as the
   canonical source string. Do NOT rename it in the CSV or mapping keys.
4. "Infilteration" is mapped to stage 3 ("Initial Access") because the
   Wednesday-28-02-2018 capture documents spear-phishing, vulnerability
   exploitation, and scanning activity — the initial stages of the kill chain.
   It does NOT provide reliable ground-truth for Lateral Movement at the
   individual-flow level, so Lateral Movement (4) is NOT assigned here.
5. If your slice lacks flows for a particular stage, that stage simply won't
   appear in the training targets; the classifier can still be trained on the
   stages that ARE present.

Supported files / labels (Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv)
-----------------------------------------------------------------------------
  "Benign"        → stage 0  (Normal)
  "Infilteration" → stage 3  (Initial Access)

Supported stages (STAGE_TO_LABEL in dataset.py)
-----------------------------------------------
  Normal          = 0
  Suspicious      = 1
  Reconnaissance  = 2
  Initial Access  = 3
  Lateral Movement = 4
"""

from typing import Dict, Set

# ---------------------------------------------------------------------------
# Canonical mapping: raw CSV label → internal stage name
# ---------------------------------------------------------------------------
# Only add entries here when a dataset label has a documented, defensible
# mapping to a specific kill-chain stage.
CIC_LABEL_TO_STAGE: Dict[str, str] = {
    # Wednesday-28-02-2018 labels (the only two present in this file)
    "Benign":        "Normal",
    "Infilteration": "Initial Access",   # official CIC spelling preserved

    # Friday-02-03-2018 labels (for future extension)
    "DoS attacks-GoldenEye": "Initial Access",
    "DoS attacks-Slowloris":  "Initial Access",
    "DoS attacks-SlowHTTPTest": "Initial Access",
    "DoS attacks-Hulk":       "Initial Access",
    "DDoS attacks-LOIC-HTTP": "Initial Access",
    "DDOS attack-HOIC":       "Initial Access",

    # Thursday-01-03-2018 labels
    "SQL Injection":  "Initial Access",
    "XSS":            "Initial Access",
    "Brute Force -Web": "Reconnaissance",
    "Brute Force -XSS": "Reconnaissance",

    # Wednesday-21-02-2018 labels
    "Bot":            "Lateral Movement",

    # Thursday-15-02-2018 labels
    "FTP-BruteForce": "Reconnaissance",
    "SSH-Bruteforce":  "Reconnaissance",
}

# ---------------------------------------------------------------------------
# Stage name → integer (must match STAGE_TO_LABEL in dataset.py)
# ---------------------------------------------------------------------------
STAGE_NAME_TO_INT: Dict[str, int] = {
    "Normal":          0,
    "Suspicious":      1,
    "Reconnaissance":  2,
    "Initial Access":  3,
    "Lateral Movement": 4,
}

# Set of labels KNOWN to be present in the Wednesday slice but deliberately
# excluded from supervised classification (empty for this dataset).
EXCLUDED_LABELS: Set[str] = set()


class UnknownLabelError(ValueError):
    """Raised when a raw dataset label has no explicit mapping."""


def map_label_to_stage_name(raw_label: str) -> str:
    """
    Map a raw CIC-IDS2018 label string to an attack-stage name.

    Parameters
    ----------
    raw_label : str
        Label as it appears in the CSV (e.g. ``"Infilteration"``).

    Returns
    -------
    str
        Stage name (e.g. ``"Initial Access"``).

    Raises
    ------
    UnknownLabelError
        If the label is not in CIC_LABEL_TO_STAGE and not in EXCLUDED_LABELS.
    """
    stripped = raw_label.strip()
    if stripped in EXCLUDED_LABELS:
        raise UnknownLabelError(
            f"Label '{stripped}' is explicitly excluded from supervised training. "
            "Exclude this flow or handle it separately."
        )
    if stripped not in CIC_LABEL_TO_STAGE:
        raise UnknownLabelError(
            f"Unknown dataset label: '{stripped}'. "
            "Add it to CIC_LABEL_TO_STAGE in label_mapper.py with an explicit, "
            "documented mapping before training."
        )
    return CIC_LABEL_TO_STAGE[stripped]


def map_label_to_int(raw_label: str) -> int:
    """
    Map a raw CIC-IDS2018 label string to the integer stage index.

    Raises UnknownLabelError for any label not in CIC_LABEL_TO_STAGE.
    """
    stage_name = map_label_to_stage_name(raw_label)
    return STAGE_NAME_TO_INT[stage_name]


def map_labels_to_int_array(raw_labels: list) -> list:
    """
    Vectorised mapping for a list of raw labels.
    Raises UnknownLabelError on the FIRST unknown label encountered.
    """
    return [map_label_to_int(lbl) for lbl in raw_labels]


def get_supported_labels() -> Dict[str, str]:
    """Return the current raw-label → stage-name mapping for inspection."""
    return dict(CIC_LABEL_TO_STAGE)


def get_stages_present(raw_labels: list) -> Set[str]:
    """Return the set of stage NAMES actually present in the provided label list."""
    stages = set()
    for lbl in raw_labels:
        stripped = lbl.strip()
        if stripped in CIC_LABEL_TO_STAGE:
            stages.add(CIC_LABEL_TO_STAGE[stripped])
    return stages
