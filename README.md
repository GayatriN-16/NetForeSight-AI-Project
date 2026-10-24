# NetForeSight AI 🛡️
> **AI-Based Network Attack Forecasting from Network Traffic Data**  
> *Smart India Hackathon (SIH26153) — National Technical Research Organisation (NTRO)*

---

## 1. Overview & Architecture

NetForeSight AI is a predictive network defense system that moves beyond reactive intrusion detection. Instead of classifying past packets, NetForeSight AI builds a **Temporal World Model** $P(S_{t+1} \mid S_t, S_{t-1}, \dots)$ that predicts continuous future network states $S(t+1)$ and anticipates upcoming cyber kill-chain progression stages before an attack executes.

```
+---------------------------------------------------------------------------------------------------+
|                                    NetForeSight AI Architecture                                   |
+---------------------------------------------------------------------------------------------------+
|  [Raw Flows / CSV / PCAP]                                                                         |
|            |                                                                                      |
|            v                                                                                      |
|   CSVParserService        --> Bidirectional flow parsing, column normalization, NaN/Inf cleanup   |
|            |                                                                                      |
|            v                                                                                      |
|  NetworkStateBuilder      --> Contiguous 5.0s window aggregation into 16-D NetworkState S(t)      |
|            |                                                                                      |
|            v                                                                                      |
|  Chronological Splitting  --> Train (70%), Val (15%), Test (15%) partitioned BEFORE scaler fit     |
|            |                                                                                      |
|            v                                                                                      |
|  StateFeatureScaler       --> StandardScaler fitted STRICTLY on train states (Zero Leakage)       |
|            |                                                                                      |
|            v                                                                                      |
|  Sliding Window Sequences --> Past history [S(t-3), ..., S(t)] --> Target S(t+1) + Stage(t+1)     |
|            |                                                                                      |
|            +------------------------------------+-------------------------------------+          |
|            |                                                                          |          |
|            v                                                                          v          |
|  [Baseline Models]                                                    [LSTM World Model]          |
|  - Persistence S(t+1) = S(t)                                          - 2-layer recurrent encoder |
|  - Static Logistic Regression                                         - Multi-task heads:         |
|                                                                         * State regression (16-D) |
|                                                                         * Stage classifier (5-cls)|
+---------------------------------------------------------------------------------------------------+
```

---

## 2. 16-Dimensional Network State Representation

Every 5.0-second time window is converted into a 16-element network state vector $S(t) \in \mathbb{R}^{16}$:

| Index | Feature Name | Description |
| :---: | :--- | :--- |
| **0** | `active_flows` | Count of distinct active flows in the window |
| **1** | `unique_src_ips` | Count of unique source host IP addresses |
| **2** | `unique_dst_ips` | Count of target destination IP addresses |
| **3** | `unique_dst_ports` | Count of unique target ports |
| **4** | `connection_fan_out` | Ratio of target endpoints to source endpoints |
| **5** | `syn_count` | Count of TCP SYN packets (half-open scan detection) |
| **6** | `ack_count` | Count of TCP ACK packets |
| **7** | `rst_count` | Count of TCP RST packets (rejected connections) |
| **8** | `syn_ack_ratio` | $\text{SYN} / (\text{ACK} + 1)$ protocol imbalance |
| **9** | `failed_connection_ratio` | $\min(1.0, (\text{RST} + \max(0, \text{SYN}-\text{ACK})) / \text{active\_flows})$ |
| **10** | `packet_rate` | Throughput in packets per second |
| **11** | `byte_rate` | Bandwidth in bytes per second |
| **12** | `mean_packet_size` | Average bytes per packet payload |
| **13** | `mean_iat` | Mean flow inter-arrival time in milliseconds |
| **14** | `iat_variance` | Variance of flow inter-arrival times ($ms^2$) |
| **15** | `port_scan_score` | Heuristic composite port dispersion score $[0.0, 1.0]$ |

---

## 3. Empirical Evaluation Results (Real Held-Out Test Data)

Evaluation was performed on the official **CSE-CIC-IDS2018** scenario (`Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv`, contiguous 247,875-flow slice).

* Total flows parsed: **247,875**
* Contiguous 5.0-second temporal states built: **8,640**
* Chronological split: **6,044 Train** (70%), **1,292 Val** (15%), **1,292 Test** (15%)
* Scaler fitting: Fitted **only on Train states**
* Evaluation boundary: **Zero overlapping-window leakage** across split boundaries

### Next-State Forecasting Performance ($S(t+1)$ Regression)

| Model | Next-State MAE | Next-State RMSE | Error Reduction vs Baseline |
| :--- | :---: | :---: | :---: |
| **Persistence Baseline** ($S_{t+1} = S_t$) | `0.4695` | `1.1202` | *Reference* |
| **PyTorch LSTM World Model** | **`0.3932`** | **`0.8477`** | **-16.24% MAE / -24.33% RMSE** |

### Next-Stage Attack Progression Classification

| Model | Stage Accuracy | Precision (Weighted) | Recall (Weighted) | F1 Score (Weighted) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression Baseline** | `53.79%` | `0.5485` | `0.5379` | `0.5228` |
| **PyTorch LSTM World Model** | **`54.49%`** | **`0.6831`** | **`0.5449`** | `0.4276` |

*Evaluation metrics are stored at [`ml/evaluation/metrics.json`](file:///Users/vaishnavinikam/Downloads/NetForeSight-AI%20/ml/evaluation/metrics.json) and served live via `/api/evaluation/metrics`.*

---

## 4. Setup & Installation

### Requirements
* Python 3.10+ (tested on Python 3.14)
* PyTorch 2.0+
* FastAPI & Uvicorn
* Scikit-Learn, Pandas, NumPy

### Setup
```bash
# Clone and enter the repository
cd NetForeSight-AI

# Create and activate a virtual environment (optional)
python3 -m venv venv
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 5. Execution Instructions

### A. Run the Complete Test Suite
```bash
python3 -m pytest tests/ -v
```
*Current test suite status*: **75 passed, 0 failed, 0 errors** across 8 test suites.

### B. Run the Model Training & Evaluation Pipeline
```bash
python3 ml/training/train_pipeline.py
```
This script runs the full end-to-end pipeline:
1. Ingests the 247,875 flows from `data/sample/cse_cic_ids2018_infiltration_scenario.csv`.
2. Assembles 8,640 five-second states $S(t)$ with aligned labels.
3. Chronologically partitions Train, Val, and Test subsets.
4. Fits the scaler only on Train states.
5. Trains baselines and the PyTorch LSTM Temporal World Model.
6. Evaluates on the held-out test split and saves weights to `ml/weights/lstm_world_model.pt`.
7. Writes empirical evaluation metrics to `ml/evaluation/metrics.json`.

### C. Run the Backend API Server
```bash
python3 -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
Once started, the following interfaces and endpoints become available:
* **Interactive UI Dashboard**: [http://localhost:8000/](http://localhost:8000/)
* **Interactive API Documentation (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **Empirical Evaluation Metrics**: `GET http://localhost:8000/api/evaluation/metrics`
* **Real-time Attack Forecast**: `POST http://localhost:8000/api/forecast`
* **Demonstration Scenario**: `GET http://localhost:8000/api/demo/forecast`
* **Health Check**: `GET http://localhost:8000/api/health`

---

## 6. Pipeline Integrity & Zero-Leakage Verification

1. **Chronological Splitting Order**: Slicing occurs *prior* to `scaler.fit()`. The scaler is fitted exclusively on the first 70% of states.
2. **Boundary Disjointness**: State lists are partitioned before sequence building, ensuring that no input sequence or prediction target straddles the train/val or val/test boundaries.
3. **Explicit Label Safety**: Raw CSE-CIC-IDS2018 labels (`"Benign"`, `"Infilteration"`) map explicitly to internal stages. Any missing, empty, or unmapped label raises `UnknownLabelError`. Silent fallback to `"Normal"` has been eliminated.
4. **Temporal Target Alignment**: Input window $[S_t, \dots, S_{t+3}]$ strictly predicts future state $S_{t+4}$ and future stage $\text{stage}(S_{t+4})$.

---

## 7. Limitations & Ethical Boundary Disclaimers

1. **Dataset-Label Approximation**: Mapping the raw dataset label `"Infilteration"` to `"Initial Access"` is an approximation based on the documented attack script for Wednesday-28-02-2018 (spear-phishing delivery and port-scanning). It does **not** prove ground-truth MITRE ATT&CK transitions or lateral movement at the individual flow level.
2. **Single-Day Capture Scenario**: The current local sample represents one attack scenario slice. Generalization across novel multi-day enterprise topologies requires diverse capture environments.
3. **Forecasting Horizon**: Autoregressive rollouts beyond $K=5$ steps (25 seconds) exhibit increasing variance and uncertainty, as reflected in the confidence intervals.
