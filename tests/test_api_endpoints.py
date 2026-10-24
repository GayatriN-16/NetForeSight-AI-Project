"""
Integration tests for FastAPI REST Endpoints
"""

import io
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_upload_and_analyze_endpoint():
    csv_content = (
        "timestamp,src_ip,dst_ip,src_port,dst_port,protocol,duration,packet_count,byte_count,syn_flag,ack_flag,rst_flag\n"
        "1.0,192.168.1.10,192.168.1.100,45000,80,TCP,0.05,5,1500,1,1,0\n"
        "2.5,192.168.1.10,192.168.1.100,45002,443,TCP,0.02,3,800,1,1,0\n"
    ).encode("utf-8")

    response = client.post(
        "/api/data/upload",
        files={"file": ("sample.csv", io.BytesIO(csv_content), "text/csv")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "sample.csv"
    assert data["flows_parsed"] == 2
    assert data["summary"]["total_flows"] == 2


def test_upload_invalid_extension():
    response = client.post(
        "/api/data/upload",
        files={"file": ("malicious.exe", io.BytesIO(b"binary"), "application/octet-stream")}
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_build_states_endpoint():
    payload = {
        "flows": [
            {
                "timestamp": 1.0,
                "src_ip": "192.168.1.10",
                "dst_ip": "192.168.1.100",
                "src_port": 5000,
                "dst_port": 80,
                "protocol": "TCP",
                "duration": 0.1,
                "packet_count": 4,
                "byte_count": 500,
                "syn_flag": True,
                "ack_flag": True,
                "rst_flag": False,
                "mean_iat": 25.0,
                "std_iat": 5.0,
                "max_iat": 30.0
            }
        ],
        "window_size_seconds": 5.0
    }
    response = client.post("/api/states/build", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_states"] == 1
    assert data["states"][0]["active_flows"] == 1


def test_demo_forecast_endpoint():
    response = client.get("/api/demo/forecast?k_steps=3")
    assert response.status_code == 200
    data = response.json()
    assert data["is_demo_mode"] is True
    assert data["forecast_horizon"] == 3
    assert len(data["forecast_steps"]) == 3
    assert data["forecast_steps"][0]["forecasted_probability"] > 0.0


def test_demo_states_endpoint():
    response = client.get("/api/demo/states")
    assert response.status_code == 200
    data = response.json()
    assert data["total_states"] == 5
    assert data["states"][0]["is_demo"] is True


def test_evaluation_metrics_endpoint():
    response = client.get("/api/evaluation/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "temporal_lstm_world_model" in data
    assert "logistic_regression_baseline" in data
    if data["is_empirical"]:
        assert data["temporal_lstm_world_model"]["f1_score"] is not None
    else:
        assert data["status"] == "Model evaluation pending."
