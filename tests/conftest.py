"""
Pytest configuration for NetForeSight AI.
Ensures project root is always on sys.path and provides shared test fixtures.
"""
import sys
import pathlib
import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.schemas.state import NetworkState


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
