"""
Unit tests for NetworkStateBuilder
"""

from backend.app.schemas.flow import FlowRecord
from backend.app.services.state_builder import NetworkStateBuilder


def test_build_states_windowing():
    # 5 flows spanning 12 seconds with 5-second window size -> expects 3 windows
    flows = [
        FlowRecord(timestamp=1.0, src_ip="10.0.0.1", dst_ip="10.0.0.2", src_port=1000, dst_port=80, syn_flag=True, ack_flag=True, packet_count=5, byte_count=500),
        FlowRecord(timestamp=3.5, src_ip="10.0.0.1", dst_ip="10.0.0.2", src_port=1001, dst_port=443, syn_flag=True, ack_flag=True, packet_count=3, byte_count=300),
        FlowRecord(timestamp=6.0, src_ip="10.0.0.1", dst_ip="10.0.0.3", src_port=1002, dst_port=80, syn_flag=True, ack_flag=False, rst_flag=True, packet_count=2, byte_count=120),
        FlowRecord(timestamp=8.2, src_ip="10.0.0.1", dst_ip="10.0.0.4", src_port=1003, dst_port=22, syn_flag=True, ack_flag=False, packet_count=1, byte_count=60),
        FlowRecord(timestamp=11.5, src_ip="10.0.0.5", dst_ip="10.0.0.2", src_port=2000, dst_port=80, syn_flag=True, ack_flag=True, packet_count=10, byte_count=1000),
    ]

    series = NetworkStateBuilder.build_states(flows, window_size_seconds=5.0)
    assert series.window_size_seconds == 5.0
    assert series.total_states == 3
    assert len(series.states) == 3

    # Window 0 (0-5s): 2 flows (timestamps 1.0, 3.5)
    s0 = series.states[0]
    assert s0.window_index == 0
    assert s0.active_flows == 2
    assert s0.unique_src_ips == 1
    assert s0.unique_dst_ips == 1
    assert s0.unique_dst_ports == 2
    assert s0.syn_count == 2
    assert s0.ack_count == 2
    assert len(s0.raw_feature_vector) == 16

    # Window 1 (5-10s): 2 flows (timestamps 6.0, 8.2)
    s1 = series.states[1]
    assert s1.window_index == 1
    assert s1.active_flows == 2
    assert s1.unique_dst_ips == 2
    assert s1.rst_count == 1

    # Window 2 (10-15s): 1 flow (timestamp 11.5)
    s2 = series.states[2]
    assert s2.window_index == 2
    assert s2.active_flows == 1
    assert s2.unique_src_ips == 1


def test_empty_flows_state_builder():
    series = NetworkStateBuilder.build_states([], window_size_seconds=5.0)
    assert series.total_states == 0
    assert len(series.states) == 0
