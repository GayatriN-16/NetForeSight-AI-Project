"""
Unit tests for MITRE ATT&CK mapper and Explainability service
"""

from backend.app.schemas.state import NetworkState
from backend.app.services.mitre_mapper import MitreMapperService
from backend.app.services.explainability import ExplainabilityService


def test_mitre_mapping_reconnaissance():
    # State with high port scan score (should trigger T1046)
    scan_state = NetworkState(
        window_index=0,
        timestamp_start=0.0,
        timestamp_end=5.0,
        window_duration=5.0,
        active_flows=50,
        unique_src_ips=1,
        unique_dst_ips=1,
        unique_dst_ports=40,
        connection_fan_out=40.0,
        syn_count=50,
        ack_count=2,
        rst_count=15,
        syn_ack_ratio=16.6,
        failed_connection_ratio=0.7,
        packet_rate=80.0,
        byte_rate=5000.0,
        mean_packet_size=64.0,
        mean_iat=10.0,
        iat_variance=12.0,
        max_iat=25.0,
        port_scan_score=0.85
    )

    report = MitreMapperService.evaluate_states([scan_state])
    assert report.has_verified_evidence is True
    technique_ids = [t.technique_id for t in report.active_mappings]
    assert "T1046" in technique_ids  # Network Service Discovery


def test_mitre_mapping_normal_traffic():
    # Benign normal state (should trigger no malicious MITRE mappings)
    normal_state = NetworkState(
        window_index=0,
        timestamp_start=0.0,
        timestamp_end=5.0,
        window_duration=5.0,
        active_flows=5,
        unique_src_ips=2,
        unique_dst_ips=2,
        unique_dst_ports=2,
        connection_fan_out=1.0,
        syn_count=5,
        ack_count=5,
        rst_count=0,
        syn_ack_ratio=0.83,
        failed_connection_ratio=0.0,
        packet_rate=10.0,
        byte_rate=3000.0,
        mean_packet_size=300.0,
        mean_iat=80.0,
        iat_variance=90.0,
        max_iat=150.0,
        port_scan_score=0.05
    )

    report = MitreMapperService.evaluate_states([normal_state])
    assert len(report.active_mappings) == 0


def test_explainability_service():
    state = NetworkState(
        window_index=0,
        timestamp_start=0.0,
        timestamp_end=5.0,
        window_duration=5.0,
        active_flows=30,
        unique_src_ips=1,
        unique_dst_ips=10,
        unique_dst_ports=15,
        connection_fan_out=15.0,
        syn_count=30,
        ack_count=2,
        rst_count=10,
        syn_ack_ratio=10.0,
        failed_connection_ratio=0.6,
        packet_rate=60.0,
        byte_rate=8000.0,
        mean_packet_size=133.0,
        mean_iat=15.0,
        iat_variance=20.0,
        max_iat=45.0,
        port_scan_score=0.75
    )

    report = ExplainabilityService.explain_forecast([state], forecasted_prob=0.82)
    assert report.is_model_derived is True
    assert len(report.top_contributions) > 0
    top_feature = report.top_contributions[0]
    assert top_feature.contribution_score > 0.0
    assert "82.0%" in report.narrative_summary or "82%" in report.narrative_summary
