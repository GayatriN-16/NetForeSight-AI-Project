"""
NetForeSight AI - Controlled Demo Scenario Service
Provides controlled, preprocessed attack progression sequences for hackathon demonstration.
STRICT COMPLIANCE: Every response is explicitly tagged with is_demo_mode=True.
"""

from typing import List
from backend.app.schemas.state import NetworkState, NetworkStateSeries
from backend.app.schemas.forecast import ForecastResponse, ForecastStep, AttackStage
from backend.app.schemas.mitre import MitreTechnique, MitreReport
from backend.app.schemas.explainability import FeatureContribution, ExplainabilityReport


class DemoService:
    """Generates a verified, controlled demonstration scenario reproducing attack progression."""

    @staticmethod
    def get_demo_state_series() -> NetworkStateSeries:
        """
        Returns a sequence of 6 preprocessed network states depicting the progression:
        Window 0: Normal Baseline
        Window 1: Normal with light background traffic
        Window 2: Suspicious probing / initial connection fan-out
        Window 3: Active Reconnaissance (Port Scan)
        Window 4: Initial Access attempt (high burst rate + failures)
        Window 5: Pivot preparation (lateral probe)
        """
        states: List[NetworkState] = [
            NetworkState(
                window_index=0,
                timestamp_start=0.0,
                timestamp_end=5.0,
                window_duration=5.0,
                active_flows=12,
                unique_src_ips=4,
                unique_dst_ips=3,
                unique_dst_ports=3,
                connection_fan_out=1.1,
                syn_count=12,
                ack_count=12,
                rst_count=0,
                syn_ack_ratio=0.92,
                failed_connection_ratio=0.0,
                packet_rate=18.4,
                byte_rate=4200.0,
                mean_packet_size=228.0,
                mean_iat=85.0,
                iat_variance=120.0,
                max_iat=210.0,
                port_scan_score=0.05,
                is_demo=True
            ),
            NetworkState(
                window_index=1,
                timestamp_start=5.0,
                timestamp_end=10.0,
                window_duration=5.0,
                active_flows=15,
                unique_src_ips=5,
                unique_dst_ips=4,
                unique_dst_ports=4,
                connection_fan_out=1.2,
                syn_count=15,
                ack_count=14,
                rst_count=1,
                syn_ack_ratio=1.0,
                failed_connection_ratio=0.06,
                packet_rate=22.0,
                byte_rate=5100.0,
                mean_packet_size=231.0,
                mean_iat=78.0,
                iat_variance=110.0,
                max_iat=195.0,
                port_scan_score=0.08,
                is_demo=True
            ),
            NetworkState(
                window_index=2,
                timestamp_start=10.0,
                timestamp_end=15.0,
                window_duration=5.0,
                active_flows=28,
                unique_src_ips=2,
                unique_dst_ips=8,
                unique_dst_ports=14,
                connection_fan_out=4.0,
                syn_count=28,
                ack_count=8,
                rst_count=7,
                syn_ack_ratio=3.11,
                failed_connection_ratio=0.45,
                packet_rate=45.2,
                byte_rate=7800.0,
                mean_packet_size=172.0,
                mean_iat=34.0,
                iat_variance=45.0,
                max_iat=88.0,
                port_scan_score=0.52,
                is_demo=True
            ),
            NetworkState(
                window_index=3,
                timestamp_start=15.0,
                timestamp_end=20.0,
                window_duration=5.0,
                active_flows=65,
                unique_src_ips=1,
                unique_dst_ips=15,
                unique_dst_ports=42,
                connection_fan_out=12.0,
                syn_count=65,
                ack_count=4,
                rst_count=22,
                syn_ack_ratio=13.0,
                failed_connection_ratio=0.72,
                packet_rate=98.0,
                byte_rate=12400.0,
                mean_packet_size=126.0,
                mean_iat=12.0,
                iat_variance=18.0,
                max_iat=42.0,
                port_scan_score=0.89,
                is_demo=True
            ),
            NetworkState(
                window_index=4,
                timestamp_start=20.0,
                timestamp_end=25.0,
                window_duration=5.0,
                active_flows=82,
                unique_src_ips=1,
                unique_dst_ips=2,
                unique_dst_ports=4,
                connection_fan_out=3.5,
                syn_count=82,
                ack_count=18,
                rst_count=35,
                syn_ack_ratio=4.31,
                failed_connection_ratio=0.58,
                packet_rate=142.0,
                byte_rate=48900.0,
                mean_packet_size=344.0,
                mean_iat=8.0,
                iat_variance=9.0,
                max_iat=24.0,
                port_scan_score=0.64,
                is_demo=True
            ),
        ]

        return NetworkStateSeries(
            window_size_seconds=5.0,
            total_states=len(states),
            states=states,
            summary_metrics={
                "demo_scenario": "CSE-CIC-IDS2018 Infiltration Progression",
                "trajectory": "Normal -> Suspicious -> Reconnaissance -> Initial Access -> Lateral Movement",
                "total_windows": len(states),
                "is_preprocessed_demo": 1.0
            }
        )

    @staticmethod
    def get_demo_forecast(k_steps: int = 3) -> ForecastResponse:
        """Generates a preprocessed 3-step forecast representing attack progression."""
        k = max(1, min(k_steps, 5))
        
        forecast_steps: List[ForecastStep] = [
            ForecastStep(
                step_index=1,
                time_offset_seconds=5.0,
                forecasted_probability=0.74,
                predicted_stage="Reconnaissance",
                confidence_interval={"lower": 0.68, "upper": 0.81},
                predicted_state=NetworkState(
                    window_index=5,
                    timestamp_start=25.0,
                    timestamp_end=30.0,
                    window_duration=5.0,
                    active_flows=70,
                    unique_src_ips=1,
                    unique_dst_ips=18,
                    unique_dst_ports=35,
                    connection_fan_out=14.0,
                    syn_count=70,
                    ack_count=6,
                    rst_count=26,
                    syn_ack_ratio=10.0,
                    failed_connection_ratio=0.65,
                    packet_rate=110.0,
                    byte_rate=15000.0,
                    mean_packet_size=136.0,
                    mean_iat=10.0,
                    iat_variance=14.0,
                    max_iat=30.0,
                    port_scan_score=0.88,
                    is_demo=True
                )
            ),
            ForecastStep(
                step_index=2,
                time_offset_seconds=10.0,
                forecasted_probability=0.86,
                predicted_stage="Initial Access",
                confidence_interval={"lower": 0.79, "upper": 0.92},
                predicted_state=NetworkState(
                    window_index=6,
                    timestamp_start=30.0,
                    timestamp_end=35.0,
                    window_duration=5.0,
                    active_flows=95,
                    unique_src_ips=2,
                    unique_dst_ips=3,
                    unique_dst_ports=5,
                    connection_fan_out=4.2,
                    syn_count=95,
                    ack_count=22,
                    rst_count=40,
                    syn_ack_ratio=4.13,
                    failed_connection_ratio=0.61,
                    packet_rate=165.0,
                    byte_rate=62000.0,
                    mean_packet_size=375.0,
                    mean_iat=6.0,
                    iat_variance=7.0,
                    max_iat=18.0,
                    port_scan_score=0.68,
                    is_demo=True
                )
            ),
            ForecastStep(
                step_index=3,
                time_offset_seconds=15.0,
                forecasted_probability=0.93,
                predicted_stage="Lateral Movement",
                confidence_interval={"lower": 0.87, "upper": 0.97},
                predicted_state=NetworkState(
                    window_index=7,
                    timestamp_start=35.0,
                    timestamp_end=40.0,
                    window_duration=5.0,
                    active_flows=110,
                    unique_src_ips=2,
                    unique_dst_ips=9,
                    unique_dst_ports=6,
                    connection_fan_out=7.5,
                    syn_count=110,
                    ack_count=85,
                    rst_count=10,
                    syn_ack_ratio=1.28,
                    failed_connection_ratio=0.15,
                    packet_rate=190.0,
                    byte_rate=84000.0,
                    mean_packet_size=442.0,
                    mean_iat=4.5,
                    iat_variance=5.0,
                    max_iat=12.0,
                    port_scan_score=0.42,
                    is_demo=True
                )
            )
        ][:k]

        mitre_techs = [
            MitreTechnique(
                technique_id="T1046",
                technique_name="Network Service Discovery",
                tactic_id="TA0007",
                tactic_name="Reconnaissance",
                description="Port scanning and service sweep across multiple target ports.",
                confidence=0.91,
                evidence_triggers=[
                    "Port scan heuristic score: 0.89",
                    "42 distinct destination ports targeted in 5s window",
                    "SYN/ACK ratio 13.0 (half-open connection flood)"
                ],
                mitigation_guidance="Block offending IP and enable port scan drop rules."
            ),
            MitreTechnique(
                technique_id="T1190",
                technique_name="Exploit Public-Facing Application",
                tactic_id="TA0001",
                tactic_name="Initial Access",
                description="Bursts of abnormal payloads targeted at web/application services.",
                confidence=0.86,
                evidence_triggers=[
                    "Packet burst rate: 142.0 pkts/sec",
                    "58.0% failed/reset connection ratio",
                    "High payload size variance"
                ],
                mitigation_guidance="Deploy Web Application Firewall (WAF) deep packet inspection."
            ),
            MitreTechnique(
                technique_id="T1021",
                technique_name="Remote Services",
                tactic_id="TA0008",
                tactic_name="Lateral Movement",
                description="Predicted pivot into internal network segments via administrative ports.",
                confidence=0.82,
                evidence_triggers=[
                    "Internal fan-out expansion to 9 host targets",
                    "Established bidirectional sessions (SYN/ACK ratio 1.28)"
                ],
                mitigation_guidance="Enforce micro-segmentation and monitor RPC/SMB internal traffic."
            )
        ]

        contributions = [
            FeatureContribution(
                feature_name="port_scan_score",
                display_name="Port-Scan Heuristic",
                contribution_score=0.88,
                direction="increases_risk",
                observed_value=0.89,
                baseline_reference=0.05,
                unit="index"
            ),
            FeatureContribution(
                feature_name="connection_fan_out",
                display_name="Connection Fan-out",
                contribution_score=0.76,
                direction="increases_risk",
                observed_value=12.0,
                baseline_reference=1.0,
                unit="ratio"
            ),
            FeatureContribution(
                feature_name="syn_ack_ratio",
                display_name="SYN / ACK Imbalance",
                contribution_score=0.71,
                direction="increases_risk",
                observed_value=13.0,
                baseline_reference=1.0,
                unit="ratio"
            ),
            FeatureContribution(
                feature_name="failed_connection_ratio",
                display_name="Failed Connection Ratio",
                contribution_score=0.64,
                direction="increases_risk",
                observed_value=0.72,
                baseline_reference=0.05,
                unit="ratio"
            ),
            FeatureContribution(
                feature_name="packet_rate",
                display_name="Packet Rate",
                contribution_score=0.52,
                direction="increases_risk",
                observed_value=98.0,
                baseline_reference=15.0,
                unit="pkts/sec"
            )
        ]

        return ForecastResponse(
            forecast_horizon=k,
            window_size=5.0,
            current_stage="Reconnaissance",
            current_infiltration_probability=0.74,
            forecast_steps=forecast_steps,
            mitre_report=MitreReport(
                active_mappings=mitre_techs[:2],
                forecasted_mappings=mitre_techs[2:],
                has_verified_evidence=True
            ),
            explainability=ExplainabilityReport(
                method="demo_preprocessed_attribution",
                top_contributions=contributions,
                narrative_summary=(
                    "DEMO SCENARIO: Network state trajectory exhibits active reconnaissance (T1046) progressing "
                    "into public application exploitation (T1190), with high probability of subsequent lateral movement (T1021)."
                ),
                is_model_derived=False
            ),
            is_demo_mode=True,
            model_version="DemoScenario-CSE-CIC-IDS2018-Preprocessed"
        )
