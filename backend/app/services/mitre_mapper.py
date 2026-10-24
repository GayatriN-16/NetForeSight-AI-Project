"""
NetForeSight AI - MITRE ATT&CK Evidence Grounding Engine
Maps observed and forecasted network state anomalies directly to verified MITRE tactics and techniques.
STRICT RULE: Never fabricate MITRE mappings. Only include mappings when evidence conditions are met.
"""

from typing import List
from backend.app.schemas.state import NetworkState
from backend.app.schemas.mitre import MitreTechnique, MitreReport


class MitreMapperService:
    """Evaluates network state vectors S(t) against MITRE ATT&CK technique evidence criteria."""

    @staticmethod
    def evaluate_states(states: List[NetworkState], forecasted_states: List[NetworkState] = None) -> MitreReport:
        """
        Maps observed state history and forecasted future states to verified MITRE techniques.
        """
        active_mappings: List[MitreTechnique] = []
        forecasted_mappings: List[MitreTechnique] = []

        if states:
            active_mappings = MitreMapperService._detect_techniques(states[-1])
        
        if forecasted_states:
            for fs in forecasted_states:
                f_techs = MitreMapperService._detect_techniques(fs)
                for t in f_techs:
                    if not any(existing.technique_id == t.technique_id for existing in forecasted_mappings):
                        forecasted_mappings.append(t)

        return MitreReport(
            active_mappings=active_mappings,
            forecasted_mappings=forecasted_mappings,
            has_verified_evidence=len(active_mappings) > 0 or len(forecasted_mappings) > 0
        )

    @staticmethod
    def _detect_techniques(state: NetworkState) -> List[MitreTechnique]:
        """Examines an individual NetworkState vector for concrete behavioral evidence triggers."""
        techniques: List[MitreTechnique] = []

        # 1. T1046: Network Service Discovery (Reconnaissance)
        # Evidence: High port scan score (> 0.40) or unique destination ports >= 10 in a single 5s window
        if state.port_scan_score >= 0.40 or state.unique_dst_ports >= 10:
            confidence = min(0.95, round(0.5 + (state.port_scan_score * 0.45), 2))
            triggers = [
                f"Port scan heuristic score: {state.port_scan_score:.2f} (Threshold >= 0.40)",
                f"Targeted destination ports in window: {state.unique_dst_ports}",
                f"SYN/ACK imbalance ratio: {state.syn_ack_ratio:.2f}"
            ]
            techniques.append(MitreTechnique(
                technique_id="T1046",
                technique_name="Network Service Discovery",
                tactic_id="TA0007",
                tactic_name="Reconnaissance",
                description="Adversary is attempting to get a listing of services running on target hosts via port sweeping.",
                confidence=confidence,
                evidence_triggers=triggers,
                mitigation_guidance="Deploy rate-limiting on border firewalls and inspect half-open TCP SYN streams."
            ))

        # 2. T1595: Active Scanning (Reconnaissance)
        # Evidence: Connection fan-out >= 3.0 combined with high failed connection ratio
        if state.connection_fan_out >= 3.0 and state.failed_connection_ratio >= 0.35:
            confidence = min(0.92, round(0.4 + (state.connection_fan_out / 10.0), 2))
            triggers = [
                f"Connection fan-out: {state.connection_fan_out:.2f} (Threshold >= 3.0)",
                f"Failed connection ratio: {state.failed_connection_ratio * 100:.1f}%",
                f"Active flows: {state.active_flows}"
            ]
            techniques.append(MitreTechnique(
                technique_id="T1595",
                technique_name="Active Scanning",
                tactic_id="TA0043",
                tactic_name="Reconnaissance",
                description="Adversary is systematically scanning IP ranges to identify vulnerable endpoints.",
                confidence=confidence,
                evidence_triggers=triggers,
                mitigation_guidance="Block offending source IP subnets at the perimeter firewall."
            ))

        # 3. T1190: Exploit Public-Facing Application (Initial Access)
        # Evidence: Sustained packet rate > 50 pps with high RST/failure ratio and byte spikes
        if state.packet_rate >= 50.0 and state.failed_connection_ratio >= 0.4 and state.connection_fan_out < 3.0:
            confidence = min(0.88, round(0.5 + (state.failed_connection_ratio * 0.35), 2))
            triggers = [
                f"Packet burst rate: {state.packet_rate:.1f} pkts/sec",
                f"High RST / failed connection ratio: {state.failed_connection_ratio * 100:.1f}%",
                f"Mean packet payload: {state.mean_packet_size:.1f} bytes"
            ]
            techniques.append(MitreTechnique(
                technique_id="T1190",
                technique_name="Exploit Public-Facing Application",
                tactic_id="TA0001",
                tactic_name="Initial Access",
                description="Adversary is attempting to exploit vulnerabilities in public-facing services.",
                confidence=confidence,
                evidence_triggers=triggers,
                mitigation_guidance="Verify Web Application Firewall (WAF) rule sets and patch vulnerable endpoints."
            ))

        # 4. T1021: Remote Services (Lateral Movement)
        # Evidence: High connection fan-out internally with balanced SYN/ACK (established connections)
        if state.connection_fan_out >= 2.5 and state.syn_ack_ratio <= 1.5 and state.active_flows >= 5:
            confidence = min(0.85, round(0.5 + (state.connection_fan_out / 8.0), 2))
            triggers = [
                f"Internal fan-out expansion: {state.connection_fan_out:.2f}",
                f"Established two-way sessions (SYN/ACK ratio: {state.syn_ack_ratio:.2f})",
                f"Active concurrent internal flows: {state.active_flows}"
            ]
            techniques.append(MitreTechnique(
                technique_id="T1021",
                technique_name="Remote Services",
                tactic_id="TA0008",
                tactic_name="Lateral Movement",
                description="Adversary is leveraging valid or compromised credentials to move laterally across internal network segments.",
                confidence=confidence,
                evidence_triggers=triggers,
                mitigation_guidance="Enforce network micro-segmentation and require MFA for internal remote administration."
            ))

        return techniques
