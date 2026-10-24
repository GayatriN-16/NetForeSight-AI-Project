"""
NetForeSight AI - Network State Builder Service
Converts discrete, irregular network flow records into timestamped Network State vectors S(t).
"""

import math
from typing import List, Dict, Any, Optional
import numpy as np

from backend.app.schemas.flow import FlowRecord
from backend.app.schemas.state import NetworkState, NetworkStateSeries
from backend.app.core.config import settings


class NetworkStateBuilder:
    """
    Transforms time-series FlowRecords into contiguous windowed NetworkState S(t) representations.
    Default window: 5.0 seconds (Configurable).
    """

    @staticmethod
    def build_states(
        flows: List[FlowRecord], 
        window_size_seconds: float = 5.0,
        is_demo: bool = False
    ) -> NetworkStateSeries:
        """
        Groups flows into contiguous time windows of width `window_size_seconds`
        and computes all topological, protocol, timing, and anomaly features for each S(t).
        """
        if not flows:
            return NetworkStateSeries(
                window_size_seconds=window_size_seconds,
                total_states=0,
                states=[],
                summary_metrics={}
            )

        # Ensure chronological ordering
        sorted_flows = sorted(flows, key=lambda f: f.timestamp)
        t_start_global = sorted_flows[0].timestamp
        t_end_global = sorted_flows[-1].timestamp
        
        # Total span
        total_span = max(window_size_seconds, t_end_global - t_start_global)
        num_windows = max(1, int(math.ceil(total_span / window_size_seconds)))

        # Bucket flows into windows
        buckets: Dict[int, List[FlowRecord]] = {w: [] for w in range(num_windows)}
        for f in sorted_flows:
            w_idx = int((f.timestamp - t_start_global) // window_size_seconds)
            w_idx = min(w_idx, num_windows - 1)
            buckets[w_idx].append(f)

        states: List[NetworkState] = []

        for w_idx in range(num_windows):
            w_flows = buckets[w_idx]
            w_t_start = t_start_global + (w_idx * window_size_seconds)
            w_t_end = w_t_start + window_size_seconds

            state = NetworkStateBuilder._aggregate_window(
                window_index=w_idx,
                timestamp_start=w_t_start,
                timestamp_end=w_t_end,
                window_duration=window_size_seconds,
                flows=w_flows,
                is_demo=is_demo
            )
            states.append(state)

        summary = {
            "total_windows": len(states),
            "window_size_sec": window_size_seconds,
            "mean_active_flows_per_window": float(np.mean([s.active_flows for s in states])) if states else 0.0,
            "max_fan_out": float(np.max([s.connection_fan_out for s in states])) if states else 0.0,
            "max_port_scan_score": float(np.max([s.port_scan_score for s in states])) if states else 0.0
        }

        return NetworkStateSeries(
            window_size_seconds=window_size_seconds,
            total_states=len(states),
            states=states,
            summary_metrics=summary
        )

    @staticmethod
    def build_states_with_labels(
        flows: List[FlowRecord],
        window_size_seconds: float = 5.0,
        is_demo: bool = False
    ) -> Tuple[List[NetworkState], List[str]]:
        """
        Groups flows into contiguous time windows of width `window_size_seconds`,
        computes NetworkState S(t) for each window, and derives the aligned window
        label string (attack label if any attack flow is present in window, else 'Benign').
        """
        series = NetworkStateBuilder.build_states(flows, window_size_seconds, is_demo)
        if not flows or not series.states:
            return [], []

        sorted_flows = sorted(flows, key=lambda f: f.timestamp)
        t_start_global = sorted_flows[0].timestamp
        num_windows = len(series.states)

        labels: List[str] = ["Benign"] * num_windows
        for f in sorted_flows:
            w_idx = int((f.timestamp - t_start_global) // window_size_seconds)
            w_idx = min(w_idx, num_windows - 1)
            if f.label and f.label.strip() != "Benign":
                labels[w_idx] = f.label.strip()

        return series.states, labels

    @staticmethod
    def _aggregate_window(
        window_index: int,
        timestamp_start: float,
        timestamp_end: float,
        window_duration: float,
        flows: List[FlowRecord],
        is_demo: bool = False
    ) -> NetworkState:
        """Computes topological, protocol, timing, and anomaly statistics for a single time window."""
        active_flows = len(flows)
        if active_flows == 0:
            # Idle / quiescent network window
            raw_vec = [0.0] * 16
            return NetworkState(
                window_index=window_index,
                timestamp_start=timestamp_start,
                timestamp_end=timestamp_end,
                window_duration=window_duration,
                active_flows=0,
                unique_src_ips=0,
                unique_dst_ips=0,
                unique_dst_ports=0,
                connection_fan_out=0.0,
                syn_count=0,
                ack_count=0,
                rst_count=0,
                syn_ack_ratio=0.0,
                failed_connection_ratio=0.0,
                packet_rate=0.0,
                byte_rate=0.0,
                mean_packet_size=0.0,
                mean_iat=0.0,
                iat_variance=0.0,
                max_iat=0.0,
                port_scan_score=0.0,
                raw_feature_vector=raw_vec,
                is_demo=is_demo
            )

        src_ips = set(f.src_ip for f in flows)
        dst_ips = set(f.dst_ip for f in flows)
        dst_ports = set(f.dst_port for f in flows)

        total_pkts = sum(f.packet_count for f in flows)
        total_bytes = sum(f.byte_count for f in flows)

        syn_count = sum(1 for f in flows if f.syn_flag)
        ack_count = sum(1 for f in flows if f.ack_flag)
        rst_count = sum(1 for f in flows if f.rst_flag)

        # Topological Fan-out: Distinct targets (IPs and Ports) per Source Endpoint
        connection_fan_out = round(float(len(dst_ports) * len(dst_ips)) / float(max(1, len(src_ips))), 3)

        # Protocol Ratios
        syn_ack_ratio = round(float(syn_count) / float(ack_count + 1), 3)
        failed_conns = rst_count + max(0, syn_count - ack_count)
        failed_connection_ratio = round(min(1.0, float(failed_conns) / float(max(1, active_flows))), 3)

        # Rates & Volume
        packet_rate = round(float(total_pkts) / float(window_duration), 2)
        byte_rate = round(float(total_bytes) / float(window_duration), 2)
        mean_packet_size = round(float(total_bytes) / float(max(1, total_pkts)), 2)

        # Inter-Arrival Times (IAT)
        if len(flows) > 1:
            timestamps = [f.timestamp for f in flows]
            iats = np.diff(timestamps) * 1000.0  # convert to ms
            mean_iat = round(float(np.mean(iats)), 2)
            iat_variance = round(float(np.var(iats)), 2)
            max_iat = round(float(np.max(iats)), 2)
        else:
            mean_iat = flows[0].mean_iat
            iat_variance = float(flows[0].std_iat ** 2)
            max_iat = flows[0].max_iat

        # Port-Scan Heuristic [0.0 - 1.0]
        # Evaluates port dispersion per src_ip and SYN dominance
        ports_per_src = float(len(dst_ports)) / float(max(1, len(src_ips)))
        scan_component_1 = min(1.0, ports_per_src / 20.0)  # High port dispersion
        scan_component_2 = min(1.0, syn_ack_ratio / 5.0)   # High SYN-to-ACK imbalance
        scan_component_3 = min(1.0, failed_connection_ratio)
        port_scan_score = round(0.5 * scan_component_1 + 0.3 * scan_component_2 + 0.2 * scan_component_3, 3)

        # Packet-level options
        ttls = [f.ttl_mean for f in flows if f.ttl_mean is not None]
        ttl_mean = float(np.mean(ttls)) if ttls else None
        ttl_var = float(np.var(ttls)) if len(ttls) > 1 else None

        wins = [f.tcp_window_mean for f in flows if f.tcp_window_mean is not None]
        tcp_window_mean = float(np.mean(wins)) if wins else None

        retrans = [f.retransmissions for f in flows if f.retransmissions is not None]
        total_retrans = int(sum(retrans)) if retrans else None

        # Standard 16-feature vector for temporal LSTM ingestion
        raw_vector = [
            float(active_flows),
            float(len(src_ips)),
            float(len(dst_ips)),
            float(len(dst_ports)),
            float(connection_fan_out),
            float(syn_count),
            float(ack_count),
            float(rst_count),
            float(syn_ack_ratio),
            float(failed_connection_ratio),
            float(packet_rate),
            float(byte_rate),
            float(mean_packet_size),
            float(mean_iat),
            float(iat_variance),
            float(port_scan_score)
        ]

        return NetworkState(
            window_index=window_index,
            timestamp_start=round(timestamp_start, 3),
            timestamp_end=round(timestamp_end, 3),
            window_duration=window_duration,
            active_flows=active_flows,
            unique_src_ips=len(src_ips),
            unique_dst_ips=len(dst_ips),
            unique_dst_ports=len(dst_ports),
            connection_fan_out=connection_fan_out,
            syn_count=syn_count,
            ack_count=ack_count,
            rst_count=rst_count,
            syn_ack_ratio=syn_ack_ratio,
            failed_connection_ratio=failed_connection_ratio,
            packet_rate=packet_rate,
            byte_rate=byte_rate,
            mean_packet_size=mean_packet_size,
            mean_iat=mean_iat,
            iat_variance=iat_variance,
            max_iat=round(max_iat, 2),
            ttl_mean=round(ttl_mean, 2) if ttl_mean is not None else None,
            ttl_variance=round(ttl_var, 2) if ttl_var is not None else None,
            tcp_window_mean=round(tcp_window_mean, 2) if tcp_window_mean is not None else None,
            retransmissions=total_retrans,
            port_scan_score=port_scan_score,
            raw_feature_vector=raw_vector,
            is_demo=is_demo
        )
