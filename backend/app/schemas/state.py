"""
NetForeSight AI - Network State Schemas
Data models for timestamped network state vectors S(t) representing temporal aggregations.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class NetworkState(BaseModel):
    """
    Timestamped Network State Vector S(t).
    Represents aggregated network metrics over a configured time window (default: 5.0 seconds).
    """
    window_index: int = Field(..., description="Monotonically increasing sequence index t of the state window")
    timestamp_start: float = Field(..., description="Window start epoch/relative timestamp in seconds")
    timestamp_end: float = Field(..., description="Window end epoch/relative timestamp in seconds")
    window_duration: float = Field(5.0, description="Duration of this window in seconds")
    
    # Topological Features
    active_flows: int = Field(0, description="Count of distinct active flows observed in this window")
    unique_src_ips: int = Field(0, description="Count of unique source IPs")
    unique_dst_ips: int = Field(0, description="Count of unique destination IPs")
    unique_dst_ports: int = Field(0, description="Count of unique destination ports targeted")
    connection_fan_out: float = Field(0.0, description="Ratio of destination endpoints to source endpoints")
    
    # Protocol & TCP Flag Features
    syn_count: int = Field(0, description="Total SYN packets/flows")
    ack_count: int = Field(0, description="Total ACK packets/flows")
    rst_count: int = Field(0, description="Total RST packets/flows")
    syn_ack_ratio: float = Field(0.0, description="SYN / (ACK + 1) ratio to detect half-open sweeps")
    failed_connection_ratio: float = Field(0.0, description="Ratio of RST or unacknowledged SYNs to total attempts")
    
    # Volume & Timing Features
    packet_rate: float = Field(0.0, description="Packets per second in this window")
    byte_rate: float = Field(0.0, description="Bytes per second in this window")
    mean_packet_size: float = Field(0.0, description="Average packet payload/size in bytes")
    mean_iat: float = Field(0.0, description="Mean flow inter-arrival time (ms)")
    iat_variance: float = Field(0.0, description="Variance of flow inter-arrival times (ms^2)")
    max_iat: float = Field(0.0, description="Maximum inter-arrival time in window (ms)")
    
    # Packet-Level / Anomaly Distributions
    ttl_mean: Optional[float] = Field(None, description="Mean IP TTL (available from PCAP or flow metadata)")
    ttl_variance: Optional[float] = Field(None, description="IP TTL variance across flows")
    tcp_window_mean: Optional[float] = Field(None, description="Mean TCP window size")
    retransmissions: Optional[int] = Field(None, description="Retransmitted packet count")
    port_scan_score: float = Field(0.0, description="Heuristic/probabilistic metric [0.0 - 1.0] of port scanning activity")
    
    # Optional raw normalized tensor representation for ML pipeline
    raw_feature_vector: Optional[List[float]] = Field(
        None, 
        description="Standardized numerical array of state features for direct ingestion into PyTorch LSTM"
    )
    
    # Metadata for verification
    is_demo: bool = Field(False, description="True if state was generated from demo/synthetic scenario")


class NetworkStateSeries(BaseModel):
    """A contiguous temporal sequence of Network States [S(t-W), ..., S(t)]."""
    window_size_seconds: float
    total_states: int
    states: List[NetworkState]
    summary_metrics: Dict[str, Any] = {}
