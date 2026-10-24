"""
NetForeSight AI - Flow Schemas
Data models for network flow records and aggregated statistical summaries.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class FlowRecord(BaseModel):
    """Represents a single parsed bidirectional or unidirectional network flow."""
    flow_id: Optional[str] = None
    timestamp: float = Field(..., description="Epoch timestamp or relative second offset of the flow start")
    src_ip: str = Field(..., description="Source IPv4 or IPv6 address")
    dst_ip: str = Field(..., description="Destination IPv4 or IPv6 address")
    src_port: int = Field(..., ge=0, le=65535, description="Source port number")
    dst_port: int = Field(..., ge=0, le=65535, description="Destination port number")
    protocol: str = Field("TCP", description="Transport layer protocol (TCP, UDP, ICMP, etc.)")
    
    # Volume and duration
    duration: float = Field(0.0, ge=0.0, description="Flow duration in seconds")
    packet_count: int = Field(1, ge=1, description="Total packet count in flow")
    byte_count: int = Field(0, ge=0, description="Total bytes in flow")
    
    # TCP Flags
    syn_flag: bool = False
    ack_flag: bool = False
    rst_flag: bool = False
    fin_flag: bool = False
    psh_flag: bool = False
    urg_flag: bool = False
    
    # Timing & IAT
    mean_iat: float = Field(0.0, ge=0.0, description="Mean inter-arrival time in milliseconds")
    std_iat: float = Field(0.0, ge=0.0, description="Standard deviation of inter-arrival time")
    max_iat: float = Field(0.0, ge=0.0, description="Maximum inter-arrival time in milliseconds")
    
    # Optional packet-level features (when extracted from PCAP)
    ttl_mean: Optional[float] = None
    tcp_window_mean: Optional[float] = None
    retransmissions: Optional[int] = None
    
    # Ground truth dataset label
    label: Optional[str] = "Benign"


class FlowAnalysisSummary(BaseModel):
    """High-level summary analytics of parsed flow traffic."""
    total_flows: int
    unique_src_ips: int
    unique_dst_ips: int
    unique_dst_ports: int
    time_span_seconds: float
    total_packets: int
    total_bytes: int
    protocol_distribution: Dict[str, int]
    top_source_ips: List[Dict[str, Any]]
    top_destination_ports: List[Dict[str, Any]]
    mean_flow_rate_per_sec: float
    mean_byte_rate_per_sec: float
