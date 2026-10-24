"""
NetForeSight AI - CSV Flow Parser & Data Ingestion Service
Safely parses and validates CSV flow records (supporting CSE-CIC-IDS2018 & NetFlow formats).
"""

import io
from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np

from backend.app.schemas.flow import FlowRecord, FlowAnalysisSummary
from backend.app.core.config import settings


# Mapping dictionaries for diverse dataset column naming conventions
COLUMN_ALIASES: Dict[str, List[str]] = {
    "timestamp": ["timestamp", "time", "flow_start", "start_time", "ts"],
    "src_ip": ["src_ip", "srcip", "source_ip", "source ip", "src address", "src_addr", "src ip"],
    "dst_ip": ["dst_ip", "dstip", "destination_ip", "destination ip", "dst address", "dst_addr", "dst ip"],
    "src_port": ["src_port", "srcport", "source_port", "source port", "src_pt", "src port"],
    "dst_port": ["dst_port", "dstport", "destination_port", "destination port", "dst port", "dst_pt"],
    "protocol": ["protocol", "proto", "transport_proto"],
    "duration": ["duration", "flow_duration", "flow duration", "dur"],
    "packet_count": ["packet_count", "total_packets", "packets", "pkts"],
    "tot_fwd_pkts": ["tot fwd pkts", "tot_fwd_pkts", "total fwd packets", "fwd packets"],
    "tot_bwd_pkts": ["tot bwd pkts", "tot_bwd_pkts", "total bwd packets", "bwd packets"],
    "byte_count": ["byte_count", "total_bytes", "bytes", "vol"],
    "totlen_fwd_pkts": ["totlen fwd pkts", "totlen_fwd_pkts", "total len fwd packets", "fwd_bytes"],
    "totlen_bwd_pkts": ["totlen bwd pkts", "totlen_bwd_pkts", "total len bwd packets", "bwd_bytes"],
    "syn_flag": ["syn_flag", "syn flag cnt", "syn_count", "syn", "syn_cnt"],
    "ack_flag": ["ack_flag", "ack flag cnt", "ack_count", "ack", "ack_cnt"],
    "rst_flag": ["rst_flag", "rst flag cnt", "rst_count", "rst", "rst_cnt"],
    "mean_iat": ["mean_iat", "flow iat mean", "iat_mean", "mean iat"],
    "std_iat": ["std_iat", "flow iat std", "iat_std", "std iat"],
    "max_iat": ["max_iat", "flow iat max", "iat_max", "max iat"],
    "label": ["label", "attack_label", "class", "target", "attack"],
}


def _resolve_column(df_columns: List[str], target_key: str) -> str | None:
    """Finds matching column name from aliases (case-insensitive and trimmed)."""
    normalized_cols = {c.strip().lower(): c for c in df_columns}
    aliases = COLUMN_ALIASES.get(target_key, [target_key])
    for alias in aliases:
        if alias.lower() in normalized_cols:
            return normalized_cols[alias.lower()]
    return None


class CSVParserService:
    """Service to parse, validate, and summarize network flow CSV files."""
    
    @staticmethod
    def parse_csv_content(csv_bytes: bytes, filename: str = "upload.csv") -> Tuple[List[FlowRecord], FlowAnalysisSummary]:
        """
        Parses raw CSV bytes into validated FlowRecord objects and returns an analytical summary.
        Safely handles Infinity/NaN values and bidirectional packet/byte summation for CSE-CIC-IDS2018.
        """
        if len(csv_bytes) == 0:
            raise ValueError("Uploaded file is empty.")
            
        if len(csv_bytes) > settings.MAX_FILE_SIZE_BYTES:
            raise ValueError(f"File size exceeds maximum permitted limit of {settings.MAX_FILE_SIZE_BYTES // (1024*1024)}MB.")

        try:
            # Read first few bytes to check format
            df = pd.read_csv(io.BytesIO(csv_bytes), encoding="utf-8", low_memory=False)
        except Exception as e:
            try:
                # Fallback encoding
                df = pd.read_csv(io.BytesIO(csv_bytes), encoding="latin-1", low_memory=False)
            except Exception as e2:
                raise ValueError(f"Unable to parse CSV file: {str(e2)}")

        if df.empty:
            raise ValueError("CSV contains no rows.")

        # Clean Infinity and NaN values (Fix 3: Robust NaN/Inf handling for CICFlowMeter)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        # Resolve essential column mappings
        col_map: Dict[str, str] = {}
        for target in ["timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "protocol", 
                       "duration", "packet_count", "tot_fwd_pkts", "tot_bwd_pkts", 
                       "byte_count", "totlen_fwd_pkts", "totlen_bwd_pkts", 
                       "syn_flag", "ack_flag", "rst_flag", "mean_iat", "std_iat", "max_iat", "label"]:
            resolved = _resolve_column(list(df.columns), target)
            if resolved:
                col_map[target] = resolved

        flow_records: List[FlowRecord] = []
        
        # Iterate and construct typed records
        for idx, row in df.iterrows():
            # Derive timestamp (Fix 4: Robust day-first datetime parsing for CSE-CIC-IDS2018)
            raw_ts = row[col_map["timestamp"]] if "timestamp" in col_map else idx * 0.1
            try:
                if isinstance(raw_ts, str) and ("-" in raw_ts or "/" in raw_ts):
                    ts_val = pd.to_datetime(raw_ts, dayfirst=True, format="mixed").timestamp()
                else:
                    ts_val = float(raw_ts)
            except Exception:
                try:
                    ts_val = pd.to_datetime(raw_ts).timestamp()
                except Exception:
                    ts_val = float(idx) * 0.1

            src_ip = str(row[col_map["src_ip"]]) if "src_ip" in col_map else f"192.168.1.{(idx % 250) + 1}"
            dst_ip = str(row[col_map["dst_ip"]]) if "dst_ip" in col_map else "192.168.1.100"
            
            try:
                src_port = int(float(row[col_map["src_port"]])) if "src_port" in col_map else (1024 + (idx % 60000))
            except Exception:
                src_port = 8080
                
            try:
                dst_port = int(float(row[col_map["dst_port"]])) if "dst_port" in col_map else 80
            except Exception:
                dst_port = 80

            protocol = str(row[col_map["protocol"]]).upper() if "protocol" in col_map else "TCP"
            if protocol in ["6", "6.0"]:
                protocol = "TCP"
            elif protocol in ["17", "17.0"]:
                protocol = "UDP"
            elif protocol in ["1", "1.0"]:
                protocol = "ICMP"

            try:
                duration = float(row[col_map["duration"]]) if "duration" in col_map else 0.0
                if duration > 100000:
                    # Flow duration in microseconds (common in CIC datasets) -> convert to seconds
                    duration = duration / 1_000_000.0
            except Exception:
                duration = 0.0

            # Fix 2: Bidirectional packet and byte aggregation (Tot Fwd + Tot Bwd)
            if "tot_fwd_pkts" in col_map or "tot_bwd_pkts" in col_map:
                fwd_pkts = float(row[col_map["tot_fwd_pkts"]]) if "tot_fwd_pkts" in col_map else 0.0
                bwd_pkts = float(row[col_map["tot_bwd_pkts"]]) if "tot_bwd_pkts" in col_map else 0.0
                packet_count = max(1, int(fwd_pkts + bwd_pkts))
            elif "packet_count" in col_map:
                try:
                    packet_count = max(1, int(float(row[col_map["packet_count"]])))
                except Exception:
                    packet_count = 1
            else:
                packet_count = 1

            if "totlen_fwd_pkts" in col_map or "totlen_bwd_pkts" in col_map:
                fwd_bytes = float(row[col_map["totlen_fwd_pkts"]]) if "totlen_fwd_pkts" in col_map else 0.0
                bwd_bytes = float(row[col_map["totlen_bwd_pkts"]]) if "totlen_bwd_pkts" in col_map else 0.0
                byte_count = max(0, int(fwd_bytes + bwd_bytes))
            elif "byte_count" in col_map:
                try:
                    byte_count = max(0, int(float(row[col_map["byte_count"]])))
                except Exception:
                    byte_count = packet_count * 64
            else:
                byte_count = packet_count * 64

            # Flags
            syn_flag = bool(float(row[col_map["syn_flag"]]) > 0) if "syn_flag" in col_map else False
            ack_flag = bool(float(row[col_map["ack_flag"]]) > 0) if "ack_flag" in col_map else False
            rst_flag = bool(float(row[col_map["rst_flag"]]) > 0) if "rst_flag" in col_map else False

            # IAT stats
            mean_iat = float(row[col_map["mean_iat"]]) if "mean_iat" in col_map else 0.0
            std_iat = float(row[col_map["std_iat"]]) if "std_iat" in col_map else 0.0
            max_iat = float(row[col_map["max_iat"]]) if "max_iat" in col_map else 0.0

            # Fix 1: Label parsing
            raw_label = str(row[col_map["label"]]).strip() if "label" in col_map else "Benign"
            label = raw_label if raw_label and raw_label != "0" and raw_label != "0.0" else "Benign"

            record = FlowRecord(
                flow_id=f"flow_{idx}",
                timestamp=ts_val,
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=max(0, min(65535, src_port)),
                dst_port=max(0, min(65535, dst_port)),
                protocol=protocol,
                duration=max(0.0, duration),
                packet_count=packet_count,
                byte_count=byte_count,
                syn_flag=syn_flag,
                ack_flag=ack_flag,
                rst_flag=rst_flag,
                mean_iat=mean_iat,
                std_iat=std_iat,
                max_iat=max_iat,
                label=label,
            )
            flow_records.append(record)

        # Sort flows chronologically by timestamp
        flow_records.sort(key=lambda r: r.timestamp)

        # Compute summary
        summary = CSVParserService._compute_summary(flow_records)
        return flow_records, summary

    @staticmethod
    def _compute_summary(flows: List[FlowRecord]) -> FlowAnalysisSummary:
        """Calculates statistical flow summary metrics."""
        if not flows:
            return FlowAnalysisSummary(
                total_flows=0,
                unique_src_ips=0,
                unique_dst_ips=0,
                unique_dst_ports=0,
                time_span_seconds=0.0,
                total_packets=0,
                total_bytes=0,
                protocol_distribution={},
                top_source_ips=[],
                top_destination_ports=[],
                mean_flow_rate_per_sec=0.0,
                mean_byte_rate_per_sec=0.0
            )

        src_ips: Dict[str, int] = {}
        dst_ips: set = set()
        dst_ports: Dict[int, int] = {}
        protocols: Dict[str, int] = {}
        total_pkts = 0
        total_bytes = 0

        min_t = flows[0].timestamp
        max_t = flows[-1].timestamp
        time_span = max(0.001, max_t - min_t)

        for f in flows:
            src_ips[f.src_ip] = src_ips.get(f.src_ip, 0) + 1
            dst_ips.add(f.dst_ip)
            dst_ports[f.dst_port] = dst_ports.get(f.dst_port, 0) + 1
            protocols[f.protocol] = protocols.get(f.protocol, 0) + 1
            total_pkts += f.packet_count
            total_bytes += f.byte_count

        top_src = sorted([{"ip": ip, "count": cnt} for ip, cnt in src_ips.items()], key=lambda x: x["count"], reverse=True)[:5]
        top_ports = sorted([{"port": str(p), "count": cnt} for p, cnt in dst_ports.items()], key=lambda x: x["count"], reverse=True)[:5]

        return FlowAnalysisSummary(
            total_flows=len(flows),
            unique_src_ips=len(src_ips),
            unique_dst_ips=len(dst_ips),
            unique_dst_ports=len(dst_ports),
            time_span_seconds=round(time_span, 2),
            total_packets=total_pkts,
            total_bytes=total_bytes,
            protocol_distribution=protocols,
            top_source_ips=top_src,
            top_destination_ports=top_ports,
            mean_flow_rate_per_sec=round(len(flows) / time_span, 2),
            mean_byte_rate_per_sec=round(total_bytes / time_span, 2),
        )
