"""
Unit tests for CSVParserService (including CSE-CIC-IDS2018 compatibility fixes)
"""

import pytest
from backend.app.services.parser_csv import CSVParserService


def test_parse_valid_csv():
    csv_data = (
        "timestamp,src_ip,dst_ip,src_port,dst_port,protocol,duration,packet_count,byte_count,syn_flag,ack_flag,rst_flag\n"
        "1.0,192.168.1.10,192.168.1.100,45000,80,TCP,0.05,5,1500,1,1,0\n"
        "2.5,192.168.1.10,192.168.1.100,45002,443,TCP,0.02,3,800,1,1,0\n"
        "3.8,192.168.1.15,192.168.1.100,51234,80,TCP,0.1,8,3200,1,1,0\n"
    ).encode("utf-8")

    flows, summary = CSVParserService.parse_csv_content(csv_data, "test.csv")
    assert len(flows) == 3
    assert flows[0].src_ip == "192.168.1.10"
    assert flows[0].dst_port == 80
    assert flows[0].syn_flag is True
    assert flows[0].byte_count == 1500
    assert flows[0].label == "Benign"
    assert summary.total_flows == 3
    assert summary.unique_src_ips == 2
    assert summary.unique_dst_ips == 1
    assert summary.unique_dst_ports == 2


def test_parse_cic_ids_bidirectional_and_label():
    # Fix 1 & Fix 2: Label parsing + Bidirectional packet and byte summation
    csv_data = (
        "Timestamp,Dst Port,Protocol,Flow Duration,Tot Fwd Pkts,Tot Bwd Pkts,TotLen Fwd Pkts,TotLen Bwd Pkts,SYN Flag Cnt,ACK Flag Cnt,RST Flag Cnt,Flow IAT Mean,Label\n"
        "10.0,80,6,50000,10,15,2000,4500,1,1,0,5.2,Infilteration\n"
        "12.0,22,6,120000,4,2,600,300,1,0,1,12.5,PortScan\n"
    ).encode("utf-8")

    flows, summary = CSVParserService.parse_csv_content(csv_data, "cic.csv")
    assert len(flows) == 2
    
    # Flow 0: Tot Fwd (10) + Tot Bwd (15) = 25 pkts; TotLen Fwd (2000) + TotLen Bwd (4500) = 6500 bytes
    assert flows[0].dst_port == 80
    assert flows[0].protocol == "TCP"
    assert flows[0].packet_count == 25
    assert flows[0].byte_count == 6500
    assert flows[0].label == "Infilteration"

    # Flow 1: Tot Fwd (4) + Tot Bwd (2) = 6 pkts; TotLen Fwd (600) + TotLen Bwd (300) = 900 bytes
    assert flows[1].dst_port == 22
    assert flows[1].packet_count == 6
    assert flows[1].byte_count == 900
    assert flows[1].rst_flag is True
    assert flows[1].label == "PortScan"


def test_parse_cic_ids_infinity_and_nan():
    # Fix 3: Robust handling of Infinity and NaN values in CSV
    csv_data = (
        "Timestamp,Dst Port,Protocol,Flow Duration,Tot Fwd Pkts,TotLen Fwd Pkts,Flow Byts/s,Flow Pkts/s,Flow IAT Mean,Label\n"
        "10.0,80,6,0,5,500,Infinity,Infinity,NaN,Benign\n"
        "12.0,443,6,1000,2,200,NaN,NaN,5.0,Benign\n"
    ).encode("utf-8")

    flows, summary = CSVParserService.parse_csv_content(csv_data, "inf_nan.csv")
    assert len(flows) == 2
    assert flows[0].packet_count == 5
    assert flows[0].byte_count == 500
    assert flows[1].packet_count == 2


def test_parse_cic_ids_dayfirst_timestamps():
    # Fix 4: Day-first timestamp string parsing (e.g. 15/02/2018 01:12:35)
    csv_data = (
        "Timestamp,Dst Port,Protocol,Flow Duration,Tot Fwd Pkts,TotLen Fwd Pkts,Label\n"
        "15/02/2018 01:12:35,80,6,10000,5,500,Benign\n"
        "15/02/2018 01:12:40,80,6,10000,6,600,Benign\n"
    ).encode("utf-8")

    flows, summary = CSVParserService.parse_csv_content(csv_data, "timestamps.csv")
    assert len(flows) == 2
    assert flows[0].timestamp > 1500000000  # Valid 2018 epoch timestamp
    assert flows[1].timestamp > flows[0].timestamp
    assert flows[1].timestamp - flows[0].timestamp == 5.0


def test_parse_empty_csv():
    with pytest.raises(ValueError, match="empty"):
        CSVParserService.parse_csv_content(b"", "empty.csv")
