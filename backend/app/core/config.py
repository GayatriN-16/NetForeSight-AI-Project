"""
NetForeSight AI - Backend Configuration
Defines application settings, security upload boundaries, and default state builder parameters.
"""

from typing import List
from pydantic import BaseModel


class Settings(BaseModel):
    PROJECT_NAME: str = "NetForeSight AI"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    DESCRIPTION: str = "AI Based Network Attack Forecasting from Network Traffic Data (SIH26153 — NTRO)"
    
    # State builder defaults
    DEFAULT_WINDOW_SIZE: float = 5.0  # seconds
    DEFAULT_FORECAST_STEPS: int = 3   # K=3 default
    
    # Security and upload boundaries
    MAX_FILE_SIZE_BYTES: int = 200 * 1024 * 1024  # 200 MB
    ALLOWED_EXTENSIONS: List[str] = [".csv", ".pcap", ".pcapng"]
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]


settings = Settings()
