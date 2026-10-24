"""
NetForeSight AI - Data Ingestion & Analysis Router
Handles network flow uploads (CSV/PCAP) and dataset statistical profiling.
"""

from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from pydantic import BaseModel

from backend.app.schemas.flow import FlowRecord, FlowAnalysisSummary
from backend.app.services.parser_csv import CSVParserService
from backend.app.core.config import settings

router = APIRouter(prefix="/data", tags=["Data Ingestion & Analysis"])


class UploadResponse(BaseModel):
    filename: str
    file_size_bytes: int
    flows_parsed: int
    summary: FlowAnalysisSummary
    message: str


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_200_OK)
async def upload_network_data(file: UploadFile = File(...)):
    """
    Ingests and validates untrusted network flow files (CSV/PCAP).
    Extracts flow records and returns structural statistical analysis.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")
        
    ext = "." + file.filename.split(".")[-1].lower() if "." in file.filename else ""
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format '{ext}'. Allowed formats: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )

    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        if len(content) > settings.MAX_FILE_SIZE_BYTES:
            raise HTTPException(status_code=400, detail=f"File exceeds limit of {settings.MAX_FILE_SIZE_BYTES // (1024*1024)}MB.")

        # Parse CSV flow format
        if ext == ".csv":
            flows, summary = CSVParserService.parse_csv_content(content, filename=file.filename)
        else:
            # PCAP support placeholder for Phase 4
            raise HTTPException(status_code=400, detail=f"Direct {ext} packet parsing will be enabled in Phase 4. Please provide a CSV flow export.")

        return UploadResponse(
            filename=file.filename,
            file_size_bytes=len(content),
            flows_parsed=len(flows),
            summary=summary,
            message="File successfully parsed and validated."
        )
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


class AnalyzeRequest(BaseModel):
    flows: List[FlowRecord]


@router.post("/analyze", response_model=FlowAnalysisSummary, status_code=status.HTTP_200_OK)
async def analyze_flows(payload: AnalyzeRequest):
    """Computes comprehensive flow-level analytics from parsed flow records."""
    if not payload.flows:
        raise HTTPException(status_code=400, detail="No flows provided in payload.")
    return CSVParserService._compute_summary(payload.flows)
