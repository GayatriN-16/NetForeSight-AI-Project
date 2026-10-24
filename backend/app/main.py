import os
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.core.config import settings
from backend.app.api.data import router as data_router
from backend.app.api.states import router as states_router
from backend.app.api.forecast import router as forecast_router
from backend.app.api.evaluation import router as evaluation_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration for frontend SOC dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permits local React dev servers on port 3000 / 5173
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(data_router, prefix=settings.API_PREFIX)
app.include_router(states_router, prefix=settings.API_PREFIX)
app.include_router(forecast_router, prefix=settings.API_PREFIX)
app.include_router(evaluation_router, prefix=settings.API_PREFIX)

# Static file serving for React SOC Frontend
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", tags=["SOC Interface"])
@app.get("/dashboard", tags=["SOC Interface"])
async def serve_dashboard():
    """Serves the interactive React SOC Frontend Dashboard."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "NetForeSight AI Backend API Active. React frontend static build pending."}


@app.get("/api/health", tags=["System Health"])
async def health_check():
    """Returns system operational status and metadata."""
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "tagline": "Predict the Attack. Stop the Next Move.",
        "problem_statement": "SIH26153 (NTRO)",
        "timestamp": time.time()
    }
