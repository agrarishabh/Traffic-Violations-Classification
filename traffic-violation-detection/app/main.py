"""
app/main.py
===========
FastAPI Application Entry Point — Phase 7

Endpoints:
  POST   /analyze                     — upload image, detect violations
  GET    /violations                  — list violations (paginated + filtered)
  GET    /violations/{id}             — single violation by DB id
  GET    /violations/by-plate/{plate} — violations for a plate number
  DELETE /violations/{id}             — delete one record
  GET    /stats                       — aggregate statistics
  GET    /evidence/image/{path}       — serve annotated image file
  GET    /health                      — service health check
  GET    /docs                        — Swagger UI (auto-generated)

HOW TO START:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Then open:  http://localhost:8000/docs
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    API_TITLE, API_VERSION, API_HOST, API_PORT, EVIDENCE_DIR
)
from core.database import init_db
from app.routers import analyze, violations, stats
from app.models.schemas import HealthResponse


# ══════════════════════════════════════════════════════════════
# LIFESPAN — load models once at startup, release at shutdown
# ══════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once when server starts.
    Loads all ML models into app.state so they're shared across requests.
    Heavy models (YOLO, EasyOCR) are loaded here — NOT per request.
    """
    print("\n" + "="*50)
    print("  Traffic Violation Detection API — Starting")
    print("="*50)

    # Initialize SQLite database (creates tables if they don't exist)
    print("  Initializing database ...")
    init_db()

    # Load violation detector (includes vehicle detector)
    print("  Loading ViolationDetector + YOLOv8 ...")
    from core.violation_detector import ViolationDetector
    vd = ViolationDetector()
    vd.load_models()
    app.state.violation_detector = vd

    # Load OCR
    print("  Loading EasyOCR ...")
    from core.ocr import LicensePlateOCR
    ocr = LicensePlateOCR(use_gpu=False)
    ocr.load_models()
    app.state.ocr = ocr

    # Load evidence generator
    from core.evidence_generator import EvidenceGenerator
    app.state.evidence_generator = EvidenceGenerator(save_to_db=True)

    app.state.models_loaded = True
    app.state.startup_time  = datetime.now().isoformat()

    print("="*50)
    print("  All models loaded — API is ready!")
    print(f"  Swagger UI: http://{API_HOST}:{API_PORT}/docs")
    print("="*50 + "\n")

    yield   # ← server runs here

    # Cleanup on shutdown (optional)
    print("  Shutting down — releasing models ...")
    app.state.models_loaded = False


# ══════════════════════════════════════════════════════════════
# APP INSTANCE
# ══════════════════════════════════════════════════════════════

app = FastAPI(
    title       = API_TITLE,
    version     = API_VERSION,
    description = """
## Traffic Violation Detection System API

AI-powered traffic image analysis using YOLOv8 and EasyOCR.

### What it does
- Detects vehicles, persons, and traffic lights in uploaded images
- Identifies 7 types of traffic violations with confidence scores
- Recognizes Indian license plate numbers via OCR
- Generates annotated evidence images with timestamps
- Stores all records in a searchable database

### Violation types detected
| Type | Description |
|------|-------------|
| helmet_non_compliance | Rider without helmet |
| seatbelt_non_compliance | Driver without seatbelt |
| triple_riding | More than 2 persons on two-wheeler |
| red_light_violation | Vehicle runs red light |
| stop_line_violation | Vehicle crosses stop line |
| wrong_side_driving | Vehicle on wrong side |
| illegal_parking | Vehicle parked in restricted zone |
""",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)


# ══════════════════════════════════════════════════════════════
# MIDDLEWARE
# ══════════════════════════════════════════════════════════════

# CORS — allow the Streamlit dashboard (port 8501) and any localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # tighten to specific origins in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ══════════════════════════════════════════════════════════════
# ROUTERS
# ══════════════════════════════════════════════════════════════

app.include_router(analyze.router)
app.include_router(violations.router)
app.include_router(stats.router)


# ══════════════════════════════════════════════════════════════
# EVIDENCE IMAGE SERVING
# ══════════════════════════════════════════════════════════════

@app.get(
    "/evidence/image/{file_path:path}",
    tags=["Evidence"],
    summary="Serve an annotated evidence image",
    response_class=FileResponse,
)
async def serve_evidence_image(file_path: str):
    """
    Returns the annotated JPG image for a given evidence ID.

    The `file_path` comes from the `annotated_image_url` field in
    the /analyze response.
    """
    full_path = EVIDENCE_DIR / file_path
    if not full_path.exists():
        return JSONResponse(
            status_code=404,
            content={"detail": f"Evidence image not found: {file_path}"})
    return FileResponse(str(full_path), media_type="image/jpeg")


# ══════════════════════════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════════════════════════

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="API health check",
)
async def health_check():
    return HealthResponse(
        status        = "ok",
        models_loaded = getattr(app.state, "models_loaded", False),
        version       = API_VERSION,
        timestamp     = datetime.now().isoformat(),
    )


# ══════════════════════════════════════════════════════════════
# ROOT
# ══════════════════════════════════════════════════════════════

@app.get("/", tags=["System"], include_in_schema=False)
async def root():
    return {
        "name":        API_TITLE,
        "version":     API_VERSION,
        "docs":        "/docs",
        "health":      "/health",
        "analyze":     "POST /analyze",
        "violations":  "GET  /violations",
        "stats":       "GET  /stats",
    }
