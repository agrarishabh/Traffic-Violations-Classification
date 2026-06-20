"""
app/routers/analyze.py
=======================
POST /analyze  — upload an image, run full CV pipeline, return violations.

Pipeline per request:
  1. Validate + read uploaded image bytes
  2. Decode to numpy BGR array
  3. Run ViolationDetector.detect_all()
  4. Run LicensePlateOCR.extract()
  5. Run EvidenceGenerator.generate()
  6. Return AnalyzeResponse JSON
"""

import io
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse

from app.models.schemas import (
    AnalyzeResponse, ViolationOut, PlateOut, DetectionSummaryOut,
    DetectionOut,
)

router = APIRouter(prefix="/analyze", tags=["Analysis"])

# Supported image MIME types
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/bmp",
                 "image/webp", "image/tiff"}
MAX_FILE_SIZE_MB = 10


# ══════════════════════════════════════════════════════════════
# POST /analyze
# ══════════════════════════════════════════════════════════════

@router.post(
    "",
    response_model=AnalyzeResponse,
    summary="Analyze a traffic image for violations",
    description="""
Upload a traffic image (JPG/PNG/BMP).

The system will:
- Detect vehicles, persons, and traffic lights
- Check for 7 violation types (helmet, seatbelt, triple riding, red light, etc.)
- Recognize license plate numbers via OCR
- Generate and save annotated evidence image + metadata JSON

Returns a structured JSON with all violations found, confidence scores, and total fines.
""",
)
async def analyze_image(
    request: Request,
    file: UploadFile = File(
        ...,
        description="Traffic image file (JPG, PNG, BMP, WebP — max 10 MB)"
    ),
):
    # ── Validate content type ─────────────────────────────────
    content_type = file.content_type or ""
    if content_type not in ALLOWED_TYPES:
        # Also accept if filename ends in known image extension
        filename = file.filename or ""
        if not any(filename.lower().endswith(ext)
                   for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]):
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type: {content_type}. "
                       f"Upload JPG, PNG, BMP, or WebP.")

    # ── Read bytes ────────────────────────────────────────────
    raw_bytes = await file.read()

    if len(raw_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB} MB.")

    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    # ── Decode image ──────────────────────────────────────────
    np_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    image  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=422,
            detail="Could not decode image. File may be corrupted.")

    # ── Get models from app state ─────────────────────────────
    state = request.app.state
    if not getattr(state, "models_loaded", False):
        raise HTTPException(
            status_code=503,
            detail="Models are still loading. Please retry in a few seconds.")

    vd  = state.violation_detector
    ocr = state.ocr
    gen = state.evidence_generator

    # ── Run pipeline ──────────────────────────────────────────
    try:
        # Step 1: Violation detection (includes preprocessing + vehicle detect)
        viol_result = vd.detect_all(image)
        det_result  = viol_result.detection_result

        # Step 2: License plate OCR
        vehicle_bboxes = [v.bbox for v in det_result.vehicles] \
                         if det_result else []
        plates = ocr.extract(image, vehicle_bboxes=vehicle_bboxes)

        # Step 3: Assign plate numbers to violations
        plate_no = plates[0].plate_text_clean if plates else None
        for v in viol_result.violations:
            if v.plate_number is None and plate_no:
                v.plate_number = plate_no

        # Step 4: Generate evidence
        image_name = file.filename or "uploaded_image.jpg"
        package    = gen.generate(
            image            = image,
            image_path       = image_name,
            detection_result = det_result,
            violation_result = viol_result,
            plate_results    = plates,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(e)}")

    # ── Build annotated image URL ─────────────────────────────
    annotated_url = None
    if package.annotated_path:
        from pathlib import Path
        from config import EVIDENCE_DIR
        try:
            rel = Path(package.annotated_path).relative_to(EVIDENCE_DIR)
            annotated_url = f"/evidence/image/{rel.as_posix()}"
        except ValueError:
            annotated_url = None

    # ── Build detection summary ───────────────────────────────
    detection_summary = None
    if det_result:
        detection_summary = DetectionSummaryOut(
            vehicle_count = det_result.vehicle_count,
            person_count  = det_result.person_count,
            light_count   = len(det_result.traffic_lights),
            inference_ms  = det_result.inference_ms,
            vehicles      = [DetectionOut(**d.to_dict())
                             for d in det_result.vehicles],
            persons       = [DetectionOut(**d.to_dict())
                             for d in det_result.persons],
        )

    # ── Serialize violations ──────────────────────────────────
    violations_out = [
        ViolationOut(
            violation_type = v.violation_type,
            display_name   = v.display_name,
            confidence     = v.confidence,
            bbox           = v.bbox,
            vehicle_bbox   = v.vehicle_bbox,
            plate_number   = v.plate_number,
            description    = v.description,
            fine_amount    = v.fine_amount,
        )
        for v in viol_result.violations
    ]

    plates_out = [
        PlateOut(
            plate_text       = p.plate_text,
            plate_text_clean = p.plate_text_clean,
            confidence       = p.confidence,
            bbox             = p.bbox,
            is_valid_format  = p.is_valid_format,
            detection_method = p.detection_method,
        )
        for p in plates
    ]

    return AnalyzeResponse(
        evidence_id          = package.evidence_id,
        timestamp            = package.timestamp,
        has_violations       = viol_result.has_violations,
        violation_count      = len(violations_out),
        total_fine_inr       = viol_result.total_fines,
        processing_ms        = viol_result.processing_ms,
        violations           = violations_out,
        plates               = plates_out,
        detection            = detection_summary,
        annotated_image_url  = annotated_url,
        metadata_path        = package.metadata_path,
        summary              = viol_result.summary,
    )
