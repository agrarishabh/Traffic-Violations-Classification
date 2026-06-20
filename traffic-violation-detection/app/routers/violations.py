"""
app/routers/violations.py
==========================
Violation record CRUD endpoints.

GET  /violations              — paginated list with optional filters
GET  /violations/{id}         — single violation record by DB id
GET  /violations/by-plate/{plate}  — all violations for a plate number
DELETE /violations/{id}       — delete one record
DELETE /violations/all        — delete all records (admin use)
"""

import json
from typing import Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from core.database import (
    get_db, get_violations, get_violation_by_id,
    delete_all_violations, ViolationRecord,
)
from app.models.schemas import (
    ViolationRecordOut, PaginatedViolations,
)
from config import VIOLATION_DISPLAY_NAMES

router = APIRouter(prefix="/violations", tags=["Violations"])


# ── Serializer helper ─────────────────────────────────────────
def _serialize(record: ViolationRecord) -> ViolationRecordOut:
    """Convert a SQLAlchemy ViolationRecord to Pydantic output schema."""
    bbox = None
    if record.bbox:
        try:
            bbox = json.loads(record.bbox)
        except Exception:
            bbox = None
    return ViolationRecordOut(
        id             = record.id,
        evidence_id    = record.evidence_id,
        violation_type = record.violation_type,
        display_name   = record.display_name,
        confidence     = record.confidence,
        bbox           = bbox,
        plate_number   = record.plate_number,
        fine_amount    = record.fine_amount,
        description    = record.description or "",
        detected_at    = record.detected_at.isoformat()
                         if record.detected_at else "",
        image_filename = record.image_filename,
        annotated_path = record.annotated_path,
    )


# ══════════════════════════════════════════════════════════════
# GET /violations
# ══════════════════════════════════════════════════════════════

@router.get(
    "",
    response_model=PaginatedViolations,
    summary="List all violations with optional filters",
)
def list_violations(
    page:           int            = Query(1,    ge=1,  description="Page number"),
    page_size:      int            = Query(20,   ge=1,  le=100, description="Records per page"),
    violation_type: Optional[str]  = Query(None, description="Filter by violation type"),
    plate_number:   Optional[str]  = Query(None, description="Filter by plate number (partial match)"),
    date_from:      Optional[str]  = Query(None, description="Filter from date (YYYY-MM-DD)"),
    date_to:        Optional[str]  = Query(None, description="Filter to date (YYYY-MM-DD)"),
    db: Session = Depends(get_db),
):
    """
    Returns paginated list of violation records.

    **Filter examples:**
    - `?violation_type=helmet_non_compliance`
    - `?plate_number=MH12`
    - `?date_from=2026-06-01&date_to=2026-06-30`
    - `?violation_type=triple_riding&page=2&page_size=10`
    """
    # Parse date filters
    dt_from = dt_to = None
    if date_from:
        try:
            dt_from = datetime.strptime(date_from, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(400, "date_from must be YYYY-MM-DD")
    if date_to:
        try:
            dt_to = datetime.strptime(date_to, "%Y-%m-%d").replace(
                hour=23, minute=59, second=59)
        except ValueError:
            raise HTTPException(400, "date_to must be YYYY-MM-DD")

    skip = (page - 1) * page_size
    records = get_violations(
        db,
        skip           = skip,
        limit          = page_size,
        violation_type = violation_type,
        plate_number   = plate_number,
        date_from      = dt_from,
        date_to        = dt_to,
    )

    # Total count (same filters, no pagination)
    total_records = get_violations(
        db,
        skip=0, limit=100_000,
        violation_type=violation_type,
        plate_number=plate_number,
        date_from=dt_from,
        date_to=dt_to,
    )
    total = len(total_records)

    return PaginatedViolations(
        total     = total,
        page      = page,
        page_size = page_size,
        items     = [_serialize(r) for r in records],
    )


# ══════════════════════════════════════════════════════════════
# GET /violations/{id}
# ══════════════════════════════════════════════════════════════

@router.get(
    "/{violation_id}",
    response_model=ViolationRecordOut,
    summary="Get a single violation by database ID",
)
def get_violation(
    violation_id: int,
    db: Session = Depends(get_db),
):
    record = get_violation_by_id(db, violation_id)
    if not record:
        raise HTTPException(
            status_code=404,
            detail=f"Violation with id={violation_id} not found.")
    return _serialize(record)


# ══════════════════════════════════════════════════════════════
# GET /violations/by-plate/{plate}
# ══════════════════════════════════════════════════════════════

@router.get(
    "/by-plate/{plate_number}",
    response_model=PaginatedViolations,
    summary="Get all violations associated with a specific plate number",
)
def get_violations_by_plate(
    plate_number: str,
    page:      int = Query(1,  ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    clean_plate = plate_number.upper().replace(" ", "").replace("-", "")
    records = get_violations(
        db,
        skip         = (page - 1) * page_size,
        limit        = page_size,
        plate_number = clean_plate,
    )
    all_records = get_violations(db, skip=0, limit=100_000,
                                  plate_number=clean_plate)
    return PaginatedViolations(
        total     = len(all_records),
        page      = page,
        page_size = page_size,
        items     = [_serialize(r) for r in records],
    )


# ══════════════════════════════════════════════════════════════
# DELETE /violations/{id}
# ══════════════════════════════════════════════════════════════

@router.delete(
    "/{violation_id}",
    summary="Delete a single violation record",
)
def delete_violation(
    violation_id: int,
    db: Session = Depends(get_db),
):
    record = get_violation_by_id(db, violation_id)
    if not record:
        raise HTTPException(404, f"Violation id={violation_id} not found.")
    db.delete(record)
    db.commit()
    return {"deleted": True, "id": violation_id}


# ══════════════════════════════════════════════════════════════
# DELETE /violations/all
# ══════════════════════════════════════════════════════════════

@router.delete(
    "/all",
    summary="Delete ALL violation records (use with caution)",
)
def delete_all(db: Session = Depends(get_db)):
    count = delete_all_violations(db)
    return {"deleted": True, "count": count,
            "message": f"Deleted {count} violation records."}
