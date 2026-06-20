"""
app/models/schemas.py
=====================
Pydantic schemas for all FastAPI request/response models.

These define the exact JSON shape sent to and returned from every endpoint.
FastAPI uses these for automatic validation + Swagger UI documentation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# ══════════════════════════════════════════════════════════════
# SHARED PRIMITIVES
# ══════════════════════════════════════════════════════════════

class BBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

    @classmethod
    def from_list(cls, bbox: list) -> "BBox":
        return cls(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])


# ══════════════════════════════════════════════════════════════
# DETECTION SCHEMAS
# ══════════════════════════════════════════════════════════════

class DetectionOut(BaseModel):
    class_id:   int
    class_name: str
    confidence: float
    bbox:       List[int]
    center:     List[int]
    area:       int


class DetectionSummaryOut(BaseModel):
    vehicle_count:  int
    person_count:   int
    light_count:    int
    inference_ms:   float
    vehicles:       List[DetectionOut] = []
    persons:        List[DetectionOut] = []


# ══════════════════════════════════════════════════════════════
# VIOLATION SCHEMAS
# ══════════════════════════════════════════════════════════════

class ViolationOut(BaseModel):
    violation_type: str
    display_name:   str
    confidence:     float
    bbox:           List[int]
    vehicle_bbox:   Optional[List[int]] = None
    plate_number:   Optional[str]       = None
    description:    str                 = ""
    fine_amount:    int


class ViolationRecordOut(BaseModel):
    """Violation record retrieved from the database."""
    id:             int
    evidence_id:    str
    violation_type: str
    display_name:   str
    confidence:     float
    bbox:           Optional[List[int]] = None
    plate_number:   Optional[str]       = None
    fine_amount:    int
    description:    str                 = ""
    detected_at:    str
    image_filename: Optional[str]       = None
    annotated_path: Optional[str]       = None


# ══════════════════════════════════════════════════════════════
# PLATE SCHEMAS
# ══════════════════════════════════════════════════════════════

class PlateOut(BaseModel):
    plate_text:       str
    plate_text_clean: str
    confidence:       float
    bbox:             List[int]
    is_valid_format:  bool
    detection_method: str


# ══════════════════════════════════════════════════════════════
# ANALYZE ENDPOINT
# ══════════════════════════════════════════════════════════════

class AnalyzeResponse(BaseModel):
    """
    Response from POST /analyze
    Full analysis result for one uploaded image.
    """
    evidence_id:     str
    timestamp:       str
    has_violations:  bool
    violation_count: int
    total_fine_inr:  int
    processing_ms:   float

    violations:  List[ViolationOut]     = []
    plates:      List[PlateOut]         = []
    detection:   Optional[DetectionSummaryOut] = None

    annotated_image_url: Optional[str]  = None
    metadata_path:       Optional[str]  = None

    summary: Dict[str, int]             = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "evidence_id":    "abc123def456",
                "timestamp":      "2026-06-18T14:32:05",
                "has_violations": True,
                "violation_count": 2,
                "total_fine_inr": 3000,
                "processing_ms":  1240.5,
                "violations": [
                    {
                        "violation_type": "triple_riding",
                        "display_name":   "Triple Riding",
                        "confidence":     0.85,
                        "bbox":           [150, 200, 500, 430],
                        "plate_number":   "MH12AB1234",
                        "fine_amount":    2000,
                    }
                ],
                "plates": [
                    {
                        "plate_text_clean": "MH12AB1234",
                        "confidence":       0.82,
                        "is_valid_format":  True,
                    }
                ],
            }
        }


# ══════════════════════════════════════════════════════════════
# VIOLATIONS LIST ENDPOINT
# ══════════════════════════════════════════════════════════════

class PaginatedViolations(BaseModel):
    """Response from GET /violations"""
    total:      int
    page:       int
    page_size:  int
    items:      List[ViolationRecordOut]


# ══════════════════════════════════════════════════════════════
# STATS ENDPOINT
# ══════════════════════════════════════════════════════════════

class ViolationTypeStats(BaseModel):
    violation_type: str
    display_name:   str
    count:          int
    total_fines:    int


class StatsResponse(BaseModel):
    """Response from GET /stats"""
    total_images:         int
    total_violations:     int
    total_fines_inr:      int
    by_type:              List[ViolationTypeStats] = []
    most_common:          Optional[str]            = None
    avg_violations_per_image: float                = 0.0

    class Config:
        json_schema_extra = {
            "example": {
                "total_images":     42,
                "total_violations": 87,
                "total_fines_inr":  124500,
                "most_common":      "helmet_non_compliance",
                "by_type": [
                    {"violation_type": "helmet_non_compliance",
                     "display_name":   "Helmet Non-Compliance",
                     "count": 35, "total_fines": 35000},
                ]
            }
        }


# ══════════════════════════════════════════════════════════════
# EVIDENCE ENDPOINT
# ══════════════════════════════════════════════════════════════

class EvidenceOut(BaseModel):
    """Response from GET /evidence/{evidence_id}"""
    evidence_id:    str
    timestamp:      str
    image_filename: str
    annotated_url:  Optional[str] = None
    metadata:       Dict[str, Any] = Field(default_factory=dict)


# ══════════════════════════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    status:        str
    models_loaded: bool
    version:       str = "1.0.0"
    timestamp:     str
