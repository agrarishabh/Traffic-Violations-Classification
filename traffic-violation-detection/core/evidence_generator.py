"""
core/evidence_generator.py
==========================
Evidence Generation — Phase 6 Implementation

Produces per-image evidence packages containing:
  1. annotated.jpg   — original image with colored boxes, labels, timestamp
  2. metadata.json   — structured violation record (JSON)
  3. SQLite record   — violation records inserted into violations.db

Color scheme:
  RED    = confirmed violation  | ORANGE = low-confidence
  CYAN   = vehicle detection    | PURPLE = person detection
  YELLOW = license plate        | WHITE  = traffic light
"""

import cv2
import sys
import json
import uuid
import numpy as np
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    EVIDENCE_DIR, COLORS, BOX_THICKNESS,
    FONT_SCALE, FONT_THICKNESS, VIOLATION_DISPLAY_NAMES,
)


# ══════════════════════════════════════════════════════════════
# EVIDENCE PACKAGE
# ══════════════════════════════════════════════════════════════

@dataclass
class EvidencePackage:
    """
    Complete evidence record for one analysed image.

    Attributes:
        evidence_id       : Unique UUID string
        timestamp         : ISO-8601 datetime string
        source_image_path : Original input image path
        annotated_image   : BGR numpy array with drawn annotations
        annotated_path    : Saved path of annotated.jpg
        metadata          : Full metadata dictionary
        metadata_path     : Saved path of metadata.json
        db_record_ids     : List of database row IDs inserted
    """
    evidence_id:        str
    timestamp:          str
    source_image_path:  str
    annotated_image:    Optional[np.ndarray]  = None
    annotated_path:     Optional[str]         = None
    metadata:           Dict[str, Any]        = field(default_factory=dict)
    metadata_path:      Optional[str]         = None
    db_record_ids:      List[int]             = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "evidence_id":       self.evidence_id,
            "timestamp":         self.timestamp,
            "source_image_path": self.source_image_path,
            "annotated_path":    self.annotated_path,
            "metadata_path":     self.metadata_path,
            "db_record_ids":     self.db_record_ids,
            "metadata":          self.metadata,
        }


# ══════════════════════════════════════════════════════════════
# EVIDENCE GENERATOR
# ══════════════════════════════════════════════════════════════

class EvidenceGenerator:
    """
    Generates annotated images and metadata for traffic violations.

    Usage:
        gen = EvidenceGenerator()
        package = gen.generate(
            image           = loaded_cv2_image,
            image_path      = "traffic.jpg",
            detection_result= det_result,
            violation_result= viol_result,
            plate_results   = plates,
        )
        print(package.annotated_path)  # evidence/2026-06-18/abc123/annotated.jpg
        print(package.metadata_path)   # evidence/2026-06-18/abc123/metadata.json
    """

    # Font used for all text on annotated images
    _FONT      = cv2.FONT_HERSHEY_SIMPLEX
    _FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

    def __init__(self, evidence_dir: Optional[Path] = None,
                 save_to_db: bool = True):
        """
        Args:
            evidence_dir : Override default evidence folder
            save_to_db   : Whether to persist records in SQLite
        """
        self.evidence_dir = Path(evidence_dir) if evidence_dir else EVIDENCE_DIR
        self.save_to_db   = save_to_db
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

    # ── Main pipeline ─────────────────────────────────────────

    def generate(self,
                 image,
                 image_path:       str       = "unknown",
                 detection_result  = None,
                 violation_result  = None,
                 plate_results:    list      = None) -> EvidencePackage:
        """
        Full evidence generation pipeline.

        Args:
            image            : BGR numpy array (original image)
            image_path       : Source file path (for metadata)
            detection_result : DetectionResult from VehicleDetector
            violation_result : ViolationResult from ViolationDetector
            plate_results    : List[PlateResult] from LicensePlateOCR

        Returns:
            EvidencePackage with annotated image + saved paths
        """
        evidence_id = str(uuid.uuid4())[:12]
        timestamp   = datetime.now().isoformat(timespec="seconds")
        plate_results = plate_results or []

        # 1. Annotate image
        annotated = self.draw_annotations(
            image, detection_result, violation_result, plate_results)
        annotated = self.add_timestamp(annotated, timestamp)
        annotated = self.add_summary_panel(
            annotated,
            violation_result.violations if violation_result else [],
            plate_results)

        # 2. Build metadata
        metadata = self.build_metadata(
            evidence_id, timestamp, image_path,
            detection_result, violation_result, plate_results)

        # 3. Create package
        package = EvidencePackage(
            evidence_id       = evidence_id,
            timestamp         = timestamp,
            source_image_path = image_path,
            annotated_image   = annotated,
            metadata          = metadata,
        )

        # 4. Save to disk
        package = self.save_evidence(package)

        # 5. Save to database
        if self.save_to_db:
            package.db_record_ids = self._save_to_database(package)

        return package

    # ── Drawing ───────────────────────────────────────────────

    def draw_annotations(self, image: np.ndarray,
                          detection_result=None,
                          violation_result=None,
                          plate_results: list = None) -> np.ndarray:
        """
        Draw all bounding boxes on the image.

        Layer order (bottom to top):
          1. Vehicle boxes (CYAN)
          2. Person boxes  (PURPLE)
          3. Traffic light boxes (WHITE)
          4. Plate boxes (YELLOW)
          5. Violation boxes (RED/ORANGE) — drawn last so always visible
        """
        out = image.copy()
        plate_results = plate_results or []

        # ── Vehicle detections ────────────────────────────────
        if detection_result:
            for v in detection_result.vehicles:
                out = self.draw_box(
                    out, v.bbox,
                    label=f"{v.class_name} {v.confidence:.0%}",
                    color=COLORS["vehicle"],
                    thickness=BOX_THICKNESS)

            for p in detection_result.persons:
                out = self.draw_box(
                    out, p.bbox,
                    label=f"person {p.confidence:.0%}",
                    color=COLORS["person"],
                    thickness=BOX_THICKNESS)

            for tl in detection_result.traffic_lights:
                out = self.draw_box(
                    out, tl.bbox,
                    label=f"traffic light {tl.confidence:.0%}",
                    color=(255, 255, 255),
                    thickness=BOX_THICKNESS)

        # ── License plates ────────────────────────────────────
        for plate in plate_results:
            label = plate.plate_text_clean or "plate"
            if plate.is_valid_format:
                label += " [valid]"
            out = self.draw_box(
                out, plate.bbox,
                label=label,
                color=COLORS["plate"],
                thickness=BOX_THICKNESS)

        # ── Violations (drawn on top) ─────────────────────────
        if violation_result:
            for v in violation_result.violations:
                color = COLORS["violation"] if v.confidence >= 0.65 \
                        else COLORS["warning"]
                out = self.draw_box(
                    out, v.bbox,
                    label=f"{v.display_name} {v.confidence:.0%}",
                    color=color,
                    thickness=BOX_THICKNESS + 1)   # thicker for violations

        return out

    def draw_box(self, image: np.ndarray, bbox: List[int],
                 label: str, color: tuple,
                 thickness: int = BOX_THICKNESS) -> np.ndarray:
        """
        Draw one labeled bounding box.

        Layout:
          Colored rectangle → colored label background → black text
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h_img, w_img = image.shape[:2]

        # Clamp to image bounds
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w_img, x2), min(h_img, y2)

        if x2c <= x1c or y2c <= y1c:
            return image

        # Bounding rectangle
        cv2.rectangle(image, (x1c, y1c), (x2c, y2c), color, thickness)

        if not label:
            return image

        # Label background
        (tw, th), baseline = cv2.getTextSize(
            label, self._FONT, FONT_SCALE, FONT_THICKNESS)
        label_y = max(y1c - 4, th + baseline + 4)
        bg_x2   = min(w_img, x1c + tw + 6)
        bg_y1   = max(0, label_y - th - baseline - 4)

        cv2.rectangle(image, (x1c, bg_y1), (bg_x2, label_y + 2), color, -1)

        # Label text in black
        cv2.putText(image, label, (x1c + 3, label_y - baseline),
                    self._FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS)
        return image

    def add_timestamp(self, image: np.ndarray, timestamp: str) -> np.ndarray:
        """
        Add a semi-transparent timestamp banner at the top-left.
        Format: 2026-06-18T14:32:05
        """
        h, w = image.shape[:2]
        banner_h = 32
        overlay  = image.copy()

        # Dark semi-transparent strip
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)

        # Timestamp text
        cv2.putText(image,
                    f"  {timestamp}   |   Traffic Violation Detection System",
                    (6, 22), self._FONT, 0.50, (220, 220, 220), 1)
        return image

    def add_summary_panel(self, image: np.ndarray,
                           violations: list,
                           plate_results: list = None) -> np.ndarray:
        """
        Add a dark information panel at the bottom of the image.

        Shows:
          Line 1: Violation count + plate numbers
          Line 2: Violation types + fines
        """
        plate_results = plate_results or []
        h, w = image.shape[:2]
        panel_h = 50
        panel = np.zeros((panel_h, w, 3), dtype=np.uint8)
        panel[:] = (15, 18, 30)   # very dark navy

        # Line 1: violation count + plates
        plates_str = ""
        if plate_results:
            plates_str = "  |  Plates: " + ", ".join(
                p.plate_text_clean for p in plate_results if p.plate_text_clean)

        viol_count = len(violations)
        line1 = (f"  Violations: {viol_count}" +
                 (f"  |  Total fine: Rs.{sum(v.fine_amount for v in violations)}"
                  if violations else "") +
                 plates_str)
        cv2.putText(panel, line1, (4, 18),
                    self._FONT, 0.50, (200, 220, 255), 1)

        # Line 2: violation names
        if violations:
            names = " | ".join(
                f"{v.display_name}({v.confidence:.0%})"
                for v in violations[:4])   # max 4 to fit
            if len(violations) > 4:
                names += f" +{len(violations)-4} more"
        else:
            names = "  No violations detected"
        cv2.putText(panel, f"  {names}", (4, 38),
                    self._FONT, 0.46, (180, 200, 160), 1)

        return np.vstack([image, panel])

    # ── Metadata ──────────────────────────────────────────────

    def build_metadata(self, evidence_id: str, timestamp: str,
                        image_path: str,
                        detection_result=None,
                        violation_result=None,
                        plate_results: list = None) -> Dict[str, Any]:
        """Build the full metadata dictionary for JSON output."""
        plate_results = plate_results or []

        meta: Dict[str, Any] = {
            "evidence_id":   evidence_id,
            "timestamp":     timestamp,
            "image_path":    image_path,
            "system":        "Traffic Violation Detection System v1.0",
        }

        # Detection summary
        if detection_result:
            meta["detection"] = detection_result.to_dict()
        else:
            meta["detection"] = {}

        # Violations
        if violation_result:
            meta["violations"]   = violation_result.to_dict()
        else:
            meta["violations"]   = {"violations": [], "total_fines": 0}

        # License plates
        meta["plates"] = [p.to_dict() for p in plate_results]

        # Convenience top-level fields for quick queries
        meta["has_violations"]  = bool(
            violation_result and violation_result.has_violations)
        meta["violation_count"] = len(
            violation_result.violations) if violation_result else 0
        meta["total_fine_inr"]  = (
            violation_result.total_fines if violation_result else 0)
        meta["plate_numbers"]   = [
            p.plate_text_clean for p in plate_results if p.plate_text_clean]

        return meta

    # ── Saving ────────────────────────────────────────────────

    def save_evidence(self, package: EvidencePackage) -> EvidencePackage:
        """
        Save annotated image and metadata JSON to disk.

        Output structure:
          evidence/
            YYYY-MM-DD/
              <evidence_id>/
                annotated.jpg
                metadata.json
        """
        ev_dir = self._make_evidence_dir(package.evidence_id,
                                         package.timestamp)

        # Save annotated image
        if package.annotated_image is not None:
            img_path = ev_dir / "annotated.jpg"
            cv2.imwrite(str(img_path), package.annotated_image)
            package.annotated_path = str(img_path)

        # Save metadata JSON
        meta_path = ev_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(package.metadata, f, indent=2, ensure_ascii=False,
                      default=str)   # default=str handles non-serializable types
        package.metadata_path = str(meta_path)

        return package

    def _make_evidence_dir(self, evidence_id: str,
                            timestamp: str) -> Path:
        """Create and return the evidence subdirectory for this record."""
        date_str = timestamp[:10]   # YYYY-MM-DD
        ev_dir   = self.evidence_dir / date_str / evidence_id
        ev_dir.mkdir(parents=True, exist_ok=True)
        return ev_dir

    # ── Database integration ──────────────────────────────────

    def _save_to_database(self, package: EvidencePackage) -> List[int]:
        """
        Persist evidence to SQLite via core/database.py.
        Returns list of inserted violation row IDs.
        """
        try:
            from core.database import (SessionLocal, init_db,
                                        save_processed_image, save_violation)
            init_db()
            db = SessionLocal()
            row_ids = []

            try:
                violations = package.metadata.get(
                    "violations", {}).get("violations", [])
                plates     = package.metadata.get("plates", [])
                plate_no   = plates[0]["plate_text_clean"] \
                             if plates else None

                # Save image record
                save_processed_image(db, {
                    "evidence_id":     package.evidence_id,
                    "filename":        Path(package.source_image_path).name,
                    "source_path":     package.source_image_path,
                    "annotated_path":  package.annotated_path,
                    "metadata_path":   package.metadata_path,
                    "has_violations":  package.metadata.get("has_violations",
                                                            False),
                    "violation_count": package.metadata.get("violation_count",
                                                            0),
                })

                # Save each violation
                for v in violations:
                    record = save_violation(db, {
                        "evidence_id":    package.evidence_id,
                        "violation_type": v["violation_type"],
                        "display_name":   v["display_name"],
                        "confidence":     v["confidence"],
                        "bbox":           v["bbox"],
                        "plate_number":   v.get("plate_number") or plate_no,
                        "fine_amount":    v["fine_amount"],
                        "description":    v.get("description", ""),
                        "image_filename": Path(
                            package.source_image_path).name,
                        "annotated_path": package.annotated_path,
                    })
                    row_ids.append(record.id)

                db.commit()
            finally:
                db.close()

            return row_ids

        except Exception as e:
            print(f"  WARN  DB save failed (non-fatal): {e}")
            return []
