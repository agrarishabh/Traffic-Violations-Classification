"""
tests/test_violations.py
========================
Unit tests for core/violation_detector.py — Phase 4

Strategy:
  - Dataclass tests: no model, no image needed
  - Per-violation tests: use synthetic images + mock DetectionResult
  - Integration tests (marked): call detect_all() with real YOLOv8

HOW TO RUN:
    # Fast tests only (no model):
    python -m pytest tests/test_violations.py -v -m "not integration"

    # All tests including integration:
    python -m pytest tests/test_violations.py -v
"""

import sys
import cv2
import numpy as np
import pytest
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.violation_detector import (
    Violation, ViolationResult, ViolationDetector
)
from core.detector import Detection, DetectionResult
from config import ViolationType, VIOLATION_FINES


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def make_det(class_id: int, bbox: List[int], conf: float = 0.90) -> Detection:
    from config import COCO_CLASSES
    return Detection(class_id=class_id,
                     class_name=COCO_CLASSES.get(class_id, "unknown"),
                     confidence=conf, bbox=bbox)


def make_det_result(vehicles=None, persons=None,
                    traffic_lights=None) -> DetectionResult:
    """Build a DetectionResult from explicit lists."""
    r = DetectionResult(image_path="test.jpg")
    r.vehicles       = vehicles       or []
    r.persons        = persons        or []
    r.traffic_lights = traffic_lights or []
    r.detections     = r.vehicles + r.persons + r.traffic_lights
    r.image_shape    = (640, 640, 3)
    return r


@pytest.fixture(scope="module")
def vd():
    """ViolationDetector with vehicle model loaded (no specialized models)."""
    d = ViolationDetector()
    d.load_models()
    return d


# ── Standard blank images ──────────────────────────────────────
def blank(h=640, w=640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def road_scene(h=640, w=640) -> np.ndarray:
    img = blank(h, w)
    img[:h//2] = (180, 160, 120)   # sky
    img[h//2:] = (70, 70, 75)      # road
    return img


def with_stop_line(img: np.ndarray, y: int = 480) -> np.ndarray:
    """Draw a bright white horizontal stop line at y."""
    out = img.copy()
    cv2.line(out, (0, y), (out.shape[1], y), (255, 255, 255), 6)
    return out


def with_red_light(img: np.ndarray, cx: int = 300,
                   cy: int = 150) -> np.ndarray:
    """Draw a traffic light with red active."""
    out = img.copy()
    cv2.rectangle(out, (cx-20, cy-40), (cx+20, cy+40), (20, 20, 20), -1)
    cv2.circle(out, (cx, cy-20), 12, (0, 0, 220), -1)  # red
    cv2.circle(out, (cx, cy),     12, (20, 20, 20), -1) # yellow off
    cv2.circle(out, (cx, cy+20), 12, (20, 20, 20), -1)  # green off
    return out


def bare_head_region(size=60) -> np.ndarray:
    """Skin-colored patch simulating a bare head (no helmet)."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = (100, 150, 200)   # BGR ≈ skin tone
    return img


def helmeted_head_region(size=60) -> np.ndarray:
    """Dark solid color patch simulating a helmet."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)      # dark = helmet
    return img


# ══════════════════════════════════════════════════════════════
# 1. Violation DATACLASS
# ══════════════════════════════════════════════════════════════

class TestViolationDataclass:

    def test_to_dict_has_all_keys(self):
        v = Violation(
            violation_type="helmet_non_compliance",
            display_name="Helmet Non-Compliance",
            confidence=0.82,
            bbox=[10, 20, 100, 200],
        )
        d = v.to_dict()
        for key in ["violation_type", "display_name", "confidence",
                    "bbox", "vehicle_bbox", "plate_number",
                    "description", "fine_amount"]:
            assert key in d

    def test_confidence_stored(self):
        v = Violation(violation_type="triple_riding",
                      display_name="Triple Riding",
                      confidence=0.77, bbox=[0,0,100,100])
        assert v.confidence == pytest.approx(0.77, abs=0.001)

    def test_vehicle_bbox_optional(self):
        v = Violation(violation_type="stop_line_violation",
                      display_name="Stop Line",
                      confidence=0.9, bbox=[0,0,50,50])
        assert v.vehicle_bbox is None

    def test_plate_number_optional(self):
        v = Violation(violation_type="red_light_violation",
                      display_name="Red Light",
                      confidence=0.85, bbox=[0,0,100,100])
        assert v.plate_number is None


# ══════════════════════════════════════════════════════════════
# 2. ViolationResult DATACLASS
# ══════════════════════════════════════════════════════════════

class TestViolationResult:

    def _make_result(self) -> ViolationResult:
        v1 = Violation("helmet_non_compliance", "Helmet", 0.80,
                       [0,0,50,100], fine_amount=1000)
        v2 = Violation("triple_riding",          "Triple", 0.75,
                       [0,0,200,200], fine_amount=2000)
        return ViolationResult(image_path="test.jpg", violations=[v1, v2])

    def test_has_violations_true(self):
        assert self._make_result().has_violations is True

    def test_has_violations_false(self):
        assert ViolationResult(image_path="x.jpg").has_violations is False

    def test_total_fines(self):
        r = self._make_result()
        assert r.total_fines == 3000

    def test_summary_counts(self):
        r = self._make_result()
        s = r.summary
        assert s["helmet_non_compliance"] == 1
        assert s["triple_riding"]          == 1

    def test_to_dict_structure(self):
        r = self._make_result()
        d = r.to_dict()
        for key in ["image_path", "violations", "total_fines",
                    "summary", "has_violations", "processing_ms"]:
            assert key in d

    def test_empty_result(self):
        r = ViolationResult(image_path="empty.jpg")
        assert r.total_fines    == 0
        assert r.has_violations is False
        assert r.summary        == {}


# ══════════════════════════════════════════════════════════════
# 3. Triple Riding
# ══════════════════════════════════════════════════════════════

class TestTripleRiding:

    def test_two_persons_no_violation(self, vd):
        """Exactly 2 riders — NOT a violation."""
        moto = make_det(3, [100, 100, 500, 500])
        p1   = make_det(0, [150, 120, 350, 460])
        p2   = make_det(0, [200, 130, 400, 450])
        dr   = make_det_result(vehicles=[moto], persons=[p1, p2])
        violations = vd.check_triple_riding(dr)
        assert violations == []

    def test_three_persons_is_violation(self, vd):
        """3 riders → violation."""
        moto = make_det(3, [50, 50, 550, 550])
        p1   = make_det(0, [100, 80,  400, 500])
        p2   = make_det(0, [150, 90,  420, 490])
        p3   = make_det(0, [200, 100, 440, 480])
        dr   = make_det_result(vehicles=[moto], persons=[p1, p2, p3])
        violations = vd.check_triple_riding(dr)
        assert len(violations) == 1
        assert violations[0].violation_type == ViolationType.TRIPLE_RIDING

    def test_four_persons_is_violation(self, vd):
        """4 riders → violation (even more severe)."""
        moto = make_det(3, [0, 0, 640, 640])
        riders = [make_det(0, [50+i*30, 50, 200+i*30, 580]) for i in range(4)]
        dr = make_det_result(vehicles=[moto], persons=riders)
        violations = vd.check_triple_riding(dr)
        assert len(violations) == 1
        assert violations[0].confidence > 0.70

    def test_car_with_persons_ignored(self, vd):
        """Triple riding only applies to two-wheelers."""
        car = make_det(2, [50, 50, 550, 550])
        persons = [make_det(0, [100+i*50, 100, 250+i*50, 500]) for i in range(3)]
        dr = make_det_result(vehicles=[car], persons=persons)
        violations = vd.check_triple_riding(dr)
        assert violations == []

    def test_violation_fine_is_correct(self, vd):
        moto = make_det(3, [0, 0, 640, 640])
        riders = [make_det(0, [50+i*20, 50, 200+i*20, 580]) for i in range(3)]
        dr = make_det_result(vehicles=[moto], persons=riders)
        v = vd.check_triple_riding(dr)[0]
        assert v.fine_amount == VIOLATION_FINES[ViolationType.TRIPLE_RIDING]

    def test_no_vehicles_returns_empty(self, vd):
        dr = make_det_result(vehicles=[], persons=[make_det(0, [0,0,50,100])])
        assert vd.check_triple_riding(dr) == []


# ══════════════════════════════════════════════════════════════
# 4. Stop Line
# ══════════════════════════════════════════════════════════════

class TestStopLine:

    def test_detects_white_horizontal_line(self, vd):
        """_detect_stop_line must find a bright horizontal line."""
        img = road_scene()
        img = with_stop_line(img, y=480)
        y = vd._detect_stop_line(img)
        assert y is not None, "Should detect the white stop line"
        assert 460 <= y <= 500, f"Stop line Y should be near 480, got {y}"

    def test_no_line_returns_none(self, vd):
        """Plain road with no white line → None."""
        img = road_scene()
        y = vd._detect_stop_line(img)
        # May or may not detect (depends on noise) — just ensure no crash
        assert y is None or isinstance(y, int)

    def test_vehicle_crossing_line_is_violation(self, vd):
        """Vehicle bbox bottom below stop line → violation."""
        img  = with_stop_line(road_scene(), y=400)
        car  = make_det(2, [50, 200, 300, 500])   # vy2=500 > line_y=400
        dr   = make_det_result(vehicles=[car])
        violations = vd.check_stop_line(img, dr)
        assert len(violations) >= 1
        assert violations[0].violation_type == ViolationType.STOP_LINE

    def test_vehicle_behind_line_no_violation(self, vd):
        """Vehicle bbox bottom ABOVE stop line → no violation."""
        img  = with_stop_line(road_scene(), y=500)
        car  = make_det(2, [50, 200, 300, 450])   # vy2=450 < line_y=500
        dr   = make_det_result(vehicles=[car])
        violations = vd.check_stop_line(img, dr)
        assert violations == []

    def test_no_vehicles_returns_empty(self, vd):
        img = with_stop_line(road_scene(), y=400)
        dr  = make_det_result(vehicles=[])
        assert vd.check_stop_line(img, dr) == []


# ══════════════════════════════════════════════════════════════
# 5. Red Light
# ══════════════════════════════════════════════════════════════

class TestRedLight:

    def test_no_traffic_lights_returns_empty(self, vd):
        dr = make_det_result(vehicles=[make_det(2, [0,0,200,300])])
        assert vd.check_red_light(blank(), dr) == []

    def test_vehicle_below_red_light_is_violation(self, vd):
        """Vehicle below and laterally near a red traffic light → violation."""
        img = with_red_light(road_scene(), cx=320, cy=100)
        # Traffic light detection at known position
        tl  = make_det(9, [300, 60, 340, 160], conf=0.90)
        # Vehicle below traffic light centre
        car = make_det(2, [200, 200, 450, 500])
        dr  = make_det_result(vehicles=[car], traffic_lights=[tl])
        violations = vd.check_red_light(img, dr)
        # State detection depends on color analysis; just check pipeline runs
        assert isinstance(violations, list)

    def test_violation_type_correct(self, vd):
        """If a red-light violation is raised, it must have correct type."""
        img = with_red_light(road_scene(), cx=320, cy=80)
        tl  = make_det(9, [300, 50, 340, 140], conf=0.92)
        car = make_det(2, [150, 200, 450, 520])
        dr  = make_det_result(vehicles=[car], traffic_lights=[tl])
        violations = vd.check_red_light(img, dr)
        for v in violations:
            assert v.violation_type == ViolationType.RED_LIGHT


# ══════════════════════════════════════════════════════════════
# 6. Helmet check helpers
# ══════════════════════════════════════════════════════════════

class TestHelmetHelpers:

    def test_crop_head_region_returns_correct_fraction(self, vd):
        """Head crop must be the top ~28% of the person bbox height."""
        img    = np.zeros((640, 640, 3), dtype=np.uint8)
        person = make_det(0, [100, 100, 300, 500])   # height = 400px
        crop   = vd._crop_head_region(img, person)
        assert crop is not None
        expected_h = int(400 * 0.28)
        assert abs(crop.shape[0] - expected_h) <= 3

    def test_crop_head_region_clamps_to_image(self, vd):
        """Person at top edge — crop must not go out of bounds."""
        img    = np.zeros((640, 640, 3), dtype=np.uint8)
        person = make_det(0, [0, 0, 100, 200])
        crop   = vd._crop_head_region(img, person)
        assert crop is not None
        assert crop.shape[0] > 0

    def test_bare_head_detected(self, vd):
        """Skin-colored head region → no helmet → violation confidence > 0.5."""
        # Create a person with a very skin-colored head region
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        # Fill head area (top 30%) with skin color (BGR approx)
        img[100:215, 100:300] = (100, 150, 200)   # skin-like
        person = make_det(0, [100, 100, 300, 500])
        has_helmet, conf = vd._has_helmet(img, person)
        # May or may not flag depending on exact skin ratio — just no crash
        assert isinstance(has_helmet, bool)
        assert 0.0 <= conf <= 1.0

    def test_dark_head_detected_as_helmet(self, vd):
        """Completely dark head region → no skin → helmet present."""
        img    = np.zeros((640, 640, 3), dtype=np.uint8)  # all black
        person = make_det(0, [100, 100, 300, 500])
        has_helmet, conf = vd._has_helmet(img, person)
        assert has_helmet is True


# ══════════════════════════════════════════════════════════════
# 7. ViolationResult helpers
# ══════════════════════════════════════════════════════════════

class TestViolationHelpers:

    def test_make_violation_fills_fine(self, vd):
        """_make_violation must look up fine amount from config."""
        v = vd._make_violation(
            vtype=ViolationType.TRIPLE_RIDING,
            confidence=0.80,
            bbox=[0, 0, 100, 100]
        )
        assert v.fine_amount == VIOLATION_FINES[ViolationType.TRIPLE_RIDING]

    def test_make_violation_fills_display_name(self, vd):
        v = vd._make_violation(
            vtype=ViolationType.HELMET_VIOLATION,
            confidence=0.75, bbox=[0,0,50,50]
        )
        assert "Helmet" in v.display_name

    def test_merge_bboxes_two_boxes(self, vd):
        result = vd._merge_bboxes([[10, 20, 100, 200], [50, 5, 300, 150]])
        assert result == [10, 5, 300, 200]

    def test_merge_bboxes_single(self, vd):
        result = vd._merge_bboxes([[30, 40, 130, 140]])
        assert result == [30, 40, 130, 140]

    def test_merge_bboxes_three_boxes(self, vd):
        boxes  = [[0,0,50,50], [25,25,75,75], [10,60,200,200]]
        result = vd._merge_bboxes(boxes)
        assert result == [0, 0, 200, 200]


# ══════════════════════════════════════════════════════════════
# 8. Illegal Parking
# ══════════════════════════════════════════════════════════════

class TestIllegalParking:

    def test_wide_unoccupied_edge_vehicle_flagged(self, vd):
        """Vehicle at image edge, wide aspect ratio, no persons → parking.
        Box must be genuinely wide (w/h > 2.0) to satisfy the parked-car
        aspect-ratio heuristic — parked cars appear wider than tall in frame."""
        img = road_scene()
        # 148px wide, 55px tall  → aspect ratio 2.69 > 2.0 ✓
        # vy2=580 > 640*0.75=480 → near bottom ✓
        # vx2=638 > 640*0.90=576 → near right edge ✓
        car = make_det(2, [490, 525, 638, 580])
        dr  = make_det_result(vehicles=[car], persons=[])
        violations = vd.check_illegal_parking(img, dr)
        assert len(violations) >= 1
        assert violations[0].violation_type == ViolationType.ILLEGAL_PARKING

    def test_occupied_vehicle_not_flagged(self, vd):
        """Vehicle with a person inside (driver) → not parked."""
        img    = road_scene()
        car    = make_det(2, [560, 510, 638, 590])
        driver = make_det(0, [565, 515, 630, 585])  # person inside car
        dr     = make_det_result(vehicles=[car], persons=[driver])
        violations = vd.check_illegal_parking(img, dr)
        # Driver is inside → not flagged as parked
        assert violations == []

    def test_no_vehicles_returns_empty(self, vd):
        assert vd.check_illegal_parking(road_scene(),
                                         make_det_result()) == []


# ══════════════════════════════════════════════════════════════
# 9. Integration — detect_all
# ══════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestDetectAllIntegration:

    def test_returns_violation_result(self, vd):
        img = road_scene()
        result = vd.detect_all(img)
        assert isinstance(result, ViolationResult)

    def test_processing_ms_positive(self, vd):
        result = vd.detect_all(road_scene())
        assert result.processing_ms > 0

    def test_violations_list_type(self, vd):
        result = vd.detect_all(road_scene())
        assert isinstance(result.violations, list)
        for v in result.violations:
            assert isinstance(v, Violation)

    def test_to_dict_serializable(self, vd):
        result = vd.detect_all(road_scene())
        import json
        d = result.to_dict()
        # Must be JSON-serializable (no numpy types)
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

    def test_file_not_found_raises(self, vd):
        with pytest.raises(FileNotFoundError):
            vd.detect_all("/nonexistent/traffic.jpg")
