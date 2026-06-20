"""
tests/test_detection.py
=======================
Unit tests for core/detector.py — Phase 3

Test strategy:
  - Data class tests (Detection, DetectionResult) need NO model
  - Geometry tests (IoU, overlap, person-on-vehicle) need NO model
  - Classification tests (vehicle type, two-wheeler) need NO model
  - Integration tests (actual YOLOv8 inference) use the real model

HOW TO RUN:
    python -m pytest tests/test_detection.py -v

    # Skip the slow model-loading integration tests:
    python -m pytest tests/test_detection.py -v -m "not integration"
"""

import sys
import numpy as np
import pytest
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.detector import Detection, DetectionResult, VehicleDetector


# ══════════════════════════════════════════════════════════════
# FIXTURES & HELPERS
# ══════════════════════════════════════════════════════════════

def make_detection(class_id: int, bbox: List[int],
                   confidence: float = 0.90) -> Detection:
    """Helper — create a Detection with minimal boilerplate."""
    from config import COCO_CLASSES
    name = COCO_CLASSES.get(class_id, "unknown")
    return Detection(class_id=class_id, class_name=name,
                     confidence=confidence, bbox=bbox)


@pytest.fixture(scope="module")
def detector():
    """VehicleDetector instance — model NOT loaded (geometry tests only)."""
    return VehicleDetector()


@pytest.fixture(scope="module")
def loaded_detector():
    """VehicleDetector with model loaded — used for integration tests."""
    d = VehicleDetector(model_name="yolov8n.pt")
    d.load_model()
    return d


def make_scene_image(h: int = 640, w: int = 640) -> np.ndarray:
    """Synthetic BGR traffic scene for integration tests."""
    import cv2
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:h//2, :] = (180, 160, 120)      # sky
    img[h//2:, :] = (70, 70, 75)         # road
    # Car-like rectangle
    cv2.rectangle(img, (100, 350), (350, 500), (40, 80, 200), -1)
    cv2.rectangle(img, (150, 290), (300, 360), (40, 80, 200), -1)
    # Person-like rectangle
    cv2.rectangle(img, (450, 320), (510, 500), (210, 180, 145), -1)
    cv2.circle(img,    (480, 300), 30,         (210, 180, 145), -1)
    return img


# ══════════════════════════════════════════════════════════════
# 1. Detection DATACLASS
# ══════════════════════════════════════════════════════════════

class TestDetectionDataclass:

    def test_area_calculation(self):
        """area = (x2-x1) * (y2-y1)."""
        d = make_detection(2, [10, 20, 110, 120])   # 100×100 = 10000
        assert d.area == 10000

    def test_area_zero_for_degenerate_box(self):
        """Zero-size box must have area 0."""
        d = make_detection(2, [50, 50, 50, 50])
        assert d.area == 0

    def test_center_calculation(self):
        """Center must be the midpoint of the bounding box."""
        d = make_detection(2, [0, 0, 100, 100])
        assert d.center == (50, 50)

    def test_center_non_square(self):
        d = make_detection(2, [10, 20, 210, 120])   # 200×100
        cx, cy = d.center
        assert cx == 110     # (10+210)//2
        assert cy == 70      # (20+120)//2

    def test_width_and_height(self):
        d = make_detection(2, [30, 40, 230, 190])   # w=200, h=150
        assert d.width  == 200
        assert d.height == 150

    def test_to_dict_keys(self):
        """to_dict() must contain all required keys."""
        d = make_detection(3, [0, 0, 50, 80])
        result = d.to_dict()
        for key in ["class_id", "class_name", "confidence",
                    "bbox", "center", "area", "width", "height"]:
            assert key in result, f"Missing key: {key}"

    def test_to_dict_values(self):
        """to_dict() values must match the input data."""
        d = make_detection(2, [10, 20, 110, 120], confidence=0.87)
        result = d.to_dict()
        assert result["class_id"]   == 2
        assert result["class_name"] == "car"
        assert result["confidence"] == pytest.approx(0.87, abs=0.001)
        assert result["bbox"]       == [10, 20, 110, 120]

    def test_track_id_default_none(self):
        """track_id must default to None."""
        d = make_detection(2, [0, 0, 100, 100])
        assert d.track_id is None

    def test_track_id_assigned(self):
        d = Detection(class_id=2, class_name="car",
                      confidence=0.9, bbox=[0, 0, 100, 100], track_id=42)
        assert d.track_id == 42

    def test_confidence_stored_correctly(self):
        d = make_detection(0, [0, 0, 50, 100], confidence=0.634)
        assert d.confidence == pytest.approx(0.634, abs=0.001)


# ══════════════════════════════════════════════════════════════
# 2. DetectionResult DATACLASS
# ══════════════════════════════════════════════════════════════

class TestDetectionResult:

    def _make_result(self) -> DetectionResult:
        """Build a DetectionResult with 2 vehicles and 1 person."""
        car   = make_detection(2, [50,  200, 250, 400])   # car
        moto  = make_detection(3, [300, 200, 450, 380])   # motorcycle
        person = make_detection(0, [310, 180, 420, 380])  # person on moto
        tl    = make_detection(9, [500, 50,  540, 150])   # traffic light

        result = DetectionResult(image_path="test.jpg")
        result.detections     = [car, moto, person, tl]
        result.vehicles       = [car, moto]
        result.persons        = [person]
        result.traffic_lights = [tl]
        result.inference_ms   = 45.0
        result.image_shape    = (640, 640, 3)
        return result

    def test_vehicle_count(self):
        result = self._make_result()
        assert result.vehicle_count == 2

    def test_person_count(self):
        result = self._make_result()
        assert result.person_count == 1

    def test_has_red_light_true(self):
        result = self._make_result()
        assert result.has_red_light is True

    def test_has_red_light_false_when_no_light(self):
        result = DetectionResult(image_path="empty.jpg")
        assert result.has_red_light is False

    def test_to_dict_structure(self):
        result = self._make_result()
        d = result.to_dict()
        for key in ["image_path", "image_shape", "total_objects",
                    "vehicle_count", "person_count",
                    "vehicles", "persons", "traffic_lights", "inference_ms"]:
            assert key in d, f"Missing key in to_dict(): {key}"

    def test_to_dict_counts_correct(self):
        result = self._make_result()
        d = result.to_dict()
        assert d["vehicle_count"]  == 2
        assert d["person_count"]   == 1
        assert d["total_objects"]  == 4

    def test_empty_result(self):
        """Empty DetectionResult must not crash."""
        result = DetectionResult(image_path="empty.jpg")
        assert result.vehicle_count  == 0
        assert result.person_count   == 0
        assert result.has_red_light is False
        d = result.to_dict()
        assert d["vehicles"]  == []
        assert d["persons"]   == []


# ══════════════════════════════════════════════════════════════
# 3. IoU & Overlap Geometry
# ══════════════════════════════════════════════════════════════

class TestIoUAndOverlap:

    def test_identical_boxes_iou_is_one(self, detector):
        """Identical boxes must have IoU = 1.0."""
        box = [10, 10, 110, 110]
        assert detector.compute_iou(box, box) == pytest.approx(1.0)

    def test_non_overlapping_iou_is_zero(self, detector):
        """Non-overlapping boxes must have IoU = 0."""
        box1 = [0,   0,  50,  50]
        box2 = [100, 100, 200, 200]
        assert detector.compute_iou(box1, box2) == pytest.approx(0.0)

    def test_partial_overlap_iou_range(self, detector):
        """Partial overlap must produce IoU between 0 and 1."""
        box1 = [0,   0, 100, 100]
        box2 = [50, 50, 150, 150]
        iou = detector.compute_iou(box1, box2)
        assert 0.0 < iou < 1.0

    def test_half_overlap_iou_value(self, detector):
        """
        Two 100×100 boxes offset by 50px horizontally:
          intersection = 50×100 = 5000
          union        = 100×100 + 100×100 - 5000 = 15000
          IoU          = 5000/15000 = 0.333...
        """
        box1 = [0, 0, 100, 100]
        box2 = [50, 0, 150, 100]
        iou = detector.compute_iou(box1, box2)
        assert iou == pytest.approx(1/3, abs=0.01)

    def test_iou_symmetry(self, detector):
        """IoU(A,B) must equal IoU(B,A)."""
        box1 = [0, 0, 80, 80]
        box2 = [40, 40, 120, 120]
        assert detector.compute_iou(box1, box2) == \
               pytest.approx(detector.compute_iou(box2, box1), abs=1e-6)

    def test_boxes_overlap_true(self, detector):
        box1 = [0,   0, 100, 100]
        box2 = [80, 80, 180, 180]   # corner overlap
        assert detector.boxes_overlap(box1, box2, iou_threshold=0.01) is True

    def test_boxes_overlap_false(self, detector):
        box1 = [0,   0, 50,  50]
        box2 = [200, 200, 300, 300]
        assert detector.boxes_overlap(box1, box2) is False

    def test_boxes_overlap_threshold_respected(self, detector):
        """Low threshold should accept small overlaps."""
        box1 = [0, 0, 100, 100]
        box2 = [95, 0, 200, 100]   # very small overlap (5px wide strip)
        # IoU = 500 / (10000 + 10500 - 500) ≈ 0.025
        assert detector.boxes_overlap(box1, box2, iou_threshold=0.01) is True
        assert detector.boxes_overlap(box1, box2, iou_threshold=0.5) is False


# ══════════════════════════════════════════════════════════════
# 4. Person-on-Vehicle Containment
# ══════════════════════════════════════════════════════════════

class TestPersonOnVehicle:

    def test_person_fully_inside_vehicle(self, detector):
        """Person box completely inside vehicle box → on vehicle."""
        vehicle = make_detection(3, [100, 100, 500, 500])  # big motorcycle
        person  = make_detection(0, [200, 150, 350, 450])  # inside it
        assert detector.is_person_on_vehicle(person, vehicle) is True

    def test_person_outside_vehicle(self, detector):
        """Person box with no overlap with vehicle → NOT on vehicle."""
        vehicle = make_detection(2, [0,   0,  200, 300])
        person  = make_detection(0, [400, 400, 500, 600])
        assert detector.is_person_on_vehicle(person, vehicle) is False

    def test_person_partially_overlapping(self, detector):
        """Person overlapping >30% of their own box → on vehicle."""
        vehicle = make_detection(3, [100, 100, 400, 500])
        person  = make_detection(0, [150, 80,  350, 480])  # mostly inside
        assert detector.is_person_on_vehicle(person, vehicle, 0.3) is True

    def test_get_persons_on_vehicle_empty(self, detector):
        """No persons → empty list returned."""
        vehicle = make_detection(3, [0, 0, 200, 300])
        assert detector.get_persons_on_vehicle(vehicle, []) == []

    def test_get_persons_on_vehicle_finds_correct_ones(self, detector):
        """Only persons overlapping the vehicle should be returned."""
        vehicle  = make_detection(3, [100, 100, 400, 450])
        rider1   = make_detection(0, [150, 120, 360, 440])  # on vehicle
        rider2   = make_detection(0, [160, 130, 370, 430])  # on vehicle
        bystander = make_detection(0, [600, 100, 700, 350]) # far away
        on_vehicle = detector.get_persons_on_vehicle(
            vehicle, [rider1, rider2, bystander])
        assert len(on_vehicle) == 2
        assert bystander not in on_vehicle

    def test_triple_riding_scenario(self, detector):
        """3 persons on one motorcycle → get_persons_on_vehicle returns 3."""
        moto = make_detection(3, [100, 100, 500, 500])
        p1 = make_detection(0, [130, 120, 300, 480])
        p2 = make_detection(0, [200, 110, 380, 460])
        p3 = make_detection(0, [250, 130, 420, 470])
        result = detector.get_persons_on_vehicle(moto, [p1, p2, p3])
        assert len(result) == 3


# ══════════════════════════════════════════════════════════════
# 5. Vehicle Classification
# ══════════════════════════════════════════════════════════════

class TestVehicleClassification:

    def test_bicycle_is_two_wheeler(self, detector):
        d = make_detection(1, [0, 0, 100, 100])   # bicycle
        assert detector.is_two_wheeler(d) is True
        assert detector.classify_vehicle_type(d) == "two_wheeler"

    def test_motorcycle_is_two_wheeler(self, detector):
        d = make_detection(3, [0, 0, 100, 100])   # motorcycle
        assert detector.is_two_wheeler(d) is True
        assert detector.classify_vehicle_type(d) == "two_wheeler"

    def test_car_is_four_wheeler(self, detector):
        d = make_detection(2, [0, 0, 200, 150])   # car
        assert detector.is_four_wheeler(d) is True
        assert detector.is_two_wheeler(d) is False
        assert detector.classify_vehicle_type(d) == "four_wheeler"

    def test_bus_is_heavy_vehicle(self, detector):
        d = make_detection(5, [0, 0, 400, 300])   # bus
        assert detector.classify_vehicle_type(d) == "heavy_vehicle"
        assert detector.is_four_wheeler(d) is True   # buses are in FOUR_WHEELER_IDS

    def test_truck_is_heavy_vehicle(self, detector):
        d = make_detection(7, [0, 0, 400, 300])   # truck
        assert detector.classify_vehicle_type(d) == "heavy_vehicle"

    def test_person_is_unknown_vehicle_type(self, detector):
        d = make_detection(0, [0, 0, 50, 150])    # person (not a vehicle)
        assert detector.classify_vehicle_type(d) == "unknown"
        assert detector.is_two_wheeler(d) is False
        assert detector.is_four_wheeler(d) is False

    @pytest.mark.parametrize("class_id,expected_type", [
        (1, "two_wheeler"),    # bicycle
        (2, "four_wheeler"),   # car
        (3, "two_wheeler"),    # motorcycle
        (5, "heavy_vehicle"),  # bus
        (7, "heavy_vehicle"),  # truck
    ])
    def test_classify_all_vehicle_types(self, detector, class_id, expected_type):
        d = make_detection(class_id, [0, 0, 100, 100])
        assert detector.classify_vehicle_type(d) == expected_type


# ══════════════════════════════════════════════════════════════
# 6. Model Loading
# ══════════════════════════════════════════════════════════════

class TestModelLoading:

    def test_model_not_loaded_before_load_model(self):
        d = VehicleDetector()
        assert d._model_loaded is False
        assert d.model is None

    def test_load_model_sets_flag(self, loaded_detector):
        assert loaded_detector._model_loaded is True
        assert loaded_detector.model is not None

    def test_model_has_coco_classes(self, loaded_detector):
        """YOLOv8 COCO model must know about 'car', 'person', 'motorcycle'."""
        names = loaded_detector.model.names
        # names is a dict: {0: 'person', 1: 'bicycle', 2: 'car', ...}
        assert names[0]  == "person"
        assert names[2]  == "car"
        assert names[3]  == "motorcycle"

    @pytest.mark.integration
    def test_detect_returns_detection_result(self, loaded_detector):
        """detect() on a real image must return a DetectionResult."""
        img = make_scene_image()
        result = loaded_detector.detect(img)
        assert isinstance(result, DetectionResult)
        assert isinstance(result.detections, list)
        assert isinstance(result.vehicles,   list)
        assert isinstance(result.persons,    list)
        assert result.inference_ms > 0

    @pytest.mark.integration
    def test_detect_inference_timing(self, loaded_detector):
        """inference_ms must be recorded and positive."""
        img = make_scene_image()
        result = loaded_detector.detect(img)
        assert result.inference_ms > 0.0

    @pytest.mark.integration
    def test_detect_image_shape_recorded(self, loaded_detector):
        """image_shape must match the input image dimensions."""
        img = make_scene_image(h=480, w=640)
        result = loaded_detector.detect(img)
        assert result.image_shape == (480, 640, 3)

    @pytest.mark.integration
    def test_detect_confidence_threshold_filters(self, loaded_detector):
        """High confidence threshold must return fewer detections."""
        img = make_scene_image()
        result_low  = loaded_detector.detect(img, conf_threshold=0.10)
        result_high = loaded_detector.detect(img, conf_threshold=0.90)
        # High threshold must give <= same number as low threshold
        assert len(result_high.detections) <= len(result_low.detections)

    @pytest.mark.integration
    def test_detect_file_not_found_raises(self, loaded_detector):
        """detect() on nonexistent path must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            loaded_detector.detect("/nonexistent/path/image.jpg")

    @pytest.mark.integration
    def test_detect_all_detections_above_threshold(self, loaded_detector):
        """Every detection in result must meet the confidence threshold."""
        img = make_scene_image()
        threshold = 0.50
        result = loaded_detector.detect(img, conf_threshold=threshold)
        for det in result.detections:
            assert det.confidence >= threshold, \
                f"Detection {det.class_name} has conf {det.confidence} < {threshold}"

    @pytest.mark.integration
    def test_vehicles_subset_of_detections(self, loaded_detector):
        """result.vehicles must be a strict subset of result.detections."""
        img = make_scene_image()
        result = loaded_detector.detect(img)
        from config import VEHICLE_CLASS_IDS
        for v in result.vehicles:
            assert v in result.detections
            assert v.class_id in VEHICLE_CLASS_IDS

    @pytest.mark.integration
    def test_persons_subset_of_detections(self, loaded_detector):
        img = make_scene_image()
        result = loaded_detector.detect(img)
        for p in result.persons:
            assert p in result.detections
            assert p.class_id == 0  # COCO person class
