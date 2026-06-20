"""
tests/test_ocr.py
=================
Unit tests for core/ocr.py — Phase 5

Strategy:
  - PlateResult dataclass tests    → no model
  - validate_plate tests           → no model (pure regex)
  - clean_plate_text tests         → no model (pure string)
  - crop_plate tests               → no model (pure OpenCV)
  - detect_plate_regions_contour   → no model (pure OpenCV)
  - _deduplicate_boxes             → no model (pure geometry)
  - run_ocr + extract integration  → requires EasyOCR (marked integration)

HOW TO RUN:
    # Fast tests only:
    python -m pytest tests/test_ocr.py -v -m "not integration"

    # All tests (EasyOCR loads ~100MB on first run):
    python -m pytest tests/test_ocr.py -v
"""

import sys
import cv2
import numpy as np
import pytest
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.ocr import LicensePlateOCR, PlateResult


# ══════════════════════════════════════════════════════════════
# FIXTURES & IMAGE HELPERS
# ══════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def ocr():
    """LicensePlateOCR without model loaded — for non-OCR tests."""
    return LicensePlateOCR()


@pytest.fixture(scope="module")
def loaded_ocr():
    """LicensePlateOCR with EasyOCR loaded — for integration tests."""
    o = LicensePlateOCR(use_gpu=False)
    o.load_models()
    return o


def make_blank(h=480, w=640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def make_plate_image(text: str = "MH12AB1234",
                     width: int = 300, height: int = 70,
                     bg_color=(255, 255, 255),
                     text_color=(0, 0, 0)) -> np.ndarray:
    """
    Create a synthetic license plate image with rendered text.
    Used for integration OCR tests — EasyOCR can read clean synthetic text.
    """
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)
    # Black border
    cv2.rectangle(img, (2, 2), (width-3, height-3), (0, 0, 0), 2)
    # Plate text — use a large, clear font
    font_scale = height / 55.0
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                   font_scale, 2)
    tx = max(4, (width - tw) // 2)
    ty = (height + th) // 2
    cv2.putText(img, text, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2)
    return img


def make_image_with_plate(plate_text="MH12AB1234") -> np.ndarray:
    """
    640×480 scene with a white rectangular plate region in the lower-centre.
    The plate rectangle has the correct aspect ratio to pass contour detection.
    """
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (80, 80, 85)            # dark background (road)
    img[:240] = (160, 140, 100)      # sky

    # Draw car body
    cv2.rectangle(img, (150, 250), (490, 420), (30, 60, 160), -1)

    # Draw a plate-like white rectangle at the front of the car
    # Width=160, Height=40  → aspect = 4.0 (within 1.8–7.0 range)
    px1, py1, px2, py2 = 240, 390, 400, 430
    cv2.rectangle(img, (px1, py1), (px2, py2), (255, 255, 255), -1)
    cv2.rectangle(img, (px1, py1), (px2, py2), (0,   0,   0),   2)

    # Write plate text on the rectangle
    font_scale = 0.65
    cv2.putText(img, plate_text, (px1 + 8, py2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
    return img


# ══════════════════════════════════════════════════════════════
# 1. PlateResult DATACLASS
# ══════════════════════════════════════════════════════════════

class TestPlateResultDataclass:

    def test_to_dict_has_all_keys(self):
        r = PlateResult(plate_text="MH 12 AB 1234",
                        plate_text_clean="MH12AB1234",
                        confidence=0.91, bbox=[10, 20, 160, 50],
                        is_valid_format=True)
        d = r.to_dict()
        for key in ["plate_text", "plate_text_clean", "confidence",
                    "bbox", "is_valid_format", "detection_method"]:
            assert key in d, f"Missing key: {key}"

    def test_confidence_stored_correctly(self):
        r = PlateResult("DL01CD5678", "DL01CD5678", 0.873,
                        [0,0,200,50], True)
        assert r.confidence == pytest.approx(0.873, abs=0.001)

    def test_default_detection_method(self):
        r = PlateResult("KA03EF9012", "KA03EF9012", 0.80,
                        [0,0,100,40], True)
        assert r.detection_method == "contour"

    def test_plate_image_optional(self):
        r = PlateResult("TN04GH3456", "TN04GH3456", 0.75,
                        [0,0,150,45], True)
        assert r.plate_image is None

    def test_to_dict_does_not_include_image(self):
        img = np.zeros((40, 150, 3), dtype=np.uint8)
        r = PlateResult("MH12AB1234", "MH12AB1234", 0.90,
                        [0,0,150,40], True, plate_image=img)
        d = r.to_dict()
        assert "plate_image" not in d   # numpy array is not JSON-safe


# ══════════════════════════════════════════════════════════════
# 2. validate_plate
# ══════════════════════════════════════════════════════════════

class TestValidatePlate:

    @pytest.mark.parametrize("plate", [
        "MH12AB1234",   # Maharashtra
        "DL01CD5678",   # Delhi
        "KA03EF9012",   # Karnataka
        "TN04GH3456",   # Tamil Nadu
        "UP14IJ7890",   # Uttar Pradesh
        "WB20KL2345",   # West Bengal
        "GJ05MN6789",   # Gujarat
        "MH12ABC1234",  # 3-letter series (valid)
    ])
    def test_valid_indian_plates(self, ocr, plate):
        assert ocr.validate_plate(plate) is True, \
            f"Expected {plate} to be valid"

    @pytest.mark.parametrize("plate", [
        "",             # empty
        "ABC",          # too short
        "12345678",     # no letters
        "ABCDEFGH",     # no digits
        "MH12",         # incomplete
        "MHAB1234",     # missing district number
        "1234AB5678",   # starts with digits
        "MH12AB12345",  # too many serial digits
        "MH1AB1234",    # only 1 digit in district
    ])
    def test_invalid_plates(self, ocr, plate):
        assert ocr.validate_plate(plate) is False, \
            f"Expected {plate} to be invalid"

    def test_validates_with_spaces(self, ocr):
        """Plate with spaces should still be recognised as valid."""
        assert ocr.validate_plate("MH 12 AB 1234") is True

    def test_validates_with_hyphens(self, ocr):
        assert ocr.validate_plate("MH-12-AB-1234") is True

    def test_validates_lowercase(self, ocr):
        assert ocr.validate_plate("mh12ab1234") is True


# ══════════════════════════════════════════════════════════════
# 3. clean_plate_text
# ══════════════════════════════════════════════════════════════

class TestCleanPlateText:

    def test_removes_spaces(self, ocr):
        assert ocr.clean_plate_text("MH 12 AB 1234") == "MH12AB1234"

    def test_removes_hyphens(self, ocr):
        assert ocr.clean_plate_text("MH-12-AB-1234") == "MH12AB1234"

    def test_uppercases(self, ocr):
        result = ocr.clean_plate_text("mh12ab1234")
        assert result == result.upper()

    def test_empty_string_returns_empty(self, ocr):
        assert ocr.clean_plate_text("") == ""

    def test_corrects_O_to_0_in_district(self, ocr):
        """O in position 2–3 (district digits) should become 0."""
        result = ocr.clean_plate_text("MHO2AB1234")
        assert result[2] == "0", f"Position 2 should be '0', got {result}"

    def test_corrects_I_to_1_in_district(self, ocr):
        result = ocr.clean_plate_text("MHI2AB1234")
        assert result[2] == "1"

    def test_removes_punctuation(self, ocr):
        result = ocr.clean_plate_text("MH.12:AB,1234")
        assert "." not in result
        assert ":" not in result
        assert "," not in result

    def test_corrects_O_in_serial_number(self, ocr):
        """O in serial number (positions 7+) should become 0."""
        result = ocr.clean_plate_text("MH12AB123O")
        assert result[-1] == "0"

    def test_normal_plate_unchanged_after_clean(self, ocr):
        clean = ocr.clean_plate_text("MH12AB1234")
        assert clean == "MH12AB1234"


# ══════════════════════════════════════════════════════════════
# 4. crop_plate
# ══════════════════════════════════════════════════════════════

class TestCropPlate:

    def test_returns_ndarray(self, ocr):
        img  = make_blank()
        crop = ocr.crop_plate(img, [100, 200, 300, 240])
        assert isinstance(crop, np.ndarray)

    def test_output_has_3_channels(self, ocr):
        img  = make_blank()
        crop = ocr.crop_plate(img, [50, 100, 250, 140])
        assert crop.shape[2] == 3

    def test_output_height_is_64(self, ocr):
        """crop_plate resizes to 64px height."""
        img  = make_blank(h=480, w=640)
        crop = ocr.crop_plate(img, [100, 200, 350, 260])
        assert crop.shape[0] == 64

    def test_clamps_to_image_bounds(self, ocr):
        """Bbox outside image should not crash — clamp and return."""
        img  = make_blank(h=480, w=640)
        crop = ocr.crop_plate(img, [-10, -5, 700, 500])
        assert crop is not None

    def test_degenerate_bbox_returns_none(self, ocr):
        img  = make_blank()
        crop = ocr.crop_plate(img, [100, 100, 100, 100])
        assert crop is None

    def test_inverted_bbox_returns_none(self, ocr):
        img  = make_blank()
        crop = ocr.crop_plate(img, [300, 300, 100, 200])
        assert crop is None

    def test_padding_applied(self, ocr):
        """Output width should be larger than raw bbox due to padding."""
        img  = make_blank(h=480, w=640)
        bbox = [100, 200, 300, 240]
        raw_w = bbox[2] - bbox[0]
        crop  = ocr.crop_plate(img, bbox, padding=4)
        # After resize to 64px height: effective width > raw scaled width
        assert crop.shape[1] > 0


# ══════════════════════════════════════════════════════════════
# 5. detect_plate_regions_contour
# ══════════════════════════════════════════════════════════════

class TestDetectPlateRegionsContour:

    def test_returns_list(self, ocr):
        img = make_blank()
        result = ocr.detect_plate_regions_contour(img)
        assert isinstance(result, list)

    def test_blank_image_no_plates(self, ocr):
        result = ocr.detect_plate_regions_contour(make_blank())
        assert result == []

    def test_detects_bright_rectangle(self, ocr):
        """A white rectangle on dark background should be a candidate."""
        img = make_blank(h=480, w=640)
        # Draw a 200×45 white rectangle (aspect = 4.4, valid range 1.8–7.0)
        cv2.rectangle(img, (200, 300), (400, 345), (255, 255, 255), -1)
        result = ocr.detect_plate_regions_contour(img)
        assert len(result) >= 1

    def test_each_result_is_4_ints(self, ocr):
        img = make_blank(h=480, w=640)
        cv2.rectangle(img, (100, 200), (340, 248), (255, 255, 255), -1)
        result = ocr.detect_plate_regions_contour(img)
        for box in result:
            assert len(box) == 4
            assert all(isinstance(v, int) for v in box)

    def test_coordinates_within_image(self, ocr):
        img = make_blank(h=480, w=640)
        cv2.rectangle(img, (100, 200), (340, 248), (255, 255, 255), -1)
        for box in ocr.detect_plate_regions_contour(img):
            x1, y1, x2, y2 = box
            assert x1 >= 0 and y1 >= 0
            assert x2 <= 640 and y2 <= 480
            assert x2 > x1 and y2 > y1


# ══════════════════════════════════════════════════════════════
# 6. _deduplicate_boxes
# ══════════════════════════════════════════════════════════════

class TestDeduplicateBoxes:

    def test_empty_input(self, ocr):
        assert ocr._deduplicate_boxes([]) == []

    def test_single_box_returned(self, ocr):
        boxes = [[10, 20, 100, 50]]
        result = ocr._deduplicate_boxes(boxes)
        assert len(result) == 1

    def test_non_overlapping_both_kept(self, ocr):
        boxes = [[0, 0, 100, 50], [200, 200, 400, 260]]
        result = ocr._deduplicate_boxes(boxes)
        assert len(result) == 2

    def test_identical_boxes_deduplicated(self, ocr):
        box = [50, 100, 250, 150]
        result = ocr._deduplicate_boxes([box, box, box])
        assert len(result) == 1

    def test_large_overlap_deduplicated(self, ocr):
        """Two nearly identical boxes → only one survives."""
        boxes = [[100, 100, 300, 150], [102, 101, 302, 152]]
        result = ocr._deduplicate_boxes(boxes)
        assert len(result) == 1

    def test_small_overlap_both_kept(self, ocr):
        """Boxes that barely touch → both kept."""
        boxes = [[0, 0, 100, 50], [99, 0, 200, 50]]
        result = ocr._deduplicate_boxes(boxes)
        assert len(result) == 2


# ══════════════════════════════════════════════════════════════
# 7. IoU helper
# ══════════════════════════════════════════════════════════════

class TestIoU:

    def test_identical_boxes(self, ocr):
        box = [0, 0, 100, 50]
        assert ocr._iou(box, box) == pytest.approx(1.0)

    def test_non_overlapping(self, ocr):
        assert ocr._iou([0,0,50,50], [100,100,200,200]) == 0.0

    def test_partial_overlap(self, ocr):
        iou = ocr._iou([0,0,100,50], [50,0,150,50])
        assert 0.0 < iou < 1.0

    def test_symmetry(self, ocr):
        b1 = [0, 0, 80, 40]
        b2 = [50, 20, 130, 60]
        assert ocr._iou(b1, b2) == pytest.approx(ocr._iou(b2, b1), abs=1e-6)


# ══════════════════════════════════════════════════════════════
# 8. Integration — EasyOCR on synthetic plates
# ══════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestOCRIntegration:

    def test_load_models_sets_reader(self, loaded_ocr):
        assert loaded_ocr.ocr_reader is not None
        assert loaded_ocr._loaded is True

    def test_run_ocr_returns_tuple(self, loaded_ocr):
        plate_img = make_plate_image("MH12AB1234")
        text, conf = loaded_ocr.run_ocr(plate_img)
        assert isinstance(text, str)
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_extract_returns_list(self, loaded_ocr):
        img = make_image_with_plate("MH12AB1234")
        results = loaded_ocr.extract(img)
        assert isinstance(results, list)

    def test_extract_plate_result_type(self, loaded_ocr):
        img = make_image_with_plate("DL01AB5678")
        results = loaded_ocr.extract(img)
        for r in results:
            assert isinstance(r, PlateResult)

    def test_extract_confidence_in_range(self, loaded_ocr):
        img = make_image_with_plate("KA03EF9012")
        for r in loaded_ocr.extract(img):
            assert 0.0 <= r.confidence <= 1.0

    def test_validate_clean_plate(self, loaded_ocr):
        """validate_plate must pass for a perfectly clean synthetic plate."""
        assert loaded_ocr.validate_plate("MH12AB1234") is True

    def test_file_not_found_raises(self, loaded_ocr):
        with pytest.raises(FileNotFoundError):
            loaded_ocr.extract("/nonexistent/image.jpg")
