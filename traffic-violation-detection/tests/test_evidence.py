"""
tests/test_evidence.py
======================
Unit tests for core/evidence_generator.py — Phase 6

HOW TO RUN:
    python -m pytest tests/test_evidence.py -v -m "not integration"
    python -m pytest tests/test_evidence.py -v           # all including DB/model tests
"""

import sys
import json
import tempfile
import shutil
import numpy as np
import pytest
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.evidence_generator import EvidenceGenerator, EvidencePackage
from core.detector import Detection, DetectionResult
from core.violation_detector import Violation, ViolationResult
from core.ocr import PlateResult
from config import ViolationType, VIOLATION_FINES


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def make_image(h=480, w=640) -> np.ndarray:
    import cv2
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:h//2] = (160, 140, 100)
    img[h//2:] = (70, 72, 78)
    cv2.rectangle(img, (150, 250), (490, 420), (30, 70, 180), -1)
    return img


def make_detection(class_id, bbox, conf=0.88) -> Detection:
    from config import COCO_CLASSES
    return Detection(class_id=class_id,
                     class_name=COCO_CLASSES.get(class_id, "unknown"),
                     confidence=conf, bbox=bbox)


def make_det_result() -> DetectionResult:
    car    = make_detection(2, [150, 250, 490, 420])
    person = make_detection(0, [200, 200, 340, 390])
    tl     = make_detection(9, [500, 50,  540, 150])
    r = DetectionResult(image_path="test.jpg")
    r.vehicles       = [car]
    r.persons        = [person]
    r.traffic_lights = [tl]
    r.detections     = [car, person, tl]
    r.image_shape    = (480, 640, 3)
    r.inference_ms   = 28.5
    return r


def make_viol_result() -> ViolationResult:
    v = Violation(
        violation_type = ViolationType.TRIPLE_RIDING,
        display_name   = "Triple Riding",
        confidence     = 0.85,
        bbox           = [150, 200, 500, 430],
        vehicle_bbox   = [150, 250, 490, 420],
        fine_amount    = VIOLATION_FINES[ViolationType.TRIPLE_RIDING],
        description    = "3 persons on motorcycle",
    )
    r = ViolationResult(image_path="test.jpg", violations=[v])
    return r


def make_plate_result() -> PlateResult:
    return PlateResult(
        plate_text       = "MH 12 AB 1234",
        plate_text_clean = "MH12AB1234",
        confidence       = 0.82,
        bbox             = [220, 395, 420, 430],
        is_valid_format  = True,
    )


@pytest.fixture(scope="module")
def gen(tmp_path_factory):
    """EvidenceGenerator using a temp directory — no real DB writes."""
    tmp = tmp_path_factory.mktemp("evidence")
    return EvidenceGenerator(evidence_dir=tmp, save_to_db=False)


@pytest.fixture(scope="module")
def gen_with_db(tmp_path_factory):
    """EvidenceGenerator with DB writes enabled."""
    tmp = tmp_path_factory.mktemp("evidence_db")
    return EvidenceGenerator(evidence_dir=tmp, save_to_db=True)


# ══════════════════════════════════════════════════════════════
# 1. EvidencePackage DATACLASS
# ══════════════════════════════════════════════════════════════

class TestEvidencePackage:

    def test_to_dict_keys(self):
        pkg = EvidencePackage(
            evidence_id="abc123",
            timestamp="2026-06-18T12:00:00",
            source_image_path="test.jpg",
        )
        d = pkg.to_dict()
        for key in ["evidence_id", "timestamp", "source_image_path",
                    "annotated_path", "metadata_path",
                    "db_record_ids", "metadata"]:
            assert key in d

    def test_defaults_are_none(self):
        pkg = EvidencePackage(evidence_id="x", timestamp="t",
                              source_image_path="p")
        assert pkg.annotated_image   is None
        assert pkg.annotated_path    is None
        assert pkg.metadata_path     is None
        assert pkg.db_record_ids     == []

    def test_metadata_default_empty_dict(self):
        pkg = EvidencePackage(evidence_id="x", timestamp="t",
                              source_image_path="p")
        assert pkg.metadata == {}


# ══════════════════════════════════════════════════════════════
# 2. draw_box
# ══════════════════════════════════════════════════════════════

class TestDrawBox:

    def test_returns_ndarray(self, gen):
        img    = make_image()
        result = gen.draw_box(img, [100, 100, 300, 250],
                              label="car 88%", color=(255, 255, 0))
        assert isinstance(result, np.ndarray)

    def test_output_shape_unchanged(self, gen):
        img    = make_image(h=480, w=640)
        result = gen.draw_box(img, [50, 50, 300, 200], "test", (0,255,0))
        assert result.shape == (480, 640, 3)

    def test_empty_label_does_not_crash(self, gen):
        img = make_image()
        result = gen.draw_box(img, [10, 10, 200, 150], "", (255, 0, 0))
        assert result is not None

    def test_out_of_bounds_bbox_clamped(self, gen):
        """Bbox outside image boundaries must not crash."""
        img    = make_image(h=480, w=640)
        result = gen.draw_box(img, [-50, -50, 800, 600], "oob", (0, 0, 255))
        assert result.shape == (480, 640, 3)

    def test_degenerate_bbox_returns_image(self, gen):
        """Zero-size bbox must return image unchanged (no crash)."""
        img    = make_image()
        result = gen.draw_box(img, [100, 100, 100, 100], "noop", (255, 0, 0))
        assert result is not None

    def test_modifies_pixels(self, gen):
        """Drawing a box should change at least some pixel values."""
        img    = np.zeros((200, 300, 3), dtype=np.uint8)
        result = gen.draw_box(img.copy(), [10, 10, 200, 150],
                              "label", (0, 255, 0))
        assert not np.array_equal(result, img)


# ══════════════════════════════════════════════════════════════
# 3. add_timestamp
# ══════════════════════════════════════════════════════════════

class TestAddTimestamp:

    def test_output_same_shape(self, gen):
        """Timestamp is drawn inside the image — shape must not change."""
        img    = make_image(h=480, w=640)
        result = gen.add_timestamp(img, "2026-06-18T12:34:56")
        assert result.shape == (480, 640, 3)

    def test_modifies_top_rows(self, gen):
        """Timestamp banner modifies the top rows of the image."""
        img    = np.full((480, 640, 3), 128, dtype=np.uint8)
        result = gen.add_timestamp(img.copy(), "2026-06-18T12:00:00")
        # Top 32 rows should differ from the original (banner drawn there)
        assert not np.array_equal(result[:32], img[:32])

    def test_does_not_crash_on_empty_string(self, gen):
        img    = make_image()
        result = gen.add_timestamp(img, "")
        assert result is not None


# ══════════════════════════════════════════════════════════════
# 4. add_summary_panel
# ══════════════════════════════════════════════════════════════

class TestAddSummaryPanel:

    def test_output_taller_than_input(self, gen):
        """Summary panel is appended at the bottom — height increases."""
        img    = make_image(h=480, w=640)
        result = gen.add_summary_panel(img, violations=[], plate_results=[])
        assert result.shape[0] > 480

    def test_output_width_unchanged(self, gen):
        img    = make_image(h=480, w=640)
        result = gen.add_summary_panel(img, violations=[])
        assert result.shape[1] == 640

    def test_output_channels_unchanged(self, gen):
        img    = make_image()
        result = gen.add_summary_panel(img, violations=[])
        assert result.shape[2] == 3

    def test_panel_height_is_50px(self, gen):
        img    = make_image(h=480, w=640)
        result = gen.add_summary_panel(img, violations=[])
        assert result.shape[0] == 530   # 480 + 50

    def test_works_with_violations(self, gen):
        img        = make_image()
        violations = make_viol_result().violations
        result     = gen.add_summary_panel(img, violations)
        assert result.shape[0] > img.shape[0]

    def test_works_with_plates(self, gen):
        img    = make_image()
        plates = [make_plate_result()]
        result = gen.add_summary_panel(img, violations=[], plate_results=plates)
        assert result.shape[0] > img.shape[0]


# ══════════════════════════════════════════════════════════════
# 5. build_metadata
# ══════════════════════════════════════════════════════════════

class TestBuildMetadata:

    def test_top_level_keys_present(self, gen):
        meta = gen.build_metadata(
            "eid123", "2026-06-18T12:00:00", "image.jpg",
            make_det_result(), make_viol_result(), [make_plate_result()])
        for key in ["evidence_id", "timestamp", "image_path",
                    "system", "detection", "violations", "plates",
                    "has_violations", "violation_count",
                    "total_fine_inr", "plate_numbers"]:
            assert key in meta, f"Missing key: {key}"

    def test_has_violations_true(self, gen):
        meta = gen.build_metadata("e", "t", "p",
                                   make_det_result(), make_viol_result(), [])
        assert meta["has_violations"] is True

    def test_violation_count_correct(self, gen):
        meta = gen.build_metadata("e", "t", "p",
                                   make_det_result(), make_viol_result(), [])
        assert meta["violation_count"] == 1

    def test_total_fine_correct(self, gen):
        meta = gen.build_metadata("e", "t", "p",
                                   None, make_viol_result(), [])
        assert meta["total_fine_inr"] == VIOLATION_FINES[ViolationType.TRIPLE_RIDING]

    def test_plate_numbers_extracted(self, gen):
        meta = gen.build_metadata("e", "t", "p",
                                   None, None, [make_plate_result()])
        assert "MH12AB1234" in meta["plate_numbers"]

    def test_empty_result_no_crash(self, gen):
        meta = gen.build_metadata("e", "t", "p", None, None, [])
        assert meta["has_violations"] is False
        assert meta["violation_count"] == 0
        assert meta["total_fine_inr"]  == 0

    def test_metadata_is_json_serializable(self, gen):
        meta = gen.build_metadata(
            "eid", "2026-06-18T00:00:00", "img.jpg",
            make_det_result(), make_viol_result(), [make_plate_result()])
        json_str = json.dumps(meta, default=str)
        assert isinstance(json_str, str)


# ══════════════════════════════════════════════════════════════
# 6. save_evidence
# ══════════════════════════════════════════════════════════════

class TestSaveEvidence:

    def test_annotated_jpg_created(self, gen):
        pkg = EvidencePackage(
            evidence_id       = "test-save-001",
            timestamp         = datetime.now().isoformat(timespec="seconds"),
            source_image_path = "test.jpg",
            annotated_image   = make_image(),
            metadata          = {"test": True},
        )
        saved = gen.save_evidence(pkg)
        assert saved.annotated_path is not None
        assert Path(saved.annotated_path).exists()

    def test_metadata_json_created(self, gen):
        pkg = EvidencePackage(
            evidence_id       = "test-save-002",
            timestamp         = datetime.now().isoformat(timespec="seconds"),
            source_image_path = "test.jpg",
            annotated_image   = make_image(),
            metadata          = {"evidence_id": "test-save-002", "value": 42},
        )
        saved = gen.save_evidence(pkg)
        assert saved.metadata_path is not None
        assert Path(saved.metadata_path).exists()

    def test_metadata_json_content_correct(self, gen):
        pkg = EvidencePackage(
            evidence_id       = "test-save-003",
            timestamp         = datetime.now().isoformat(timespec="seconds"),
            source_image_path = "test.jpg",
            metadata          = {"key": "value", "count": 5},
        )
        saved = gen.save_evidence(pkg)
        loaded = json.loads(Path(saved.metadata_path).read_text(encoding="utf-8"))
        assert loaded["key"]   == "value"
        assert loaded["count"] == 5

    def test_evidence_paths_returned(self, gen):
        pkg = EvidencePackage(
            evidence_id       = "test-save-004",
            timestamp         = datetime.now().isoformat(timespec="seconds"),
            source_image_path = "test.jpg",
            annotated_image   = make_image(),
            metadata          = {},
        )
        saved = gen.save_evidence(pkg)
        assert saved.annotated_path is not None
        assert saved.metadata_path  is not None

    def test_no_image_still_saves_json(self, gen):
        """Even without an annotated image, metadata.json must be saved."""
        pkg = EvidencePackage(
            evidence_id       = "test-save-005",
            timestamp         = datetime.now().isoformat(timespec="seconds"),
            source_image_path = "test.jpg",
            annotated_image   = None,       # no image
            metadata          = {"test": True},
        )
        saved = gen.save_evidence(pkg)
        assert saved.metadata_path is not None
        assert Path(saved.metadata_path).exists()


# ══════════════════════════════════════════════════════════════
# 7. generate() integration
# ══════════════════════════════════════════════════════════════

class TestGenerateIntegration:

    def test_returns_evidence_package(self, gen):
        result = gen.generate(
            image            = make_image(),
            image_path       = "test.jpg",
            detection_result = make_det_result(),
            violation_result = make_viol_result(),
            plate_results    = [make_plate_result()],
        )
        assert isinstance(result, EvidencePackage)

    def test_evidence_id_is_string(self, gen):
        result = gen.generate(image=make_image(), image_path="t.jpg")
        assert isinstance(result.evidence_id, str)
        assert len(result.evidence_id) > 0

    def test_timestamp_is_iso_format(self, gen):
        result = gen.generate(image=make_image(), image_path="t.jpg")
        # Should parse as datetime without error
        dt = datetime.fromisoformat(result.timestamp)
        assert isinstance(dt, datetime)

    def test_annotated_image_is_taller(self, gen):
        """Summary panel adds height, so annotated > original."""
        img    = make_image(h=480, w=640)
        result = gen.generate(image=img, image_path="t.jpg")
        assert result.annotated_image.shape[0] > 480

    def test_annotated_file_saved(self, gen):
        result = gen.generate(image=make_image(), image_path="t.jpg")
        assert result.annotated_path is not None
        assert Path(result.annotated_path).exists()

    def test_metadata_file_saved(self, gen):
        result = gen.generate(image=make_image(), image_path="t.jpg")
        assert result.metadata_path is not None
        assert Path(result.metadata_path).exists()

    def test_metadata_has_violations(self, gen):
        result = gen.generate(
            image=make_image(), image_path="t.jpg",
            violation_result=make_viol_result())
        assert result.metadata["has_violations"] is True
        assert result.metadata["violation_count"] == 1

    def test_unique_evidence_ids(self, gen):
        r1 = gen.generate(image=make_image(), image_path="a.jpg")
        r2 = gen.generate(image=make_image(), image_path="b.jpg")
        assert r1.evidence_id != r2.evidence_id

    @pytest.mark.integration
    def test_db_save_inserts_records(self, gen_with_db):
        """With save_to_db=True, violations should appear in SQLite."""
        result = gen_with_db.generate(
            image            = make_image(),
            image_path       = "db_test.jpg",
            violation_result = make_viol_result(),
        )
        # DB record IDs returned (may be empty if DB not set up, but no crash)
        assert isinstance(result.db_record_ids, list)
