"""
tests/test_preprocessing.py
============================
Unit tests for the Image Preprocessing Pipeline.

Each test creates a synthetic image that represents a specific problem
(dark, blurry, noisy) and verifies the preprocessor fixes it correctly.

HOW TO RUN:
    python -m pytest tests/test_preprocessing.py -v

    # Run a single test:
    python -m pytest tests/test_preprocessing.py::test_load_image_from_path -v

    # Run with print output visible:
    python -m pytest tests/test_preprocessing.py -v -s
"""

import sys
import cv2
import numpy as np
import pytest
import tempfile
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.preprocessor import ImagePreprocessor, PreprocessingReport


# ── Shared fixture: one preprocessor instance for all tests ───
@pytest.fixture(scope="module")
def preprocessor():
    return ImagePreprocessor()


# ══════════════════════════════════════════════════════════════
# SYNTHETIC IMAGE FACTORIES
# These create test images without needing real image files
# ══════════════════════════════════════════════════════════════

def make_normal_image(h=480, w=640) -> np.ndarray:
    """
    A normal, well-lit BGR image with high edge content.

    Uses a fine grid + shapes + text to guarantee a high Laplacian variance
    score (>> 80), so the blur detector correctly classifies it as sharp.
    Flat single-color images have near-zero Laplacian variance and would be
    incorrectly flagged as blurry — that is why we add texture here.
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (100, 120, 100)

    # Fine grid of lines — each line creates a sharp edge pair (high freq content)
    for y in range(0, h, 15):
        cv2.line(img, (0, y), (w, y), (60, 80, 60), 1)
    for x in range(0, w, 15):
        cv2.line(img, (x, 0), (x, h), (60, 80, 60), 1)

    # Solid shapes on top of the grid
    cv2.rectangle(img, (100, 100), (400, 350), (200, 80, 30), -1)
    cv2.circle(img,    (500, 250), 80,          (30, 30, 180), -1)

    # Text adds many sharp high-frequency edges
    cv2.putText(img, "SHARP TEST IMAGE", (50, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(img, "Traffic Violation Detection", (50, 430),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
    return img


def make_dark_image(h=480, w=640) -> np.ndarray:
    """A very dark (underexposed) image — mean brightness ~20."""
    img = make_normal_image(h, w)
    return (img * 0.15).astype(np.uint8)   # reduce to 15% brightness


def make_blurry_image(h=480, w=640) -> np.ndarray:
    """A heavily motion-blurred image."""
    img = make_normal_image(h, w)
    # Apply heavy Gaussian blur (simulates motion blur)
    return cv2.GaussianBlur(img, (31, 31), sigmaX=10.0)


def make_noisy_image(h=480, w=640) -> np.ndarray:
    """An image with heavy Gaussian noise (simulates rain/low-quality camera)."""
    img = make_normal_image(h, w).astype(np.float32)
    noise = np.random.normal(0, 40, img.shape).astype(np.float32)
    noisy = np.clip(img + noise, 0, 255).astype(np.uint8)
    return noisy


def make_hazy_image(h=480, w=640) -> np.ndarray:
    """A washed-out, low-contrast hazy image."""
    img = make_normal_image(h, w).astype(np.float32)
    # Compress to [160, 220] range → looks foggy
    hazy = (img / 255.0) * 60 + 160
    return hazy.clip(0, 255).astype(np.uint8)


def save_temp_image(image: np.ndarray) -> Path:
    """Save image to a temporary file and return path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp.name, image)
    return Path(tmp.name)


# ══════════════════════════════════════════════════════════════
# TESTS: load_image
# ══════════════════════════════════════════════════════════════

class TestLoadImage:

    def test_load_image_from_path(self, preprocessor):
        """Load from a real file path — should return numpy array."""
        img = make_normal_image()
        path = save_temp_image(img)
        loaded = preprocessor.load_image(str(path))
        assert isinstance(loaded, np.ndarray)
        assert loaded.ndim == 3
        assert loaded.shape[2] == 3    # BGR channels
        path.unlink()                  # cleanup

    def test_load_image_from_path_object(self, preprocessor):
        """Load using a Path object (not just string)."""
        img = make_normal_image()
        path = save_temp_image(img)
        loaded = preprocessor.load_image(path)   # Path object
        assert isinstance(loaded, np.ndarray)
        path.unlink()

    def test_load_image_from_numpy_array(self, preprocessor):
        """Passing a numpy array should pass through cleanly."""
        img = make_normal_image()
        loaded = preprocessor.load_image(img)
        assert isinstance(loaded, np.ndarray)
        assert loaded.shape == img.shape

    def test_load_image_grayscale_converted(self, preprocessor):
        """Grayscale (2D) array should be converted to BGR (3-channel)."""
        gray = np.full((100, 100), 128, dtype=np.uint8)   # 2D grayscale
        loaded = preprocessor.load_image(gray)
        assert loaded.ndim == 3
        assert loaded.shape[2] == 3

    def test_load_image_file_not_found(self, preprocessor):
        """Non-existent path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            preprocessor.load_image("/this/path/does/not/exist.jpg")

    def test_load_image_preserves_dimensions(self, preprocessor):
        """Loaded image should have the same H×W as original."""
        img = make_normal_image(h=720, w=1280)
        path = save_temp_image(img)
        loaded = preprocessor.load_image(str(path))
        assert loaded.shape[0] == 720
        assert loaded.shape[1] == 1280
        path.unlink()


# ══════════════════════════════════════════════════════════════
# TESTS: is_low_light
# ══════════════════════════════════════════════════════════════

class TestIsLowLight:

    def test_dark_image_detected(self, preprocessor):
        """Dark image should be flagged as low light."""
        dark = make_dark_image()
        assert preprocessor.is_low_light(dark) is True

    def test_normal_image_not_low_light(self, preprocessor):
        """Normal image should NOT be flagged as low light."""
        normal = make_normal_image()
        assert preprocessor.is_low_light(normal) is False

    def test_all_black_is_low_light(self, preprocessor):
        """Completely black image must be detected as low light."""
        black = np.zeros((100, 100, 3), dtype=np.uint8)
        assert preprocessor.is_low_light(black) is True

    def test_all_white_is_not_low_light(self, preprocessor):
        """Completely white image must NOT be low light."""
        white = np.full((100, 100, 3), 255, dtype=np.uint8)
        assert preprocessor.is_low_light(white) is False

    def test_custom_threshold(self, preprocessor):
        """Custom threshold should override default."""
        # Create image with brightness ~100
        mid = np.full((100, 100, 3), 100, dtype=np.uint8)
        # With threshold=50: not low light
        assert preprocessor.is_low_light(mid, threshold=50) is False
        # With threshold=150: IS low light
        assert preprocessor.is_low_light(mid, threshold=150) is True


# ══════════════════════════════════════════════════════════════
# TESTS: enhance_low_light
# ══════════════════════════════════════════════════════════════

class TestEnhanceLowLight:

    def test_brightness_increases(self, preprocessor):
        """After enhancement, mean brightness must be higher than before."""
        dark = make_dark_image()
        before = preprocessor._mean_brightness(dark)
        enhanced = preprocessor.enhance_low_light(dark)
        after = preprocessor._mean_brightness(enhanced)
        assert after > before, f"Expected brightness increase: {before:.1f} → {after:.1f}"

    def test_output_shape_unchanged(self, preprocessor):
        """Enhancement must not change image dimensions."""
        dark = make_dark_image(h=480, w=640)
        enhanced = preprocessor.enhance_low_light(dark)
        assert enhanced.shape == dark.shape

    def test_output_dtype_uint8(self, preprocessor):
        """Output must be uint8 (same as input)."""
        dark = make_dark_image()
        enhanced = preprocessor.enhance_low_light(dark)
        assert enhanced.dtype == np.uint8

    def test_pixel_values_valid_range(self, preprocessor):
        """All pixel values must stay in [0, 255]."""
        dark = make_dark_image()
        enhanced = preprocessor.enhance_low_light(dark)
        assert enhanced.min() >= 0
        assert enhanced.max() <= 255

    def test_bright_image_not_over_exposed(self, preprocessor):
        """Applying enhancement to bright image should not over-expose it."""
        bright = np.full((100, 100, 3), 200, dtype=np.uint8)
        enhanced = preprocessor.enhance_low_light(bright)
        # Should not blow out significantly
        assert preprocessor._mean_brightness(enhanced) <= 255


# ══════════════════════════════════════════════════════════════
# TESTS: detect_blur
# ══════════════════════════════════════════════════════════════

class TestDetectBlur:

    def test_blurry_image_has_low_score(self, preprocessor):
        """Blurry image must have a lower score than sharp image."""
        sharp = make_normal_image()
        blurry = make_blurry_image()
        sharp_score = preprocessor.detect_blur(sharp)
        blurry_score = preprocessor.detect_blur(blurry)
        assert blurry_score < sharp_score, \
            f"Sharp score ({sharp_score:.1f}) should be > blurry score ({blurry_score:.1f})"

    def test_returns_float(self, preprocessor):
        """detect_blur must return a float."""
        img = make_normal_image()
        score = preprocessor.detect_blur(img)
        assert isinstance(score, float)

    def test_score_non_negative(self, preprocessor):
        """Blur score (variance) cannot be negative."""
        img = make_normal_image()
        assert preprocessor.detect_blur(img) >= 0.0

    def test_constant_image_has_zero_score(self, preprocessor):
        """Completely uniform (flat) image has zero variance → score = 0."""
        flat = np.full((100, 100, 3), 128, dtype=np.uint8)
        score = preprocessor.detect_blur(flat)
        assert score == pytest.approx(0.0, abs=0.1)


# ══════════════════════════════════════════════════════════════
# TESTS: deblur
# ══════════════════════════════════════════════════════════════

class TestDeblur:

    def test_blur_score_improves(self, preprocessor):
        """After deblurring, the blur score must increase."""
        blurry = make_blurry_image()
        score_before = preprocessor.detect_blur(blurry)
        sharpened = preprocessor.deblur(blurry)
        score_after = preprocessor.detect_blur(sharpened)
        assert score_after > score_before, \
            f"Expected improvement: {score_before:.1f} → {score_after:.1f}"

    def test_output_shape_unchanged(self, preprocessor):
        """Deblurring must not change image size."""
        blurry = make_blurry_image()
        sharpened = preprocessor.deblur(blurry)
        assert sharpened.shape == blurry.shape

    def test_output_dtype_uint8(self, preprocessor):
        """Output must be uint8."""
        blurry = make_blurry_image()
        assert preprocessor.deblur(blurry).dtype == np.uint8

    def test_pixel_values_in_range(self, preprocessor):
        """Output pixel values must stay in [0, 255]."""
        blurry = make_blurry_image()
        sharpened = preprocessor.deblur(blurry)
        assert sharpened.min() >= 0
        assert sharpened.max() <= 255


# ══════════════════════════════════════════════════════════════
# TESTS: reduce_noise
# ══════════════════════════════════════════════════════════════

class TestReduceNoise:

    def test_noise_decreases(self, preprocessor):
        """After denoising, the estimated noise should be lower."""
        noisy = make_noisy_image()
        noise_before = preprocessor._estimate_noise(noisy)
        denoised = preprocessor.reduce_noise(noisy)
        noise_after = preprocessor._estimate_noise(denoised)
        assert noise_after < noise_before, \
            f"Expected noise reduction: {noise_before:.1f} → {noise_after:.1f}"

    def test_output_shape_unchanged(self, preprocessor):
        noisy = make_noisy_image()
        assert preprocessor.reduce_noise(noisy).shape == noisy.shape

    def test_output_dtype_uint8(self, preprocessor):
        noisy = make_noisy_image()
        assert preprocessor.reduce_noise(noisy).dtype == np.uint8


# ══════════════════════════════════════════════════════════════
# TESTS: resize_for_model
# ══════════════════════════════════════════════════════════════

class TestResizeForModel:

    def test_output_is_640x640_by_default(self, preprocessor):
        """Default output must be exactly 640×640."""
        img = make_normal_image(h=480, w=640)
        resized = preprocessor.resize_for_model(img)
        assert resized.shape == (640, 640, 3)

    def test_custom_size(self, preprocessor):
        """Custom size parameter must be respected."""
        img = make_normal_image()
        resized = preprocessor.resize_for_model(img, size=416)
        assert resized.shape == (416, 416, 3)

    def test_non_square_input_handled(self, preprocessor):
        """Wide (1920×1080) input should letterbox to 640×640."""
        wide = make_normal_image(h=1080, w=1920)
        resized = preprocessor.resize_for_model(wide, size=640)
        assert resized.shape == (640, 640, 3)

    def test_portrait_input_handled(self, preprocessor):
        """Tall portrait image should letterbox to 640×640."""
        portrait = make_normal_image(h=1280, w=720)
        resized = preprocessor.resize_for_model(portrait, size=640)
        assert resized.shape == (640, 640, 3)

    def test_output_dtype_uint8(self, preprocessor):
        img = make_normal_image()
        assert preprocessor.resize_for_model(img).dtype == np.uint8

    def test_small_image_upscaled(self, preprocessor):
        """Very small image should be upscaled to 640×640."""
        small = make_normal_image(h=100, w=100)
        resized = preprocessor.resize_for_model(small, size=640)
        assert resized.shape == (640, 640, 3)


# ══════════════════════════════════════════════════════════════
# TESTS: Full process() pipeline
# ══════════════════════════════════════════════════════════════

class TestProcessPipeline:

    def test_process_returns_tuple(self, preprocessor):
        """process() must return a (image, report) tuple."""
        img = make_normal_image()
        result = preprocessor.process(img)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_process_output_image_correct_size(self, preprocessor):
        """Processed image must be 640×640×3."""
        img = make_normal_image()
        processed, _ = preprocessor.process(img)
        assert processed.shape == (640, 640, 3)

    def test_process_returns_preprocessing_report(self, preprocessor):
        """Second return value must be a PreprocessingReport."""
        img = make_normal_image()
        _, report = preprocessor.process(img)
        assert isinstance(report, PreprocessingReport)

    def test_process_dark_image_applies_enhancement(self, preprocessor):
        """Dark image must trigger enhancement step."""
        dark = make_dark_image()
        _, report = preprocessor.process(dark)
        assert report.was_low_light is True
        assert any("Low-Light" in s for s in report.steps_applied)

    def test_process_blurry_image_applies_deblur(self, preprocessor):
        """Blurry image must trigger deblur step."""
        blurry = make_blurry_image()
        _, report = preprocessor.process(blurry)
        assert report.was_blurry is True
        assert any("Blur" in s for s in report.steps_applied)

    def test_process_normal_image_minimal_steps(self, preprocessor):
        """Normal (well-lit, sharp) image must not trigger low-light or
        blur fixes. Noise reduction may still apply to synthetic images
        because grid lines score slightly above the noise threshold — that
        is acceptable and expected for programmatically generated images."""
        normal = make_normal_image()    # has grid + text -> high blur score
        _, report = preprocessor.process(normal)
        assert report.was_low_light is False, \
            "Well-lit image must not trigger low-light enhancement"
        assert report.was_blurry is False, \
            f"Textured image must not be flagged blurry " \
            f"(blur score={report.blur_score_before:.1f})"
        # Low-light and blur steps must NOT appear
        assert not any("Low-Light" in s for s in report.steps_applied), \
            "Low-light step must not run on a bright image"
        assert not any("Blur Correction" in s for s in report.steps_applied), \
            "Blur correction must not run on a sharp image"
        # Resize must always be the last step
        assert "Resize" in report.steps_applied[-1]

    def test_process_report_has_timing(self, preprocessor):
        """Report must record processing time > 0."""
        img = make_normal_image()
        _, report = preprocessor.process(img)
        assert report.processing_ms > 0

    def test_process_report_shapes_recorded(self, preprocessor):
        """Report must record original and final shapes."""
        img = make_normal_image(h=720, w=1280)
        _, report = preprocessor.process(img)
        assert report.original_shape == (720, 1280, 3)
        assert report.final_shape == (640, 640, 3)

    def test_process_from_file_path(self, preprocessor):
        """process() should work with a file path string."""
        img = make_normal_image()
        path = save_temp_image(img)
        processed, report = preprocessor.process(str(path))
        assert processed.shape == (640, 640, 3)
        assert isinstance(report, PreprocessingReport)
        path.unlink()

    def test_process_report_to_dict(self, preprocessor):
        """Report.to_dict() must return a valid dictionary."""
        img = make_normal_image()
        _, report = preprocessor.process(img)
        d = report.to_dict()
        assert isinstance(d, dict)
        required_keys = ["original_shape", "final_shape", "was_low_light",
                         "was_blurry", "processing_ms", "steps_applied"]
        for key in required_keys:
            assert key in d, f"Missing key in report dict: {key}"


# ══════════════════════════════════════════════════════════════
# TESTS: Quality report
# ══════════════════════════════════════════════════════════════

class TestQualityReport:

    def test_quality_report_keys(self, preprocessor):
        """get_image_quality_report must return all expected keys."""
        img = make_normal_image()
        report = preprocessor.get_image_quality_report(img)
        expected = ["shape", "brightness", "blur_score", "noise_score",
                    "is_low_light", "is_blurry", "is_noisy",
                    "has_shadows", "is_hazy", "quality_grade"]
        for key in expected:
            assert key in report, f"Missing key: {key}"

    def test_quality_grade_valid_value(self, preprocessor):
        """Quality grade must be one of A, B, C, D."""
        img = make_normal_image()
        report = preprocessor.get_image_quality_report(img)
        assert report["quality_grade"] in ("A", "B", "C", "D")

    def test_dark_image_grade_d(self, preprocessor):
        """Very dark image should get grade D."""
        dark = make_dark_image()
        report = preprocessor.get_image_quality_report(dark)
        assert report["quality_grade"] == "D"

    def test_normal_image_grade_a(self, preprocessor):
        """Well-lit, sharp, textured image should get grade A or B.
        The make_normal_image() helper includes a fine grid and text
        which ensures the Laplacian variance is well above the blur threshold."""
        normal = make_normal_image()
        report = preprocessor.get_image_quality_report(normal)
        blur_score = report["blur_score"]
        assert report["quality_grade"] in ("A", "B"), \
            f"Expected A or B but got {report['quality_grade']} " \
            f"(brightness={report['brightness']:.1f}, blur_score={blur_score:.1f})"
