"""
core/preprocessor.py
====================
Image Preprocessing Pipeline — Phase 2 Implementation

Handles all image quality problems before feeding to detection models:

Problem              Solution Used
──────────────────   ─────────────────────────────────────────────────
Low light / dark     CLAHE on LAB L-channel (preserves color)
Motion blur          Unsharp masking (sharpening kernel)
Noise / rain         Non-local Means Denoising (NLM)
Shadows              Illumination normalization via HLS color space
Haze / fog           Contrast stretch + CLAHE
General resize       Letterbox resize → 640×640 (preserves aspect ratio)

All fixes are applied ONLY when needed — detected automatically.
"""

import cv2
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Union, Tuple, List, Optional

# ── Type alias ────────────────────────────────────────────────
ImageInput = Union[str, Path, np.ndarray]


# ══════════════════════════════════════════════════════════════
# PREPROCESSING REPORT
# ══════════════════════════════════════════════════════════════

@dataclass
class PreprocessingReport:
    """
    Documents every change made to an image during preprocessing.
    Shown in the dashboard to explain what the system did.
    """
    original_shape:    Tuple[int, int, int] = (0, 0, 3)
    final_shape:       Tuple[int, int, int] = (0, 0, 3)
    was_low_light:     bool  = False
    was_blurry:        bool  = False
    was_noisy:         bool  = False
    had_shadows:       bool  = False
    was_hazy:          bool  = False
    brightness_before: float = 0.0
    brightness_after:  float = 0.0
    blur_score_before: float = 0.0
    blur_score_after:  float = 0.0
    noise_score:       float = 0.0
    processing_ms:     float = 0.0
    steps_applied:     List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "original_shape":    list(self.original_shape),
            "final_shape":       list(self.final_shape),
            "was_low_light":     self.was_low_light,
            "was_blurry":        self.was_blurry,
            "was_noisy":         self.was_noisy,
            "had_shadows":       self.had_shadows,
            "was_hazy":          self.was_hazy,
            "brightness_before": round(self.brightness_before, 2),
            "brightness_after":  round(self.brightness_after, 2),
            "blur_score_before": round(self.blur_score_before, 2),
            "blur_score_after":  round(self.blur_score_after, 2),
            "processing_ms":     round(self.processing_ms, 2),
            "steps_applied":     self.steps_applied,
        }

    def summary(self) -> str:
        if not self.steps_applied:
            return "No preprocessing needed — image quality was good."
        return "Applied: " + " | ".join(self.steps_applied)


# ══════════════════════════════════════════════════════════════
# MAIN PREPROCESSOR CLASS
# ══════════════════════════════════════════════════════════════

class ImagePreprocessor:
    """
    Preprocesses traffic images for optimal YOLOv8 detection.

    Usage:
        preprocessor = ImagePreprocessor()

        # Full auto pipeline (recommended):
        image, report = preprocessor.process("path/to/image.jpg")

        # Individual methods:
        img = preprocessor.load_image("path/to/image.jpg")
        if preprocessor.is_low_light(img):
            img = preprocessor.enhance_low_light(img)
    """

    # ── Thresholds (tuned for traffic images) ─────────────────
    LOW_LIGHT_THRESHOLD   = 60    # Mean L-channel brightness < this → dark image
    BLUR_THRESHOLD        = 80.0  # Laplacian variance < this → blurry image
    NOISE_THRESHOLD       = 15.0  # Estimated noise std > this → noisy
    HAZE_THRESHOLD        = 180   # Mean brightness > this AND low contrast → hazy
    SHADOW_STD_THRESHOLD  = 40.0  # Low std in L-channel → uneven illumination

    def __init__(self,
                 target_size: int = 640,
                 low_light_threshold: int = 60,
                 blur_threshold: float = 80.0):
        """
        Args:
            target_size          : Model input size (YOLOv8 = 640)
            low_light_threshold  : Override default brightness threshold
            blur_threshold       : Override default blur threshold
        """
        self.target_size         = target_size
        self.LOW_LIGHT_THRESHOLD = low_light_threshold
        self.BLUR_THRESHOLD      = blur_threshold

        # CLAHE object — reused for efficiency
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # ══════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════

    def process(self, image_input: ImageInput) -> Tuple[np.ndarray, PreprocessingReport]:
        """
        Full auto-preprocessing pipeline.

        Detects image quality issues and applies the appropriate fixes
        in the correct order. Returns the processed image and a report.

        Pipeline order (order matters):
          load → analyse → low-light fix → shadow fix → haze fix
          → noise reduction → blur fix → resize

        Args:
            image_input: File path (str/Path) or numpy BGR array

        Returns:
            (processed_image, PreprocessingReport)
        """
        start = time.time()
        report = PreprocessingReport()

        # ── 1. Load ────────────────────────────────────────────
        image = self.load_image(image_input)
        report.original_shape = image.shape

        # ── 2. Analyse BEFORE state ────────────────────────────
        report.brightness_before = self._mean_brightness(image)
        report.blur_score_before = self.detect_blur(image)
        report.noise_score       = self._estimate_noise(image)

        # ── 3. Detect issues ───────────────────────────────────
        report.was_low_light = self.is_low_light(image)
        report.was_blurry    = report.blur_score_before < self.BLUR_THRESHOLD
        report.was_noisy     = report.noise_score > self.NOISE_THRESHOLD
        report.had_shadows   = self._has_shadows(image)
        report.was_hazy      = self._is_hazy(image)

        # ── 4. Fix low light (do this FIRST — affects everything else) ──
        if report.was_low_light:
            image = self.enhance_low_light(image)
            report.steps_applied.append("Low-Light Enhancement (CLAHE)")

        # ── 5. Fix shadows / uneven illumination ──────────────
        if report.had_shadows:
            image = self.remove_shadows(image)
            report.steps_applied.append("Shadow Normalization")

        # ── 6. Fix haze / fog ─────────────────────────────────
        if report.was_hazy:
            image = self.dehaze(image)
            report.steps_applied.append("Haze/Fog Reduction")

        # ── 7. Reduce noise (before sharpening — order critical) ─
        if report.was_noisy:
            image = self.reduce_noise(image)
            report.steps_applied.append("Noise Reduction (NLM)")

        # ── 8. Fix blur (last, so we sharpen the clean image) ─
        if report.was_blurry:
            image = self.deblur(image)
            report.steps_applied.append("Blur Correction (Unsharp Mask)")

        # ── 9. Resize to model input size ─────────────────────
        image = self.resize_for_model(image, self.target_size)
        report.steps_applied.append(f"Resize → {self.target_size}×{self.target_size}")

        # ── 10. Record AFTER state ─────────────────────────────
        report.final_shape       = image.shape
        report.brightness_after  = self._mean_brightness(image)
        report.blur_score_after  = self.detect_blur(image)
        report.processing_ms     = (time.time() - start) * 1000

        return image, report

    def process_batch(self, image_paths: List[ImageInput]) \
            -> List[Tuple[np.ndarray, PreprocessingReport]]:
        """
        Process a list of images. Returns list of (image, report) tuples.

        Args:
            image_paths: List of file paths or numpy arrays

        Returns:
            List of (processed_image, PreprocessingReport) tuples
        """
        results = []
        for img_input in image_paths:
            try:
                processed, report = self.process(img_input)
                results.append((processed, report))
            except Exception as e:
                print(f"  WARN  Failed to process {img_input}: {e}")
        return results

    # ══════════════════════════════════════════════════════════
    # LOAD
    # ══════════════════════════════════════════════════════════

    def load_image(self, image_input: ImageInput) -> np.ndarray:
        """
        Load an image from disk or pass through a numpy array.

        Accepts:
          - str path:   "C:/images/traffic.jpg"
          - Path object: Path("C:/images/traffic.jpg")
          - numpy array: already-loaded BGR image

        Returns:
            BGR numpy array (height, width, 3), dtype=uint8

        Raises:
            FileNotFoundError if path doesn't exist
            ValueError if image cannot be decoded
        """
        if isinstance(image_input, np.ndarray):
            # Already an image — just validate it
            if image_input.ndim == 2:
                # Grayscale → convert to BGR
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            if image_input.shape[2] == 4:
                # RGBA → BGR
                return cv2.cvtColor(image_input, cv2.COLOR_RGBA2BGR)
            return image_input.copy()

        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        # cv2.imread returns None if the file is unreadable
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(
                f"OpenCV could not decode image: {path}\n"
                f"Supported formats: JPG, PNG, BMP, TIFF, WebP"
            )
        return image

    # ══════════════════════════════════════════════════════════
    # DETECTION METHODS  (is the image problematic?)
    # ══════════════════════════════════════════════════════════

    def is_low_light(self, image: np.ndarray,
                     threshold: Optional[int] = None) -> bool:
        """
        Returns True if the image is too dark.

        Algorithm:
          Convert to LAB color space. The L channel represents lightness
          on a scale of 0 (black) to 255 (white). If mean L < threshold,
          the image is considered under-exposed.

        Args:
            image    : BGR numpy array
            threshold: Override default LOW_LIGHT_THRESHOLD

        Returns:
            True if mean brightness is below threshold
        """
        thresh = threshold or self.LOW_LIGHT_THRESHOLD
        return self._mean_brightness(image) < thresh

    def detect_blur(self, image: np.ndarray) -> float:
        """
        Measures how blurry an image is using Laplacian variance.

        Algorithm:
          Convert to grayscale. Apply Laplacian filter (edge detector).
          A sharp image has high variance (strong edges).
          A blurry image has low variance (soft/smeared edges).

        Score interpretation:
          < 50    → Very blurry
          50–100  → Blurry
          100–200 → Acceptable
          > 200   → Sharp

        Args:
            image: BGR numpy array

        Returns:
            Laplacian variance score (higher = sharper)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # ══════════════════════════════════════════════════════════
    # FIX METHODS  (apply the enhancement)
    # ══════════════════════════════════════════════════════════

    def enhance_low_light(self, image: np.ndarray) -> np.ndarray:
        """
        Brightens dark images while preserving natural colors.

        Algorithm: CLAHE on the L channel of LAB color space.
          - LAB separates lightness (L) from color (A, B channels)
          - Applying CLAHE only to L fixes brightness WITHOUT distorting colors
          - CLAHE (Contrast Limited AHE) avoids over-brightening bright areas

        Args:
            image: BGR numpy array (dark image)

        Returns:
            Brightened BGR numpy array
        """
        # Convert BGR → LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Split into L, A, B channels
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to L (lightness) channel only
        l_enhanced = self.clahe.apply(l_channel)

        # Merge back: enhanced L + original A, B
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])

        # Convert back to BGR
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    def reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Removes noise, grain, and rain artifacts from images.

        Algorithm: Non-Local Means Denoising (NLM)
          - Compares each pixel to similar pixels elsewhere in the image
          - Averages similar patches → removes random noise
          - Preserves edges better than simple Gaussian blur

        Args:
            image: BGR numpy array (noisy image)

        Returns:
            Denoised BGR numpy array
        """
        # Parameters:
        #   h=10         : filter strength (higher = more denoising, less detail)
        #   hColor=10    : color filter strength
        #   templateWindowSize=7  : patch size for comparison
        #   searchWindowSize=21   : search area size
        return cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=10,
            hColor=10,
            templateWindowSize=7,
            searchWindowSize=21
        )

    def deblur(self, image: np.ndarray,
               strength: float = 1.5) -> np.ndarray:
        """
        Sharpens blurry images using Unsharp Masking.

        Algorithm: Unsharp Masking
          1. Create a blurred version of the image (Gaussian blur)
          2. Compute "unsharp mask" = original - blurred  (the edges)
          3. Add scaled edges back: sharpened = original + strength × edges
          Result: edges are enhanced → image looks sharper

        Args:
            image   : BGR numpy array
            strength: Sharpening amount (1.0=subtle, 2.0=strong)

        Returns:
            Sharpened BGR numpy array
        """
        # Create Gaussian blurred version (sigma=1 for mild blur)
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)

        # Unsharp mask: original + strength × (original - blurred)
        # cv2.addWeighted(src1, alpha, src2, beta, gamma)
        # = alpha*src1 + beta*src2 + gamma
        sharpened = cv2.addWeighted(
            image,   1.0 + strength,   # amplify original
            blurred, -strength,         # subtract blurred (removes low freqs)
            0                           # no brightness offset
        )

        # Clip to valid range [0, 255]
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def remove_shadows(self, image: np.ndarray) -> np.ndarray:
        """
        Reduces uneven shadows and illumination gradients.

        Algorithm: Morphological normalization via HLS
          1. Convert to HLS (Hue, Lightness, Saturation)
          2. Apply morphological dilation to L channel → bright background
          3. Divide original L by background → normalized lightness
          4. Reconstruct and convert back

        Args:
            image: BGR numpy array

        Returns:
            Shadow-normalized BGR numpy array
        """
        # Convert BGR → HLS
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        h, l_channel, s = cv2.split(hls)

        # Dilate to create "background illumination estimate"
        # Kernel size 7×7 — larger = more illumination estimation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        background = cv2.dilate(l_channel, kernel)

        # Normalize: divide original by background
        # This removes the illumination gradient
        l_float = l_channel.astype(np.float32) + 1.0
        bg_float = background.astype(np.float32) + 1.0
        l_normalized = (l_float / bg_float * 128).clip(0, 255).astype(np.uint8)

        # Merge back
        hls_normalized = cv2.merge([h, l_normalized, s])
        return cv2.cvtColor(hls_normalized, cv2.COLOR_HLS2BGR)

    def dehaze(self, image: np.ndarray) -> np.ndarray:
        """
        Reduces haze and fog effect using contrast stretching.

        Algorithm: CLAHE-based dehazing
          Hazy images have compressed contrast (all values bunched together).
          CLAHE stretches the local contrast, revealing detail hidden by haze.

        Args:
            image: BGR numpy array (hazy/foggy image)

        Returns:
            Dehazed BGR numpy array
        """
        # Stronger CLAHE for haze removal
        clahe_strong = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        # Work in LAB to preserve color
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_dehazed = clahe_strong.apply(l)
        lab_dehazed = cv2.merge([l_dehazed, a, b])
        return cv2.cvtColor(lab_dehazed, cv2.COLOR_LAB2BGR)

    def resize_for_model(self, image: np.ndarray,
                         size: int = 640) -> np.ndarray:
        """
        Resize image to model input size using letterbox method.

        Why letterbox (not simple resize)?
          Simple resize distorts the aspect ratio — a car looks squashed.
          Letterbox resizes to fit within size×size, then pads with grey
          borders to fill the rest. YOLOv8 is trained this way.

        Example:
          Input:  1280×720 (16:9)
          Output: 640×640 with grey bars top and bottom
          Car proportions: preserved!

        Args:
            image: BGR numpy array (any size)
            size : Target size (default 640 for YOLOv8)

        Returns:
            size×size BGR numpy array
        """
        h, w = image.shape[:2]

        # Compute scale to fit within size×size
        scale = min(size / w, size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize to new dimensions
        resized = cv2.resize(image, (new_w, new_h),
                             interpolation=cv2.INTER_LINEAR)

        # Create grey canvas (114 is the YOLOv8 standard padding color)
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)

        # Center the resized image on the canvas
        x_offset = (size - new_w) // 2
        y_offset = (size - new_h) // 2
        canvas[y_offset:y_offset + new_h,
               x_offset:x_offset + new_w] = resized

        return canvas

    # ══════════════════════════════════════════════════════════
    # PRIVATE HELPER METHODS
    # ══════════════════════════════════════════════════════════

    def _mean_brightness(self, image: np.ndarray) -> float:
        """
        Returns the mean brightness of an image (0–255).
        Uses the L channel of LAB — more perceptually accurate than RGB mean.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        return float(lab[:, :, 0].mean())

    def _estimate_noise(self, image: np.ndarray) -> float:
        """
        Estimates noise level using the sigma estimation method.

        Algorithm:
          Apply a high-pass filter (difference from neighbour).
          In a noisy image, these differences are large.
          Returns standard deviation of the high-frequency component.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # High-pass kernel: detects pixel-to-pixel variation (noise)
        kernel = np.array([[1, -2, 1],
                           [-2, 4, -2],
                           [1, -2, 1]], dtype=np.float32)
        filtered = cv2.filter2D(gray, -1, kernel)
        sigma = float(np.std(filtered))
        return sigma

    def _has_shadows(self, image: np.ndarray) -> bool:
        """
        Detects uneven illumination / shadows.

        Converts to L channel, checks if there's large spatial
        variation in brightness (bright and dark regions in same image).
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l = lab[:, :, 0].astype(np.float32)

        # Divide image into quadrants, measure brightness per quadrant
        h, w = l.shape
        mid_h, mid_w = h // 2, w // 2
        quadrants = [
            l[:mid_h, :mid_w].mean(),    # top-left
            l[:mid_h, mid_w:].mean(),    # top-right
            l[mid_h:, :mid_w].mean(),    # bottom-left
            l[mid_h:, mid_w:].mean(),    # bottom-right
        ]

        # If brightness varies a lot between quadrants → shadows present
        return float(np.std(quadrants)) > self.SHADOW_STD_THRESHOLD

    def _is_hazy(self, image: np.ndarray) -> bool:
        """
        Detects haze/fog: image is bright but has low contrast.
        Hazy images have high mean brightness AND low standard deviation.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        mean_brightness = float(gray.mean())
        contrast = float(gray.std())
        return mean_brightness > self.HAZE_THRESHOLD and contrast < 45.0

    def get_image_quality_report(self, image_input: ImageInput) -> dict:
        """
        Analyse an image and return quality metrics WITHOUT modifying it.
        Useful for dashboards and diagnostics.
        """
        image = self.load_image(image_input)
        blur_score = self.detect_blur(image)
        noise_score = self._estimate_noise(image)
        brightness = self._mean_brightness(image)
        return {
            "shape":          list(image.shape),
            "brightness":     round(brightness, 2),
            "blur_score":     round(blur_score, 2),
            "noise_score":    round(noise_score, 2),
            "is_low_light":   brightness < self.LOW_LIGHT_THRESHOLD,
            "is_blurry":      blur_score < self.BLUR_THRESHOLD,
            "is_noisy":       noise_score > self.NOISE_THRESHOLD,
            "has_shadows":    self._has_shadows(image),
            "is_hazy":        self._is_hazy(image),
            "quality_grade":  self._quality_grade(brightness, blur_score),
        }

    def _quality_grade(self, brightness: float, blur_score: float) -> str:
        """Returns A/B/C/D quality grade based on brightness and blur."""
        score = 0
        if brightness >= self.LOW_LIGHT_THRESHOLD:
            score += 50
        if blur_score >= self.BLUR_THRESHOLD:
            score += 50
        if score >= 90:   return "A"
        if score >= 70:   return "B"
        if score >= 50:   return "C"
        return "D"
