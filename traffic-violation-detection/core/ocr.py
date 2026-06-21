"""
core/ocr.py
===========
License Plate Detection and OCR — Phase 5 Implementation

Pipeline:
  1. detect_plate_regions()  — find plate bounding boxes (YOLO or contours)
  2. crop_plate()            — extract + preprocess the plate image
  3. run_ocr()               — EasyOCR text extraction
  4. clean_plate_text()      — normalize whitespace / special chars
  5. validate_plate()        — check Indian plate format (XX 00 XX 0000)
  6. extract()               — full pipeline, returns List[PlateResult]

Indian plate format:
  XX  00  XX   0000
  MH  12  AB   1234
  State | District | Series | Number
"""

import re
import sys
import cv2
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import PLATE_CONF_THRESHOLD, YOLO_PLATE_MODEL

# ── Indian plate regex ────────────────────────────────────────
# Matches: MH12AB1234  or  MH 12 AB 1234  or  DL-01-AB-1234
PLATE_PATTERN = re.compile(
    r'^[A-Z]{2}[\s\-]?\d{2}[\s\-]?[A-Z]{1,3}[\s\-]?\d{4}$',
    re.IGNORECASE
)

# Common OCR mis-reads and their corrections
OCR_CORRECTIONS = {
    'O': '0',  'I': '1',  'Z': '2',  'S': '5',
    'G': '6',  'B': '8',  ' ': '',   '-': '',
    '.': '',   ',': '',   ':': '',
}


# ══════════════════════════════════════════════════════════════
# RESULT DATACLASS
# ══════════════════════════════════════════════════════════════

@dataclass
class PlateResult:
    """
    Detected + recognised license plate.

    Attributes:
        plate_text        : Raw OCR output (may have spaces/noise)
        plate_text_clean  : Cleaned, no-space uppercase (e.g. MH12AB1234)
        confidence        : OCR confidence 0.0–1.0
        bbox              : [x1, y1, x2, y2] in original image coordinates
        is_valid_format   : True if matches Indian plate regex
        detection_method  : 'yolo' | 'contour' | 'full_image'
        plate_image       : Cropped plate numpy array (BGR)
    """
    plate_text:       str
    plate_text_clean: str
    confidence:       float
    bbox:             List[int]
    is_valid_format:  bool
    detection_method: str               = "contour"
    plate_image:      Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {
            "plate_text":       self.plate_text,
            "plate_text_clean": self.plate_text_clean,
            "confidence":       round(self.confidence, 4),
            "bbox":             self.bbox,
            "is_valid_format":  self.is_valid_format,
            "detection_method": self.detection_method,
        }


# ══════════════════════════════════════════════════════════════
# MAIN OCR CLASS
# ══════════════════════════════════════════════════════════════

class LicensePlateOCR:
    """
    Detects license plates and extracts text using EasyOCR.

    Usage:
        ocr = LicensePlateOCR()
        ocr.load_models()
        plates = ocr.extract("image.jpg")
        for p in plates:
            print(p.plate_text_clean, p.confidence)
    """

    # Plate aspect ratio range (width / height)
    # Relaxed from 1.8–7.0 to catch angled/partial plates
    MIN_ASPECT = 1.2
    MAX_ASPECT = 8.0

    # Minimum plate area as fraction of image area
    # Lowered from 0.001 to catch small distant plates
    MIN_AREA_RATIO = 0.0003
    MAX_AREA_RATIO = 0.20

    def __init__(self, use_gpu: bool = False):
        """
        Args:
            use_gpu: Pass True if CUDA GPU is available for faster OCR.
                     On CPU (default for demo), EasyOCR still works fine.
        """
        self.use_gpu      = use_gpu
        self.ocr_reader   = None   # EasyOCR Reader
        self.plate_model  = None   # YOLOv8 plate detection model
        self._loaded      = False

    # ── Model loading ─────────────────────────────────────────

    def load_models(self) -> None:
        """
        Load EasyOCR + optional YOLOv8 plate detection model.

        First call downloads EasyOCR English models (~100 MB) to
        ~/.EasyOCR/  — this takes 1–3 minutes on first run only.
        Subsequent calls use the local cache (instant).
        """
        self._load_easyocr()
        self._load_plate_detector()
        self._loaded = True

    def _load_easyocr(self) -> None:
        print("  Loading EasyOCR (downloads ~100 MB on first run) ...")
        try:
            import easyocr
            self.ocr_reader = easyocr.Reader(
                ['en'],
                gpu=self.use_gpu,
                verbose=False
            )
            print("  EasyOCR ready")
        except ImportError:
            raise RuntimeError(
                "easyocr not installed. Run: pip install easyocr==1.7.2")

    def _load_plate_detector(self) -> None:
        """Load YOLOv8 plate model if the weight file exists."""
        if not Path(YOLO_PLATE_MODEL).exists():
            print(f"  INFO  No plate model at {YOLO_PLATE_MODEL}"
                  " — using contour detection fallback")
            return
        try:
            from ultralytics import YOLO
            self.plate_model = YOLO(str(YOLO_PLATE_MODEL))
            print(f"  OK  Loaded plate detection model")
        except Exception as e:
            print(f"  WARN  Could not load plate model: {e}")

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load_models()

    # ── Main entry point ──────────────────────────────────────

    def extract(self, image_input,
                vehicle_bboxes: Optional[List[List[int]]] = None
                ) -> List[PlateResult]:
        """
        Detect plates and extract text from an image.

        Runs ALL detection strategies in parallel and merges results.
        Strategies tried (in order of reliability):
          1. YOLOv8 plate model        (if downloaded)
          2. Full-image OCR scan       (finds any plate-like text anywhere)
          3. Vehicle-region contour    (search near vehicle bboxes)
          4. Full-image contour        (last resort)
        """
        self._ensure_loaded()
        image = self._load_image(image_input)

        all_results: List[PlateResult] = []
        seen_texts = set()

        def add_results(new_results):
            for r in new_results:
                key = r.plate_text_clean[:6] if len(r.plate_text_clean) >= 6 \
                      else r.plate_text_clean
                if key and key not in seen_texts:
                    seen_texts.add(key)
                    all_results.append(r)

        # ── Strategy 1: YOLO model ────────────────────────────
        if self.plate_model is not None:
            regions = self.detect_plate_regions_yolo(image)
            if regions:
                add_results(self._regions_to_results(image, regions, "yolo"))

        # ── Strategy 2: Full-image OCR scan ──────────────────
        # This often works better on real photos than contour detection
        full_img_results = self._ocr_full_image(image)
        add_results(full_img_results)

        # ── Strategy 3: Vehicle region contour search ─────────
        if vehicle_bboxes:
            regions = self._search_in_vehicles(image, vehicle_bboxes)
            if regions:
                add_results(self._regions_to_results(image, regions, "contour"))

        # ── Strategy 4: Full-image contour detection ──────────
        regions = self.detect_plate_regions_contour(image)
        if regions:
            add_results(self._regions_to_results(image, regions, "contour"))

        # Sort: valid format first, then by confidence
        all_results.sort(
            key=lambda r: (r.is_valid_format, r.confidence),
            reverse=True)
        return all_results

    def _regions_to_results(self, image: np.ndarray,
                             regions: List[List[int]],
                             method: str) -> List[PlateResult]:
        """Convert a list of bboxes → PlateResult list."""
        results = []
        for bbox in regions:
            plate_img = self.crop_plate(image, bbox)
            if plate_img is None or plate_img.size == 0:
                continue
            text, conf = self.run_ocr(plate_img)
            if not text:
                continue
            clean = self.clean_plate_text(text)
            if len(clean) < 4:   # too short (was 5, now 4)
                continue
            results.append(PlateResult(
                plate_text       = text,
                plate_text_clean = clean,
                confidence       = conf,
                bbox             = bbox,
                is_valid_format  = self.validate_plate(clean),
                detection_method = method,
                plate_image      = plate_img,
            ))
        return results

    # ── Plate detection ───────────────────────────────────────

    def detect_plate_regions(self, image: np.ndarray) -> List[List[int]]:
        """
        Auto-select detection method and return plate bounding boxes.
        Prefers YOLO if model available, else contour detection.
        """
        if self.plate_model is not None:
            return self.detect_plate_regions_yolo(image)
        return self.detect_plate_regions_contour(image)

    def detect_plate_regions_yolo(self, image: np.ndarray) -> List[List[int]]:
        """Run YOLOv8 plate detector. Returns list of [x1,y1,x2,y2]."""
        if self.plate_model is None:
            return []
        results = self.plate_model(image, conf=PLATE_CONF_THRESHOLD,
                                   verbose=False)
        boxes = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes.xyxy.cpu().numpy():
                boxes.append([int(b) for b in box[:4]])
        return boxes

    def detect_plate_regions_contour(self,
                                      image: np.ndarray) -> List[List[int]]:
        """
        Detect license plate regions using contour analysis.
        Runs multiple parameter sets and merges results for better coverage.
        """
        h, w = image.shape[:2]
        img_area = h * w
        all_candidates = []

        # Run detection with multiple strategies
        for strategy in self._contour_strategies(image):
            candidates = self._find_plate_contours(strategy, img_area, w)
            all_candidates.extend(candidates)

        return self._deduplicate_boxes(all_candidates)

    def _contour_strategies(self, image: np.ndarray):
        """Return multiple preprocessed images to search for plates."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        strategies = []

        # Strategy 1: Sobel edges
        blur1   = cv2.GaussianBlur(gray, (5, 5), 0)
        sobel_x = cv2.Sobel(blur1, cv2.CV_64F, 1, 0, ksize=3)
        s1 = cv2.convertScaleAbs(sobel_x)
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        strategies.append(cv2.morphologyEx(s1, cv2.MORPH_CLOSE, kernel1))

        # Strategy 2: Adaptive threshold
        blur2 = cv2.GaussianBlur(gray, (3, 3), 0)
        s2    = cv2.adaptiveThreshold(blur2, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 19, 9)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 6))
        strategies.append(cv2.morphologyEx(s2, cv2.MORPH_CLOSE, kernel2))

        # Strategy 3: Otsu threshold (works well on plates with strong contrast)
        _, s3 = cv2.threshold(blur1, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
        strategies.append(cv2.morphologyEx(s3, cv2.MORPH_CLOSE, kernel3))

        # Strategy 4: CLAHE + Canny (good for low-contrast plates)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        edges = cv2.Canny(enhanced, 50, 150)
        kernel4 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        strategies.append(cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel4))

        return strategies

    def _find_plate_contours(self, binary_img: np.ndarray,
                              img_area: int, img_w: int) -> List[List[int]]:
        """Find bounding boxes of plate-like contours in a binary image."""
        contours, _ = cv2.findContours(
            binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in contours:
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            if rh == 0:
                continue
            aspect     = rw / rh
            area       = rw * rh
            area_ratio = area / max(1, img_area)
            if (self.MIN_ASPECT <= aspect <= self.MAX_ASPECT and
                    self.MIN_AREA_RATIO <= area_ratio <= self.MAX_AREA_RATIO):
                candidates.append([rx, ry, rx + rw, ry + rh])
        return candidates

    def _search_in_vehicles(self, image: np.ndarray,
                             vehicle_bboxes: List[List[int]]) -> List[List[int]]:
        """
        Search for plates within the lower portion of each vehicle bbox.
        Plates are typically in the bottom 25% of a vehicle's bounding box.
        """
        h_img, w_img = image.shape[:2]
        all_plates = []

        for vx1, vy1, vx2, vy2 in vehicle_bboxes:
            # Lower 50% of vehicle (plates can be front OR rear)
            vh = vy2 - vy1
            roi_y1 = max(0, vy2 - int(vh * 0.55))
            roi_y2 = min(h_img, vy2 + 20)
            roi_x1 = max(0, vx1 - 10)
            roi_x2 = min(w_img, vx2 + 10)

            roi = image[roi_y1:roi_y2, roi_x1:roi_x2]
            if roi.size == 0:
                continue

            # Run contour detection on ROI
            local_plates = self.detect_plate_regions_contour(roi)
            # Translate back to full image coordinates
            for lx1, ly1, lx2, ly2 in local_plates:
                all_plates.append([
                    roi_x1 + lx1, roi_y1 + ly1,
                    roi_x1 + lx2, roi_y1 + ly2
                ])
        return all_plates

    # ── Plate image preprocessing ─────────────────────────────

    def crop_plate(self, image: np.ndarray, bbox: List[int],
                   padding: int = 4) -> Optional[np.ndarray]:
        """
        Crop and preprocess the plate region for OCR.

        Preprocessing:
          1. Add padding around the detected region
          2. Resize to a standard height (64px) while keeping aspect ratio
          3. Convert to grayscale
          4. Apply adaptive threshold (handles variable lighting)
          5. Stack grayscale + binary as 3-channel for EasyOCR
        """
        h_img, w_img = image.shape[:2]
        x1, y1, x2, y2 = bbox

        # Reject degenerate or inverted bboxes BEFORE expanding with padding
        if x2 <= x1 or y2 <= y1:
            return None

        # Add padding, clamp to image bounds
        x1p = max(0, x1 - padding)
        y1p = max(0, y1 - padding)
        x2p = min(w_img, x2 + padding)
        y2p = min(h_img, y2 + padding)

        crop = image[y1p:y2p, x1p:x2p]
        if crop.size == 0:
            return None

        # Resize to standard height, keep aspect ratio
        target_h = 64
        scale    = target_h / crop.shape[0]
        target_w = max(10, int(crop.shape[1] * scale))
        resized  = cv2.resize(crop, (target_w, target_h),
                               interpolation=cv2.INTER_CUBIC)

        # Grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Adaptive threshold for clean black-on-white text
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2)

        # Return as BGR (EasyOCR works on color images)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # ── OCR ───────────────────────────────────────────────────

    def run_ocr(self, plate_image: np.ndarray) -> Tuple[str, float]:
        """
        Run EasyOCR on a preprocessed plate image.
        Tries 5 preprocessing variants and returns the best result.
        """
        if self.ocr_reader is None:
            return "", 0.0

        best_text = ""
        best_conf = 0.0

        variants = self._make_ocr_variants(plate_image)

        for variant in variants:
            try:
                results = self.ocr_reader.readtext(
                    variant,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ',
                    detail=1,
                    paragraph=False,
                    min_size=8,
                    text_threshold=0.5,
                    low_text=0.3,
                )
            except Exception:
                continue
            if not results:
                continue
            texts    = [r[1] for r in results]
            confs    = [float(r[2]) for r in results]
            joined   = " ".join(texts).strip()
            avg_conf = float(np.mean(confs)) if confs else 0.0
            if avg_conf > best_conf and len(joined) >= 4:
                best_text = joined
                best_conf = avg_conf

        return best_text, best_conf

    def _make_ocr_variants(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Generate multiple preprocessing variants of a plate image.
        EasyOCR often reads one variant when others fail.
        """
        variants = [image]   # original always included

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Variant 2: Adaptive threshold (black text on white)
        binary = cv2.adaptiveThreshold(gray, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
        variants.append(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))

        # Variant 3: Otsu threshold
        _, otsu = cv2.threshold(gray, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))

        # Variant 4: Inverted Otsu (white text on black)
        variants.append(cv2.cvtColor(255 - otsu, cv2.COLOR_GRAY2BGR))

        # Variant 5: CLAHE enhanced
        clahe     = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        enhanced  = clahe.apply(gray)
        variants.append(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR))

        # Variant 6: Sharpened
        kernel    = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        variants.append(sharpened)

        return variants

    # ── Text cleaning ─────────────────────────────────────────

    def clean_plate_text(self, raw_text: str) -> str:
        """
        Normalize OCR output to standard Indian plate format.

        Steps:
          1. Uppercase
          2. Remove spaces, dashes, punctuation
          3. Apply common OCR mis-read corrections (O→0, I→1, etc.)
             but only in the numeric sections of the plate
        """
        if not raw_text:
            return ""

        text = raw_text.upper().strip()
        # Remove common noise characters
        text = re.sub(r'[^A-Z0-9]', '', text)

        # Apply character corrections only in known numeric positions
        # Indian plate format: [2 letters][2 digits][1-3 letters][4 digits]
        # Positions 2–3 (index) should be digits → fix letter lookalikes
        corrected = list(text)
        for i, ch in enumerate(corrected):
            if i >= 2 and i <= 3:    # district code = digits
                if ch in OCR_CORRECTIONS:
                    corrected[i] = OCR_CORRECTIONS[ch]
            if i >= 7:               # serial number = digits
                if ch in ('O', 'I', 'Z', 'S', 'G', 'B'):
                    corrected[i] = OCR_CORRECTIONS.get(ch, ch)

        return "".join(corrected)

    # ── Validation ────────────────────────────────────────────

    def validate_plate(self, text: str) -> bool:
        """
        Check whether cleaned text matches Indian vehicle registration format.

        Format: XX00XX0000 (no spaces, uppercase)
        e.g. MH12AB1234, DL01CD5678, KA03EF9012

        Returns True if valid, False otherwise.
        """
        cleaned = re.sub(r'[\s\-]', '', text).upper()
        return bool(PLATE_PATTERN.match(cleaned))

    # ── Helpers ───────────────────────────────────────────────

    def _ocr_full_image(self, image: np.ndarray) -> List[PlateResult]:
        """
        Run OCR on the full image and pick out plate-like text strings.

        This is surprisingly effective on real Indian traffic photos
        where the plate is clearly visible and readable.

        Tries both original and enhanced versions of the image.
        Accepts partial plate matches (not just perfect format).
        """
        if self.ocr_reader is None:
            return []

        plates = []
        seen   = set()

        # Try multiple image variants for full-image scan
        gray     = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        for scan_img in [image, enhanced_bgr]:
            try:
                results = self.ocr_reader.readtext(
                    scan_img,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ',
                    detail=1,
                    paragraph=False,
                    min_size=10,
                    text_threshold=0.4,
                    low_text=0.3,
                    canvas_size=1280,   # cap detector canvas — bounds CPU memory
                )
            except Exception:
                continue

            # Look for plate-like text in all detected strings
            for (bbox_pts, text, conf) in results:
                if conf < 0.25:
                    continue
                clean = self.clean_plate_text(text)
                if len(clean) < 4:
                    continue

                # Check for valid or partial plate pattern
                is_valid = self.validate_plate(clean)
                is_partial = self._is_partial_plate(clean)

                if not (is_valid or is_partial):
                    continue

                # Deduplicate
                key = clean[:6] if len(clean) >= 6 else clean
                if key in seen:
                    continue
                seen.add(key)

                xs   = [int(p[0]) for p in bbox_pts]
                ys   = [int(p[1]) for p in bbox_pts]
                bbox = [min(xs), min(ys), max(xs), max(ys)]

                plates.append(PlateResult(
                    plate_text       = text,
                    plate_text_clean = clean,
                    confidence       = float(conf),
                    bbox             = bbox,
                    is_valid_format  = is_valid,
                    detection_method = "full_image",
                ))

        return plates

    def _is_partial_plate(self, text: str) -> bool:
        """
        Check for a partial plate match.
        Indian plates start with 2 letters + 2 digits.
        Even partial reads are useful.
        """
        cleaned = re.sub(r'[\s\-]', '', text).upper()
        # Starts with 2 letters + 2 digits = likely a plate
        partial = re.compile(r'^[A-Z]{2}\d{2}', re.IGNORECASE)
        return bool(partial.match(cleaned)) and len(cleaned) >= 4

    def _deduplicate_boxes(self, boxes: List[List[int]],
                            iou_thresh: float = 0.4) -> List[List[int]]:
        """Remove overlapping candidate boxes — keep the largest by area."""
        if not boxes:
            return []
        # Sort by area descending
        boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]),
                        reverse=True)
        kept = []
        for box in boxes:
            overlap = False
            for kept_box in kept:
                if self._iou(box, kept_box) > iou_thresh:
                    overlap = True
                    break
            if not overlap:
                kept.append(box)
        return kept

    @staticmethod
    def _iou(b1: List[int], b2: List[int]) -> float:
        ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        if inter == 0:
            return 0.0
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        return inter / max(1, a1 + a2 - inter)

    @staticmethod
    def _load_image(image_input) -> np.ndarray:
        if isinstance(image_input, np.ndarray):
            return image_input
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Cannot decode image: {path}")
        return img
