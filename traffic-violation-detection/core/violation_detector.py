"""
core/violation_detector.py
==========================
Traffic Violation Detection — Phase 4 Implementation

Detects all 7 violation types:
  1. Helmet Non-Compliance   — head region skin-color heuristic
  2. Seatbelt Non-Compliance — diagonal edge analysis in driver region
  3. Triple Riding           — person count > 2 on one two-wheeler
  4. Red Light Violation     — traffic light state + vehicle position
  5. Stop Line Violation     — Hough line + vehicle bbox crossing
  6. Wrong-Side Driving      — lane-side geometric analysis
  7. Illegal Parking         — stationary vehicle in edge/restricted zone
"""

import cv2
import sys
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    ViolationType, VIOLATION_DISPLAY_NAMES, VIOLATION_FINES,
    HELMET_CONF_THRESHOLD, SEATBELT_CONF_THRESHOLD,
    VIOLATION_CONF_THRESHOLD, TRIPLE_RIDING_THRESHOLD,
    YOLO_HELMET_MODEL, YOLO_SEATBELT_MODEL,
    VEHICLE_CONF_THRESHOLD, PERSON_ON_VEHICLE_THRESHOLD,
)
from core.detector import VehicleDetector, Detection, DetectionResult
from core.preprocessor import ImagePreprocessor


# ══════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════

@dataclass
class Violation:
    """One detected traffic violation instance."""
    violation_type: str
    display_name:   str
    confidence:     float
    bbox:           List[int]
    vehicle_bbox:   Optional[List[int]] = None
    plate_number:   Optional[str]       = None
    description:    str                 = ""
    fine_amount:    int                 = 0

    def to_dict(self) -> dict:
        return {
            "violation_type": self.violation_type,
            "display_name":   self.display_name,
            "confidence":     round(self.confidence, 4),
            "bbox":           self.bbox,
            "vehicle_bbox":   self.vehicle_bbox,
            "plate_number":   self.plate_number,
            "description":    self.description,
            "fine_amount":    self.fine_amount,
        }


@dataclass
class ViolationResult:
    """All violations found in one image."""
    image_path: str
    violations: List[Violation]         = field(default_factory=list)
    detection_result: Optional[object]  = None   # DetectionResult ref
    processing_ms: float                = 0.0

    @property
    def total_fines(self) -> int:
        return sum(v.fine_amount for v in self.violations)

    @property
    def summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for v in self.violations:
            counts[v.violation_type] = counts.get(v.violation_type, 0) + 1
        return counts

    @property
    def has_violations(self) -> bool:
        return len(self.violations) > 0

    def to_dict(self) -> dict:
        return {
            "image_path":     self.image_path,
            "violations":     [v.to_dict() for v in self.violations],
            "total_fines":    self.total_fines,
            "summary":        self.summary,
            "has_violations": self.has_violations,
            "processing_ms":  round(self.processing_ms, 2),
        }


# ══════════════════════════════════════════════════════════════
# VIOLATION DETECTOR — MAIN CLASS
# ══════════════════════════════════════════════════════════════

class ViolationDetector:
    """
    Orchestrates all 7 traffic violation checks.

    Usage:
        vd = ViolationDetector()
        vd.load_models()
        result = vd.detect_all("traffic.jpg")
        for v in result.violations:
            print(v.display_name, v.confidence)
    """

    # ── Skin color HSV ranges for Indian skin tones ──────────
    # Range 1: Light to medium Indian skin
    _SKIN_LOWER1 = np.array([0,  15,  40],  dtype=np.uint8)
    _SKIN_UPPER1 = np.array([22, 200, 255], dtype=np.uint8)
    # Range 2: Medium to dark Indian skin (slightly higher hue)
    _SKIN_LOWER2 = np.array([0,  10,  20],  dtype=np.uint8)
    _SKIN_UPPER2 = np.array([18, 180, 210], dtype=np.uint8)
    # Range 3: Catches pinkish/reddish skin tones
    _SKIN_LOWER3 = np.array([160, 10, 40],  dtype=np.uint8)
    _SKIN_UPPER3 = np.array([180, 180, 255], dtype=np.uint8)

    # Helmet-absent threshold: if > this fraction of head is skin → no helmet
    # Lowered from 0.35 so bare heads are caught more aggressively
    _HELMET_SKIN_THRESHOLD = 0.20

    # Seatbelt: min diagonal edge pixels in driver region to count as belted
    _SEATBELT_EDGE_THRESHOLD = 40

    def __init__(self, vehicle_model: str = "yolov8s.pt"):
        self.vehicle_detector = VehicleDetector(vehicle_model)
        self.preprocessor     = ImagePreprocessor()
        self.helmet_model     = None    # loaded if file exists
        self.seatbelt_model   = None    # loaded if file exists
        self._loaded          = False

    # ── Model loading ─────────────────────────────────────────

    def load_models(self) -> None:
        """Load base vehicle detector + optional specialized models."""
        self.vehicle_detector.load_model()
        self._try_load_specialized(YOLO_HELMET_MODEL,  "helmet")
        self._try_load_specialized(YOLO_SEATBELT_MODEL, "seatbelt")
        self._loaded = True
        print("  ViolationDetector ready")

    def _try_load_specialized(self, model_path: Path, name: str) -> None:
        """Try loading a specialized model; silently skip if not found."""
        if not Path(model_path).exists():
            print(f"  INFO  No {name} model at {model_path} — using heuristic")
            return
        try:
            from ultralytics import YOLO
            model = YOLO(str(model_path))
            if name == "helmet":
                self.helmet_model = model
            else:
                self.seatbelt_model = model
            print(f"  OK  Loaded specialized {name} model")
        except Exception as e:
            print(f"  WARN  Could not load {name} model: {e}")

    # ── Main pipeline ─────────────────────────────────────────

    def detect_all(self, image_input) -> ViolationResult:
        """
        Run all 7 violation checks on one image.

        Pipeline:
          preprocess → vehicle detection → run each check → aggregate

        Args:
            image_input: File path (str/Path) or numpy BGR array

        Returns:
            ViolationResult containing every violation found
        """
        if not self._loaded:
            self.load_models()

        t0 = time.perf_counter()

        # 1. Load raw image
        image = self._load_image(image_input)
        path_str = str(image_input) if not isinstance(image_input, np.ndarray) \
                   else "array"

        # 2. Preprocess
        processed, _ = self.preprocessor.process(image)

        # 3. Vehicle + person detection (use config threshold, not hardcoded)
        det_result = self.vehicle_detector.detect(
            processed, conf_threshold=VEHICLE_CONF_THRESHOLD)

        # 4. Run all checks (collect violations)
        all_violations: List[Violation] = []
        all_violations.extend(self.check_helmet(processed, det_result))
        all_violations.extend(self.check_seatbelt(processed, det_result))
        all_violations.extend(self.check_triple_riding(det_result))
        all_violations.extend(self.check_red_light(processed, det_result))
        all_violations.extend(self.check_stop_line(processed, det_result))
        all_violations.extend(self.check_wrong_side(det_result))
        all_violations.extend(self.check_illegal_parking(processed, det_result))

        # 5. Global deduplication — remove same violation on same vehicle
        all_violations = self._deduplicate_violations(all_violations)

        ms = (time.perf_counter() - t0) * 1000
        return ViolationResult(
            image_path       = path_str,
            violations       = all_violations,
            detection_result = det_result,
            processing_ms    = ms,
        )

    def _load_image(self, image_input) -> np.ndarray:
        if isinstance(image_input, np.ndarray):
            return image_input
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Cannot decode: {path}")
        return img

    # ══════════════════════════════════════════════════════════
    # VIOLATION 1 — HELMET NON-COMPLIANCE
    # ══════════════════════════════════════════════════════════

    def check_helmet(self, image: np.ndarray,
                     det_result: DetectionResult) -> List[Violation]:
        """
        Detect riders on two-wheelers without helmets.

        ONE violation per vehicle — not per rider.
        If multiple riders on the same bike lack helmets, we raise
        one violation for that vehicle and note the count in the description.
        This prevents multiplied fines for the same incident.
        """
        violations = []
        seen_vehicles = set()

        for vehicle in det_result.vehicles:
            if not self.vehicle_detector.is_two_wheeler(vehicle):
                continue

            vid = tuple(vehicle.bbox)
            if vid in seen_vehicles:
                continue

            # Find all riders on this vehicle
            riders = self.vehicle_detector.get_persons_on_vehicle(
                vehicle, det_result.persons,
                containment_threshold=PERSON_ON_VEHICLE_THRESHOLD)

            if not riders:
                riders = self._proximity_match_riders(
                    vehicle, det_result.persons)

            # Check each rider — collect non-compliant ones
            non_compliant = []
            for rider in riders:
                has_helmet, conf = self._has_helmet(image, rider)
                if not has_helmet:
                    non_compliant.append((rider, conf))

            if non_compliant:
                seen_vehicles.add(vid)
                count     = len(non_compliant)
                # Use highest confidence among non-compliant riders
                best_conf = max(c for _, c in non_compliant)
                # Bbox covers all non-compliant riders merged with vehicle
                all_bboxes = [r.bbox for r, _ in non_compliant] + [vehicle.bbox]
                merged_bbox = self._merge_bboxes(all_bboxes)

                violations.append(self._make_violation(
                    vtype       = ViolationType.HELMET_VIOLATION,
                    confidence  = best_conf,
                    bbox        = merged_bbox,
                    vehicle_bbox= vehicle.bbox,
                    description = (
                        f"{count} rider(s) on {vehicle.class_name} "
                        f"without helmet"
                    ),
                ))
        return violations

    def _proximity_match_riders(self, vehicle, persons):
        """
        Match persons to vehicle by horizontal proximity.
        Used when containment overlap is too small (e.g. top-down camera).
        A person is a 'rider' if their horizontal center is within the
        vehicle's x-range (±30% tolerance).
        """
        vx1, vy1, vx2, vy2 = vehicle.bbox
        vw   = vx2 - vx1
        margin = int(vw * 0.30)
        riders = []
        for person in persons:
            px1, py1, px2, py2 = person.bbox
            pcx = (px1 + px2) // 2
            # Person center within extended vehicle x-range
            if (vx1 - margin) <= pcx <= (vx2 + margin):
                # Person vertically overlaps or is above vehicle (rider position)
                if py2 >= vy1 - int((vy2 - vy1) * 0.5):
                    riders.append(person)
        return riders

    def _has_helmet(self, image: np.ndarray,
                    person) -> Tuple[bool, float]:
        """
        Returns (has_helmet: bool, confidence: float).

        Uses three complementary methods:
          1. Custom YOLO model (most accurate — if downloaded)
          2. Multi-range HSV skin detection on head crop
          3. Head texture analysis (checks for smooth uniform helmet color)
        """
        # ── Method 1: Custom model ────────────────────────────
        if self.helmet_model is not None:
            return self._helmet_model_check(image, person)

        # ── Method 2: Multi-range skin detection ─────────────
        head = self._crop_head_region(image, person, fraction=0.32)
        if head is None or head.size == 0:
            # No head crop — flag conservatively (no helmet assumed)
            return False, 0.60

        if head.shape[0] < 8 or head.shape[1] < 8:
            # Head crop too small to analyze reliably
            return False, 0.55

        hsv = cv2.cvtColor(head, cv2.COLOR_BGR2HSV)

        # Apply 3 skin ranges (covers all Indian skin tones)
        mask1 = cv2.inRange(hsv, self._SKIN_LOWER1, self._SKIN_UPPER1)
        mask2 = cv2.inRange(hsv, self._SKIN_LOWER2, self._SKIN_UPPER2)
        mask3 = cv2.inRange(hsv, self._SKIN_LOWER3, self._SKIN_UPPER3)
        skin_mask = cv2.bitwise_or(mask1, cv2.bitwise_or(mask2, mask3))

        # Noise reduction: small isolated pixels are probably not skin
        kernel    = np.ones((3, 3), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

        total_px   = head.shape[0] * head.shape[1]
        skin_px    = int(np.sum(skin_mask > 0))
        skin_ratio = skin_px / max(1, total_px)

        if skin_ratio > self._HELMET_SKIN_THRESHOLD:
            conf = min(0.95, 0.58 + (skin_ratio - self._HELMET_SKIN_THRESHOLD) * 1.8)
            return False, round(conf, 3)

        # ── Method 3: Uniformity check ────────────────────────
        # Helmets are uniformly colored; bare heads have hair texture variation
        gray = cv2.cvtColor(head, cv2.COLOR_BGR2GRAY)
        # Coefficient of variation: high = textured (hair), low = uniform (helmet)
        mean_v = float(gray.mean())
        std_v  = float(gray.std())
        cv_val = std_v / max(1.0, mean_v)

        # Very low CV + not-black region = likely a smooth helmet
        if cv_val < 0.12 and mean_v > 30:
            return True, 0.0    # uniform colored region = helmet

        # Moderate skin ratio — uncertain, flag with lower confidence
        if skin_ratio > self._HELMET_SKIN_THRESHOLD * 0.7:
            return False, 0.55

        return True, 0.0   # default: helmet present

    def _helmet_model_check(self, image: np.ndarray,
                             person) -> Tuple[bool, float]:
        """Run custom helmet YOLO model on the full person crop."""
        x1, y1, x2, y2 = person.bbox
        h, w = image.shape[:2]
        crop = image[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
        if crop.size == 0:
            return False, 0.5
        results = self.helmet_model(crop, verbose=False,
                                    conf=HELMET_CONF_THRESHOLD)
        if not results or results[0].boxes is None:
            return False, 0.65   # no detection → assume no helmet
        for cls_id, conf in zip(results[0].boxes.cls.cpu().numpy(),
                                 results[0].boxes.conf.cpu().numpy()):
            label = results[0].names[int(cls_id)].lower()
            if "helmet" in label and "no" not in label:
                return True, 0.0
        return False, float(max(results[0].boxes.conf.cpu().numpy(), default=0.6))

    def _crop_head_region(self, image: np.ndarray,
                          person, fraction: float = 0.32) -> Optional[np.ndarray]:
        """
        Crop the head region from a person detection.
        Uses top `fraction` of the person bounding box height.
        Default 32% (slightly more than before at 28%).
        """
        x1, y1, x2, y2 = person.bbox
        h_img, w_img = image.shape[:2]
        head_h = max(10, int((y2 - y1) * fraction))
        x1c = max(0, x1);  y1c = max(0, y1)
        x2c = min(w_img, x2)
        y2c = min(h_img, y1 + head_h)
        if x2c <= x1c or y2c <= y1c:
            return None
        return image[y1c:y2c, x1c:x2c]

    # ══════════════════════════════════════════════════════════
    # VIOLATION 2 — SEATBELT NON-COMPLIANCE
    # ══════════════════════════════════════════════════════════

    def check_seatbelt(self, image: np.ndarray,
                       det_result: DetectionResult) -> List[Violation]:
        """
        Detect drivers/passengers in four-wheelers without seatbelts.

        Strategy:
          1. Try to find detected persons inside/near cars (lowered threshold)
          2. If no persons detected inside car (common — windshield blocks YOLO),
             directly analyse the driver seat region of the car
          3. One violation per vehicle maximum
        """
        violations  = []
        seen_veh    = set()

        for vehicle in det_result.vehicles:
            if not self.vehicle_detector.is_four_wheeler(vehicle):
                continue
            vid = tuple(vehicle.bbox)
            if vid in seen_veh:
                continue

            # ── Try person-based detection first ─────────────
            drivers = self.vehicle_detector.get_persons_on_vehicle(
                vehicle, det_result.persons,
                containment_threshold=0.12)   # very low — windshield clips bbox

            # Fallback: proximity match (person near car x-range)
            if not drivers:
                drivers = self._proximity_match_car_occupants(
                    vehicle, det_result.persons)

            if drivers:
                for driver in drivers:
                    has_belt, conf = self._has_seatbelt(image, driver)
                    if not has_belt:
                        seen_veh.add(vid)
                        violations.append(self._make_violation(
                            vtype       = ViolationType.SEATBELT_VIOLATION,
                            confidence  = conf,
                            bbox        = driver.bbox,
                            vehicle_bbox= vehicle.bbox,
                            description = (f"Driver/passenger in "
                                           f"{vehicle.class_name} without seatbelt"),
                        ))
                        break   # one violation per vehicle
            else:
                # ── No person detected — analyse car driver area directly ──
                has_belt, conf = self._check_driver_area(image, vehicle)
                if not has_belt:
                    seen_veh.add(vid)
                    violations.append(self._make_violation(
                        vtype       = ViolationType.SEATBELT_VIOLATION,
                        confidence  = conf,
                        bbox        = vehicle.bbox,
                        vehicle_bbox= vehicle.bbox,
                        description = (f"No seatbelt detected in "
                                       f"{vehicle.class_name} driver area"),
                    ))
        return violations

    def _proximity_match_car_occupants(self, vehicle, persons):
        """
        Match persons to a car by horizontal proximity.
        Persons who are within the car's x-range and at similar y-position
        are likely occupants visible through the window.
        """
        vx1, vy1, vx2, vy2 = vehicle.bbox
        occupants = []
        for person in persons:
            px1, py1, px2, py2 = person.bbox
            pcx = (px1 + px2) // 2
            pcy = (py1 + py2) // 2
            # Person centre inside car x-range and y-range
            if vx1 <= pcx <= vx2 and vy1 <= pcy <= vy2:
                occupants.append(person)
        return occupants

    def _check_driver_area(self, image: np.ndarray,
                            vehicle) -> Tuple[bool, float]:
        """
        Directly analyse the front-seat area of a car without needing
        a detected person. Crops the expected driver window region and
        looks for a seatbelt band.

        India is RHD (right-hand drive), so driver is on the RIGHT side
        of the car from a front-facing camera (left side from rear camera).
        We check BOTH sides to handle both camera orientations.
        """
        h_img, w_img = image.shape[:2]
        vx1, vy1, vx2, vy2 = vehicle.bbox
        vw = vx2 - vx1
        vh = vy2 - vy1

        # Driver window is upper half of vehicle, left or right third
        window_y1 = max(0, vy1 + int(vh * 0.05))
        window_y2 = min(h_img, vy1 + int(vh * 0.65))

        results = []
        for side_x1, side_x2 in [
            # Left third (driver in LHD or rear camera view)
            (max(0, vx1), min(w_img, vx1 + int(vw * 0.45))),
            # Right third (driver in RHD or front camera view)
            (max(0, vx2 - int(vw * 0.45)), min(w_img, vx2)),
        ]:
            if side_x2 <= side_x1 or window_y2 <= window_y1:
                continue
            region = image[window_y1:window_y2, side_x1:side_x2]
            if region.size == 0:
                continue
            # Create a fake Detection-like object for the heuristic
            class _FakePerson:
                bbox = [side_x1, window_y1, side_x2, window_y2]
            has_belt, conf = self._seatbelt_heuristic(image, _FakePerson())
            results.append((has_belt, conf))

        # If ANY side shows a seatbelt → consider compliant
        if any(hb for hb, _ in results):
            return True, 0.0
        # No seatbelt found on either side
        if results:
            avg_conf = float(np.mean([c for _, c in results]))
            return False, round(min(0.72, avg_conf), 3)
        return False, 0.55

    def _has_seatbelt(self, image: np.ndarray,
                      person) -> Tuple[bool, float]:
        """Returns (has_seatbelt, confidence)."""
        if self.seatbelt_model is not None:
            return self._seatbelt_model_check(image, person)
        return self._seatbelt_heuristic(image, person)

    def _seatbelt_heuristic(self, image: np.ndarray,
                             person) -> Tuple[bool, float]:
        """
        Multi-method seatbelt detection heuristic.

        Method 1: Diagonal Hough lines in torso (original approach improved)
        Method 2: Grey-band color detection (seatbelts are grey/dark)
        Method 3: Texture asymmetry (seatbelt creates left-right asymmetry)
        """
        x1, y1, x2, y2 = person.bbox
        h_img, w_img = image.shape[:2]
        person_h = max(1, y2 - y1)
        person_w = max(1, x2 - x1)

        # Try multiple torso crop ranges
        for frac_start, frac_end in [(0.20, 0.70), (0.15, 0.65), (0.25, 0.75)]:
            torso_y1 = max(0, y1 + int(person_h * frac_start))
            torso_y2 = min(h_img, y1 + int(person_h * frac_end))
            x1c = max(0, x1); x2c = min(w_img, x2)

            if torso_y2 <= torso_y1 or x2c <= x1c:
                continue
            torso = image[torso_y1:torso_y2, x1c:x2c]
            if torso.size == 0:
                continue

            th, tw = torso.shape[:2]

            # ── Method 1: Diagonal Hough lines ───────────────
            gray  = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
            blur  = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(blur, 20, 80)
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180,
                threshold    = max(15, tw // 4),
                minLineLength= max(8,  th // 4),
                maxLineGap   = 10)
            if lines is not None:
                for line in lines:
                    lx1, ly1, lx2, ly2 = line[0]
                    dx = abs(lx2 - lx1)
                    dy = abs(ly2 - ly1)
                    if dy == 0:
                        continue
                    angle = np.degrees(np.arctan2(dy, dx))
                    if 20 <= angle <= 70 and dx >= self._SEATBELT_EDGE_THRESHOLD:
                        return True, 0.0

            # ── Method 2: Grey band detection ────────────────
            # Seatbelts appear as a thin grey/dark diagonal band
            hsv  = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
            # Grey/dark band: low saturation, medium value
            grey_mask = cv2.inRange(hsv,
                                    np.array([0, 0, 30]),
                                    np.array([180, 50, 180]))
            grey_ratio = np.sum(grey_mask > 0) / max(1, th * tw)
            # A seatbelt covers ~5-15% of torso area
            if 0.04 < grey_ratio < 0.22:
                # Check if grey pixels form a diagonal band
                grey_cols = np.where(grey_mask.sum(axis=0) > th * 0.15)[0]
                if len(grey_cols) > tw * 0.15:
                    return True, 0.0

        return False, 0.60

    def _seatbelt_model_check(self, image: np.ndarray,
                               person: Detection) -> Tuple[bool, float]:
        """Run custom seatbelt model on driver crop."""
        x1, y1, x2, y2 = person.bbox
        h, w = image.shape[:2]
        crop = image[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
        if crop.size == 0:
            return False, 0.55
        results = self.seatbelt_model(crop, verbose=False,
                                      conf=SEATBELT_CONF_THRESHOLD)
        if not results or results[0].boxes is None:
            return False, 0.60
        for cls_id, conf in zip(results[0].boxes.cls.cpu().numpy(),
                                 results[0].boxes.conf.cpu().numpy()):
            label = results[0].names[int(cls_id)].lower()
            if "belt" in label and "no" not in label:
                return True, 0.0
        return False, 0.65

    # ══════════════════════════════════════════════════════════
    # VIOLATION 3 — TRIPLE RIDING
    # ══════════════════════════════════════════════════════════

    def check_triple_riding(self, det_result: DetectionResult) -> List[Violation]:
        """
        Flag two-wheelers with more than 2 persons on them.

        Algorithm:
          For each two-wheeler, count how many detected persons have
          a bounding box that overlaps significantly with the vehicle box.
          count > TRIPLE_RIDING_THRESHOLD (=2) → violation.
        """
        violations = []
        for vehicle in det_result.vehicles:
            if not self.vehicle_detector.is_two_wheeler(vehicle):
                continue
            riders = self.vehicle_detector.get_persons_on_vehicle(
                vehicle, det_result.persons, containment_threshold=PERSON_ON_VEHICLE_THRESHOLD)
            count = len(riders)

            # Fallback: proximity match if containment finds nothing
            if count == 0:
                riders = self._proximity_match_riders(vehicle, det_result.persons)
                count  = len(riders)
            if count > TRIPLE_RIDING_THRESHOLD:
                # Merge all rider bboxes into one encompassing box
                all_boxes = [r.bbox for r in riders] + [vehicle.bbox]
                merged = self._merge_bboxes(all_boxes)
                conf = min(0.98, 0.70 + (count - TRIPLE_RIDING_THRESHOLD) * 0.10)
                violations.append(self._make_violation(
                    vtype       = ViolationType.TRIPLE_RIDING,
                    confidence  = conf,
                    bbox        = merged,
                    vehicle_bbox= vehicle.bbox,
                    description = (f"{count} persons detected on "
                                   f"{vehicle.class_name} (max allowed: 2)"),
                ))
        return violations

    # ══════════════════════════════════════════════════════════
    # VIOLATION 4 — RED LIGHT VIOLATION
    # ══════════════════════════════════════════════════════════

    def check_red_light(self, image: np.ndarray,
                        det_result: DetectionResult) -> List[Violation]:
        """
        Detect vehicles that enter intersection while traffic light is red.

        ONE violation per vehicle — regardless of how many traffic lights
        are detected. The same car should not be fined 13 times because
        YOLOv8 detected 13 overlapping traffic light bounding boxes.
        """
        violations   = []
        flagged_vehs = set()   # track which vehicles are already fined
        img_h, img_w = image.shape[:2]

        for light in det_result.traffic_lights:
            state = self.vehicle_detector.get_traffic_light_state(image, light)
            if state != "red":
                continue

            light_cx, light_cy = light.center
            zone_x1 = max(0, light_cx - int(img_w * 0.40))
            zone_x2 = min(img_w, light_cx + int(img_w * 0.40))
            zone_y  = light_cy

            for vehicle in det_result.vehicles:
                vid = tuple(vehicle.bbox)
                if vid in flagged_vehs:
                    continue   # already fined for red light — skip

                vx1, vy1, vx2, vy2 = vehicle.bbox
                vcx = vehicle.center[0]
                if vy2 > zone_y and zone_x1 <= vcx <= zone_x2:
                    flagged_vehs.add(vid)
                    conf = min(0.95, light.confidence * 0.9)
                    violations.append(self._make_violation(
                        vtype       = ViolationType.RED_LIGHT,
                        confidence  = conf,
                        bbox        = vehicle.bbox,
                        vehicle_bbox= vehicle.bbox,
                        description = (f"{vehicle.class_name} entered "
                                       f"intersection during red light"),
                    ))
        return violations

    # ══════════════════════════════════════════════════════════
    # VIOLATION 5 — STOP LINE VIOLATION
    # ══════════════════════════════════════════════════════════

    def check_stop_line(self, image: np.ndarray,
                        det_result: DetectionResult) -> List[Violation]:
        """
        Detect vehicles that cross the stop line.

        Algorithm:
          1. Detect white horizontal lines in the lower 60% of the image
             using Canny edge detection + Hough transform
          2. Select the most prominent horizontal line in that zone
          3. Any vehicle whose bounding box bottom (vy2) > stop_line_y
             is considered to have crossed the line → violation
        """
        if not det_result.vehicles:
            return []

        stop_line_y = self._detect_stop_line(image)
        if stop_line_y is None:
            return []

        violations = []
        for vehicle in det_result.vehicles:
            _, _, _, vy2 = vehicle.bbox
            if vy2 > stop_line_y:
                conf = min(0.92, 0.65 + (vy2 - stop_line_y) /
                           max(1, image.shape[0]) * 1.5)
                violations.append(self._make_violation(
                    vtype       = ViolationType.STOP_LINE,
                    confidence  = conf,
                    bbox        = vehicle.bbox,
                    vehicle_bbox= vehicle.bbox,
                    description = (f"{vehicle.class_name} crossed stop line "
                                   f"(line at y={stop_line_y}, "
                                   f"vehicle bottom at y={vy2})"),
                ))
        return violations

    def _detect_stop_line(self, image: np.ndarray) -> Optional[int]:
        """
        Returns the Y-coordinate of the detected stop line, or None.

        Uses Hough Probabilistic Transform on a Canny edge map.
        Only searches the lower 60% of the image (where stop lines appear).
        Prioritises nearly-horizontal, wide white lines.
        """
        h, w = image.shape[:2]
        roi_y = int(h * 0.40)   # search from 40% down only
        roi   = image[roi_y:, :]

        gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Enhance white lines: threshold bright pixels
        _, white_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(white_mask, 50, 150)

        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold  = max(30, w // 8),
            minLineLength = max(40, w // 5),
            maxLineGap = 20
        )
        if lines is None:
            return None

        h_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(abs(y2 - y1),
                                               abs(x2 - x1) + 1e-6)))
            if angle < 12:            # nearly horizontal
                h_lines.append(y1 + roi_y)

        if not h_lines:
            return None
        return int(np.median(h_lines))

    # ══════════════════════════════════════════════════════════
    # VIOLATION 6 — WRONG-SIDE DRIVING
    # ══════════════════════════════════════════════════════════

    def check_wrong_side(self, det_result: DetectionResult) -> List[Violation]:
        """
        Detect vehicles driving on the wrong side of the road.

        Simplified algorithm for static images (India — left-hand traffic):
          1. Estimate lane centre from detected vehicle positions
          2. In left-hand traffic the majority of vehicles should be on the
             LEFT half of the frame (from driver's perspective)
          3. A lone vehicle significantly to the RIGHT of all other vehicles
             with no other vehicles near it on the right → flag as wrong side

        Note: Accurate wrong-side detection from a single static image is
        hard. This is a heuristic that works well for camera-facing traffic.
        """
        violations = []
        vehicles = det_result.vehicles
        if len(vehicles) < 2:
            return violations   # need context (at least 2 vehicles)

        img_w = 640   # default; overridden if detection_result has shape
        if det_result.image_shape:
            img_w = det_result.image_shape[1]

        # Compute vehicle center-X positions
        centers_x = [v.center[0] for v in vehicles]
        median_cx  = float(np.median(centers_x))

        for vehicle in vehicles:
            vcx = vehicle.center[0]
            # Vehicle is on wrong side if it is >35% image width away from
            # median position AND on the right side of image centre
            deviation = vcx - median_cx
            if deviation > img_w * 0.35 and vcx > img_w * 0.60:
                conf = min(0.88, 0.55 + deviation / img_w)
                violations.append(self._make_violation(
                    vtype       = ViolationType.WRONG_SIDE,
                    confidence  = conf,
                    bbox        = vehicle.bbox,
                    vehicle_bbox= vehicle.bbox,
                    description = (f"{vehicle.class_name} appears to be on "
                                   f"wrong side of road"),
                ))
        return violations

    # ══════════════════════════════════════════════════════════
    # VIOLATION 7 — ILLEGAL PARKING
    # ══════════════════════════════════════════════════════════

    def check_illegal_parking(self, image: np.ndarray,
                               det_result: DetectionResult) -> List[Violation]:
        """
        Detect vehicles illegally parked on kerbs, pavements, or no-parking zones.

        Simplified heuristic for static images:
          1. Vehicles that are stationary are assumed to be VERY close to the
             bottom edge of the frame (near the camera / roadside)
          2. Their bounding box is also WIDE relative to their height
             (parked vehicles appear wide and low in the frame)
          3. They have no persons overlapping them (unoccupied)

        Real deployment would use zone polygons from map data.
        This heuristic flags common roadside parking patterns.
        """
        violations = []
        if not det_result.vehicles:
            return violations

        h, w = image.shape[:2]

        for vehicle in det_result.vehicles:
            vx1, vy1, vx2, vy2 = vehicle.bbox
            vw = vx2 - vx1
            vh = vy2 - vy1

            if vh == 0:
                continue

            # Heuristic conditions for illegal parking:
            # a) Vehicle occupies bottom 25% of image (parked at roadside)
            near_bottom = vy2 > h * 0.75

            # b) Vehicle is wide relative to height (parked aspect ratio)
            wide_aspect = (vw / vh) > 2.0

            # c) No person inside it (unoccupied = parked)
            riders = self.vehicle_detector.get_persons_on_vehicle(
                vehicle, det_result.persons, containment_threshold=0.20)
            unoccupied = len(riders) == 0

            # d) Vehicle is near the image edge (parked on side)
            near_edge = vx1 < w * 0.10 or vx2 > w * 0.90

            if near_bottom and wide_aspect and unoccupied and near_edge:
                conf = 0.62
                violations.append(self._make_violation(
                    vtype       = ViolationType.ILLEGAL_PARKING,
                    confidence  = conf,
                    bbox        = vehicle.bbox,
                    vehicle_bbox= vehicle.bbox,
                    description = (f"{vehicle.class_name} appears parked in "
                                   f"restricted zone (near edge, unoccupied)"),
                ))
        return violations

    # ══════════════════════════════════════════════════════════
    # SHARED HELPERS
    # ══════════════════════════════════════════════════════════

    def _make_violation(self, vtype: str, confidence: float,
                        bbox: List[int],
                        vehicle_bbox: Optional[List[int]] = None,
                        description: str = "") -> Violation:
        """Factory method — creates a Violation with correct display name & fine."""
        return Violation(
            violation_type = vtype,
            display_name   = VIOLATION_DISPLAY_NAMES.get(vtype, vtype),
            confidence     = round(confidence, 4),
            bbox           = bbox,
            vehicle_bbox   = vehicle_bbox,
            description    = description,
            fine_amount    = VIOLATION_FINES.get(vtype, 0),
        )

    def _merge_bboxes(self, bboxes: List[List[int]]) -> List[int]:
        """Return the smallest bounding box that contains all input boxes."""
        xs1 = [b[0] for b in bboxes];  ys1 = [b[1] for b in bboxes]
        xs2 = [b[2] for b in bboxes];  ys2 = [b[3] for b in bboxes]
        return [min(xs1), min(ys1), max(xs2), max(ys2)]

    def _deduplicate_violations(self,
                                 violations: List[Violation]) -> List[Violation]:
        """
        Remove duplicate violations of the same type on the same vehicle.

        Two violations are considered duplicates when:
          - Same violation_type
          - Their bounding boxes overlap by IoU >= 0.40

        Keeps the one with the highest confidence.
        """
        if not violations:
            return violations

        kept: List[Violation] = []

        for v in violations:
            is_dup = False
            for k in kept:
                if k.violation_type != v.violation_type:
                    continue
                iou = self._bbox_iou(v.bbox, k.bbox)
                if iou >= 0.40:
                    # Same violation on same vehicle — keep higher confidence
                    if v.confidence > k.confidence:
                        kept.remove(k)
                        kept.append(v)
                    is_dup = True
                    break
            if not is_dup:
                kept.append(v)

        return kept

    @staticmethod
    def _bbox_iou(b1: List[int], b2: List[int]) -> float:
        """Compute IoU between two [x1,y1,x2,y2] boxes."""
        ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        if inter == 0:
            return 0.0
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        return inter / max(1, a1 + a2 - inter)
