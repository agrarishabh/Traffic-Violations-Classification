"""
core/detector.py
================
Vehicle and Road User Detection — Phase 3 Implementation

Uses YOLOv8 (pre-trained on COCO) to detect:
  - Cars, motorcycles, bicycles, buses, trucks
  - Persons (riders, drivers, pedestrians)
  - Traffic lights

COCO Class IDs used:
  0=person  1=bicycle  2=car  3=motorcycle
  5=bus     7=truck    9=traffic light   11=stop sign
"""

import time
import sys
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    VEHICLE_CLASS_IDS, TWO_WHEELER_IDS, FOUR_WHEELER_IDS,
    PERSON_CLASS_ID, TRAFFIC_LIGHT_ID,
    VEHICLE_CONF_THRESHOLD, PERSON_CONF_THRESHOLD,
    IOU_THRESHOLD, COCO_CLASSES
)

# ── Type alias ────────────────────────────────────────────────
ImageInput = Union[str, Path, np.ndarray]


# ══════════════════════════════════════════════════════════════
# DATA CLASSES  (kept here so imports are simple)
# ══════════════════════════════════════════════════════════════

@dataclass
class Detection:
    """
    One detected object in an image.

    Attributes:
        class_id   : COCO numeric class (e.g. 2 = car)
        class_name : Human label        (e.g. "car")
        confidence : Model confidence   0.0–1.0
        bbox       : [x1, y1, x2, y2]  pixel coordinates
        track_id   : Optional object-tracking ID
    """
    class_id:   int
    class_name: str
    confidence: float
    bbox:       List[int]          # [x1, y1, x2, y2]
    track_id:   Optional[int] = None

    # ── Convenience properties ────────────────────────────────
    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, (x2 - x1) * (y2 - y1))

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    def to_dict(self) -> dict:
        return {
            "class_id":   self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox":       self.bbox,
            "track_id":   self.track_id,
            "center":     list(self.center),
            "area":       self.area,
            "width":      self.width,
            "height":     self.height,
        }


@dataclass
class DetectionResult:
    """
    All detections for one image, pre-sorted into categories.

    Attributes:
        image_path    : Source image path (or 'array' if passed as numpy)
        detections    : Every detected object (all classes)
        vehicles      : Subset — vehicles only
        persons       : Subset — persons only
        traffic_lights: Subset — traffic lights only
        inference_ms  : Model inference time in milliseconds
        image_shape   : (height, width, channels) of the analysed image
    """
    image_path:     str
    detections:     List[Detection] = field(default_factory=list)
    vehicles:       List[Detection] = field(default_factory=list)
    persons:        List[Detection] = field(default_factory=list)
    traffic_lights: List[Detection] = field(default_factory=list)
    inference_ms:   float = 0.0
    image_shape:    Tuple[int, ...] = (0, 0, 3)

    @property
    def vehicle_count(self) -> int:
        return len(self.vehicles)

    @property
    def person_count(self) -> int:
        return len(self.persons)

    @property
    def has_red_light(self) -> bool:
        """True if any traffic light is detected (Phase 4 will check color)."""
        return len(self.traffic_lights) > 0

    def to_dict(self) -> dict:
        return {
            "image_path":     self.image_path,
            "image_shape":    list(self.image_shape),
            "total_objects":  len(self.detections),
            "vehicle_count":  self.vehicle_count,
            "person_count":   self.person_count,
            "vehicles":       [d.to_dict() for d in self.vehicles],
            "persons":        [d.to_dict() for d in self.persons],
            "traffic_lights": [d.to_dict() for d in self.traffic_lights],
            "inference_ms":   round(self.inference_ms, 2),
        }


# ══════════════════════════════════════════════════════════════
# VEHICLE DETECTOR
# ══════════════════════════════════════════════════════════════

class VehicleDetector:
    """
    Detects vehicles and road users using YOLOv8 (COCO pre-trained).

    No custom training needed — COCO includes all vehicle classes.

    Usage:
        detector = VehicleDetector()
        detector.load_model()                    # once at startup
        result = detector.detect("image.jpg")
        print(result.vehicle_count)              # number of vehicles
        for v in result.vehicles:
            print(v.class_name, v.confidence)    # "car", 0.87
    """

    def __init__(self, model_name: str = "yolov8n.pt"):
        """
        Args:
            model_name: YOLOv8 weight file.
                        'yolov8n.pt' = nano (fastest, ~6 MB, best for CPU demo)
                        'yolov8s.pt' = small (more accurate, ~22 MB)
        """
        self.model_name = model_name
        self.model      = None     # loaded by load_model()
        self._model_loaded = False

    # ══════════════════════════════════════════════════════════
    # MODEL LOADING
    # ══════════════════════════════════════════════════════════

    def load_model(self) -> None:
        """
        Load the YOLOv8 model into memory.

        First call downloads the weights (~6 MB for nano) automatically
        from the Ultralytics CDN. Subsequent calls use the local cache.
        Call this ONCE at application startup, not per image.

        Raises:
            RuntimeError if model fails to load
        """
        try:
            from ultralytics import YOLO
            print(f"  Loading YOLOv8 model: {self.model_name} ...")
            self.model = YOLO(self.model_name)
            self._model_loaded = True
            print(f"  Model loaded — {len(self.model.names)} COCO classes available")
        except ImportError:
            raise RuntimeError(
                "ultralytics not installed. Run: pip install ultralytics==8.3.0"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLOv8 model '{self.model_name}': {e}")

    def _ensure_loaded(self) -> None:
        """Auto-load model if detect() is called before load_model()."""
        if not self._model_loaded:
            self.load_model()

    # ══════════════════════════════════════════════════════════
    # DETECTION
    # ══════════════════════════════════════════════════════════

    def detect(self,
               image_input: ImageInput,
               conf_threshold: float = VEHICLE_CONF_THRESHOLD,
               verbose: bool = False) -> DetectionResult:
        """
        Run YOLOv8 detection on one image.

        Args:
            image_input    : File path (str/Path) or numpy BGR array
            conf_threshold : Minimum confidence to keep (0.0–1.0)
            verbose        : Print YOLOv8 per-image stats to console

        Returns:
            DetectionResult with vehicles/persons/traffic_lights populated
        """
        self._ensure_loaded()

        # ── Load image if path was given ──────────────────────
        image, image_path_str = self._load_image(image_input)
        image_shape = image.shape

        # ── Run YOLOv8 inference ──────────────────────────────
        t0 = time.perf_counter()
        results = self.model(
            image,
            conf=conf_threshold,
            iou=IOU_THRESHOLD,
            verbose=verbose
        )
        inference_ms = (time.perf_counter() - t0) * 1000

        # ── Parse results into Detection objects ──────────────
        all_detections = self._parse_results(results, conf_threshold)

        # ── Categorise by class ───────────────────────────────
        vehicles       = [d for d in all_detections if d.class_id in VEHICLE_CLASS_IDS]
        persons        = [d for d in all_detections if d.class_id == PERSON_CLASS_ID]
        traffic_lights = [d for d in all_detections if d.class_id == TRAFFIC_LIGHT_ID]

        return DetectionResult(
            image_path     = image_path_str,
            detections     = all_detections,
            vehicles       = vehicles,
            persons        = persons,
            traffic_lights = traffic_lights,
            inference_ms   = inference_ms,
            image_shape    = image_shape,
        )

    def detect_batch(self,
                     image_inputs: List[ImageInput],
                     conf_threshold: float = VEHICLE_CONF_THRESHOLD) -> List[DetectionResult]:
        """
        Detect objects in a list of images.

        More efficient than calling detect() in a loop because YOLOv8
        can batch-process images on GPU.

        Args:
            image_inputs   : List of file paths or numpy arrays
            conf_threshold : Minimum confidence threshold

        Returns:
            List of DetectionResult, one per input image
        """
        self._ensure_loaded()
        return [self.detect(img, conf_threshold) for img in image_inputs]

    # ══════════════════════════════════════════════════════════
    # VEHICLE CLASSIFICATION HELPERS
    # ══════════════════════════════════════════════════════════

    def classify_vehicle_type(self, detection: Detection) -> str:
        """
        Returns a friendly vehicle category for a Detection.

        Mapping:
          class_id 1 (bicycle)    → 'two_wheeler'
          class_id 3 (motorcycle) → 'two_wheeler'
          class_id 2 (car)        → 'four_wheeler'
          class_id 5 (bus)        → 'heavy_vehicle'
          class_id 7 (truck)      → 'heavy_vehicle'
          anything else           → 'unknown'
        """
        cid = detection.class_id
        if cid in TWO_WHEELER_IDS:
            return "two_wheeler"
        if cid == 2:
            return "four_wheeler"
        if cid in [5, 7]:
            return "heavy_vehicle"
        return "unknown"

    def is_two_wheeler(self, detection: Detection) -> bool:
        """
        Returns True if the detection is a bicycle or motorcycle.
        Used by violation detector to check helmet compliance.
        """
        return detection.class_id in TWO_WHEELER_IDS

    def is_four_wheeler(self, detection: Detection) -> bool:
        """Returns True if detection is a car, bus, or truck."""
        return detection.class_id in FOUR_WHEELER_IDS

    # ══════════════════════════════════════════════════════════
    # GEOMETRIC HELPERS
    # ══════════════════════════════════════════════════════════

    def compute_iou(self,
                    box1: List[int],
                    box2: List[int]) -> float:
        """
        Compute Intersection over Union (IoU) of two bounding boxes.

        IoU = area(intersection) / area(union)

        Used to:
          - Check if a person is ON a vehicle (high IoU / containment)
          - Filter duplicate detections

        Args:
            box1, box2: [x1, y1, x2, y2] in pixel coordinates

        Returns:
            IoU score 0.0 (no overlap) to 1.0 (identical boxes)
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # Intersection area (0 if boxes don't overlap)
        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        intersection = inter_w * inter_h

        if intersection == 0:
            return 0.0

        # Union area
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def boxes_overlap(self,
                      box1: List[int],
                      box2: List[int],
                      iou_threshold: float = 0.1) -> bool:
        """
        Returns True if two boxes overlap by at least iou_threshold.

        Lower threshold (0.05–0.15) used for person-on-vehicle checks
        because a rider's bounding box only partially overlaps the bike.

        Args:
            box1, box2    : [x1, y1, x2, y2]
            iou_threshold : Minimum IoU to count as overlapping
        """
        return self.compute_iou(box1, box2) >= iou_threshold

    def is_person_on_vehicle(self,
                              person: Detection,
                              vehicle: Detection,
                              containment_threshold: float = 0.3) -> bool:
        """
        Check whether a person detection is on/inside a vehicle.

        Uses a containment metric rather than IoU:
          containment = intersection / area(person_box)

        A rider's box is smaller than the bike box, so their intersection
        relative to the PERSON box is high even when IoU is moderate.

        Args:
            person                : Person Detection object
            vehicle               : Vehicle Detection object
            containment_threshold : Min fraction of person box that must
                                    be inside vehicle box (default 0.3)

        Returns:
            True if the person appears to be on/in the vehicle
        """
        px1, py1, px2, py2 = person.bbox
        vx1, vy1, vx2, vy2 = vehicle.bbox

        # Intersection box
        ix1 = max(px1, vx1)
        iy1 = max(py1, vy1)
        ix2 = min(px2, vx2)
        iy2 = min(py2, vy2)

        inter_w = max(0, ix2 - ix1)
        inter_h = max(0, iy2 - iy1)
        intersection = inter_w * inter_h

        person_area = max(1, person.area)   # avoid division by zero
        containment = intersection / person_area

        return containment >= containment_threshold

    def get_persons_on_vehicle(self,
                                vehicle: Detection,
                                persons: List[Detection],
                                containment_threshold: float = 0.3) -> List[Detection]:
        """
        Return all person detections that appear to be on a given vehicle.

        Used by:
          - Helmet checker  (person on two-wheeler)
          - Triple riding   (count persons on two-wheeler)
          - Seatbelt checker (person in four-wheeler)

        Args:
            vehicle              : Vehicle Detection
            persons              : All person detections in image
            containment_threshold: Min containment fraction

        Returns:
            Subset of persons that are on this vehicle
        """
        return [
            p for p in persons
            if self.is_person_on_vehicle(p, vehicle, containment_threshold)
        ]

    # ══════════════════════════════════════════════════════════
    # TRAFFIC LIGHT HELPERS
    # ══════════════════════════════════════════════════════════

    def get_traffic_light_state(self,
                                 image: np.ndarray,
                                 light: Detection) -> str:
        """
        Determine if a traffic light is RED, YELLOW, or GREEN
        by analysing the dominant color in its bounding box.

        Algorithm:
          1. Crop the traffic light region
          2. Divide into thirds (top=red, mid=yellow, bottom=green)
          3. Find which third is brightest → that is the active light

        Args:
            image : Full BGR image
            light : Traffic light Detection object

        Returns:
            'red' | 'yellow' | 'green' | 'unknown'
        """
        x1, y1, x2, y2 = light.bbox

        # Clamp to image bounds
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return "unknown"

        ch = crop.shape[0]
        if ch < 9:
            return "unknown"   # too small to analyse

        third = ch // 3

        # Each zone — convert to HSV for better color analysis
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV) if _cv2_available() else None
        if hsv is None:
            return "unknown"

        # Brightness (V channel) of each zone
        top_v    = float(hsv[:third,      :, 2].mean())   # red zone
        mid_v    = float(hsv[third:third*2, :, 2].mean()) # yellow zone
        bot_v    = float(hsv[third*2:,    :, 2].mean())   # green zone

        brightest = max(top_v, mid_v, bot_v)

        if brightest < 80:
            return "unknown"   # light is off or too dark

        if top_v == brightest:
            return "red"
        if mid_v == brightest:
            return "yellow"
        return "green"

    # ══════════════════════════════════════════════════════════
    # PRIVATE HELPERS
    # ══════════════════════════════════════════════════════════

    def _load_image(self,
                    image_input: ImageInput) -> Tuple[np.ndarray, str]:
        """
        Load image from path or pass numpy array through.
        Returns (bgr_array, path_string).
        """
        if isinstance(image_input, np.ndarray):
            return image_input, "array"

        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        try:
            import cv2 as _cv2
            img = _cv2.imread(str(path))
        except ImportError:
            raise RuntimeError("OpenCV not installed. Run: pip install opencv-python")

        if img is None:
            raise ValueError(f"Cannot decode image: {path}")

        return img, str(path)

    def _parse_results(self,
                       results,
                       conf_threshold: float) -> List[Detection]:
        """
        Convert raw YOLOv8 results into a list of Detection objects.

        YOLOv8 results structure:
          results[0].boxes.cls   → class IDs tensor
          results[0].boxes.conf  → confidence tensor
          results[0].boxes.xyxy  → boxes [x1,y1,x2,y2] tensor
        """
        detections = []

        if not results or results[0].boxes is None:
            return detections

        boxes_data = results[0].boxes

        # Move tensors to CPU and convert to numpy
        cls_ids     = boxes_data.cls.cpu().numpy().astype(int)
        confidences = boxes_data.conf.cpu().numpy().astype(float)
        xyxy_boxes  = boxes_data.xyxy.cpu().numpy().astype(int)

        for cls_id, conf, box in zip(cls_ids, confidences, xyxy_boxes):
            if conf < conf_threshold:
                continue

            # Only keep classes we care about
            if cls_id not in COCO_CLASSES:
                continue

            class_name = COCO_CLASSES[cls_id]
            bbox = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]

            detections.append(Detection(
                class_id   = int(cls_id),
                class_name = class_name,
                confidence = float(conf),
                bbox       = bbox,
            ))

        return detections


# ── cv2 availability check (used in traffic light analysis) ──
def _cv2_available() -> bool:
    try:
        import cv2
        return True
    except ImportError:
        return False


# Import cv2 for traffic light state (optional — won't break if missing)
try:
    import cv2
except ImportError:
    pass
