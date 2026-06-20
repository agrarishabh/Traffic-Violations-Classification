"""
config.py
=========
Central configuration for the Traffic Violation Detection System.
All paths, thresholds, and constants live here.
To change any setting, edit this file — nowhere else.
"""

import os
from pathlib import Path"""

import os
from pathlib import Path

# ============================================================
# ROOT PATHS
# ============================================================
# BASE_DIR = the folder where this config.py file lives
BASE_DIR = Path(__file__).resolve().parent

# Cloud environments (Railway, Render, HuggingFace, Streamlit Cloud)
# often have read-only filesystems except /tmp and home directory.
# TRAFFICAI_CLOUD=1 tells us to redirect writable paths to /tmp.
_CLOUD = os.environ.get("TRAFFICAI_CLOUD", "0") == "1"
_WRITE_ROOT = Path("/tmp/trafficai") if _CLOUD else BASE_DIR

DATA_DIR        = BASE_DIR / "data"
MODELS_DIR      = DATA_DIR / "models"        # YOLOv8 .pt weight files
DATASETS_DIR    = DATA_DIR / "datasets"      # Training/validation datasets
TEST_DIR        = DATA_DIR / "test"          # Test images for evaluation
EVIDENCE_DIR    = _WRITE_ROOT / "evidence"   # Annotated output images + JSON
SAMPLES_DIR     = BASE_DIR / "samples"       # Demo/sample images
DB_PATH         = _WRITE_ROOT / "violations.db"  # SQLite database file

# ============================================================
# YOLOV8 MODEL WEIGHTS
# ============================================================
# These files are downloaded automatically by scripts/download_models.py
# 'n' = nano (fastest, least accurate) — good for CPU demo
# 's' = small | 'm' = medium | 'l' = large | 'x' = extra-large
YOLO_BASE_MODEL         = "yolov8s.pt"          # Small model — better accuracy than nano
YOLO_HELMET_MODEL       = MODELS_DIR / "helmet_yolov8.pt"
YOLO_SEATBELT_MODEL     = MODELS_DIR / "seatbelt_yolov8.pt"
YOLO_PLATE_MODEL        = MODELS_DIR / "license_plate_yolov8.pt"

# ============================================================
# DETECTION CONFIDENCE THRESHOLDS
# ============================================================
# A detection is accepted only if its confidence >= this value (0.0 to 1.0)
# Lowered from 0.50 for better recall on real traffic photos
VEHICLE_CONF_THRESHOLD      = 0.25   # General vehicle detection
PERSON_CONF_THRESHOLD       = 0.25   # Person/rider detection
HELMET_CONF_THRESHOLD       = 0.35   # Helmet detection
SEATBELT_CONF_THRESHOLD     = 0.40   # Seatbelt detection
PLATE_CONF_THRESHOLD        = 0.35   # License plate detection
VIOLATION_CONF_THRESHOLD    = 0.35   # Minimum confidence to log a violation

# IoU threshold for Non-Maximum Suppression (removes duplicate boxes)
IOU_THRESHOLD = 0.40

# ============================================================
# COCO DATASET CLASS IDs  (what YOLOv8 base model can detect)
# ============================================================
# These are the numeric class IDs in the COCO dataset
COCO_CLASSES = {
    0:  "person",
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
    9:  "traffic light",
    11: "stop sign",
}

# Which COCO IDs count as "vehicles" for us
VEHICLE_CLASS_IDS   = [1, 2, 3, 5, 7]   # bicycle, car, motorcycle, bus, truck
TWO_WHEELER_IDS     = [1, 3]             # bicycle, motorcycle
FOUR_WHEELER_IDS    = [2, 5, 7]          # car, bus, truck
PERSON_CLASS_ID     = 0
TRAFFIC_LIGHT_ID    = 9

# ============================================================
# VIOLATION TYPES
# ============================================================
# These are the violation class names used throughout the system
class ViolationType:
    HELMET_VIOLATION    = "helmet_non_compliance"
    SEATBELT_VIOLATION  = "seatbelt_non_compliance"
    TRIPLE_RIDING       = "triple_riding"
    WRONG_SIDE          = "wrong_side_driving"
    STOP_LINE           = "stop_line_violation"
    RED_LIGHT           = "red_light_violation"
    ILLEGAL_PARKING     = "illegal_parking"

# Human-readable display names for each violation
VIOLATION_DISPLAY_NAMES = {
    ViolationType.HELMET_VIOLATION:   "Helmet Non-Compliance",
    ViolationType.SEATBELT_VIOLATION: "Seatbelt Non-Compliance",
    ViolationType.TRIPLE_RIDING:      "Triple Riding",
    ViolationType.WRONG_SIDE:         "Wrong-Side Driving",
    ViolationType.STOP_LINE:          "Stop-Line Violation",
    ViolationType.RED_LIGHT:          "Red-Light Violation",
    ViolationType.ILLEGAL_PARKING:    "Illegal Parking",
}

# Fine amounts (INR) per violation type — for the report
VIOLATION_FINES = {
    ViolationType.HELMET_VIOLATION:   1000,
    ViolationType.SEATBELT_VIOLATION: 1000,
    ViolationType.TRIPLE_RIDING:      2000,
    ViolationType.WRONG_SIDE:         5000,
    ViolationType.STOP_LINE:          500,
    ViolationType.RED_LIGHT:          1000,
    ViolationType.ILLEGAL_PARKING:    500,
}

# ============================================================
# ANNOTATION / EVIDENCE SETTINGS
# ============================================================
# Colors for drawing bounding boxes — (B, G, R) in OpenCV format
COLORS = {
    "violation":   (0,   0,   255),   # Red   — confirmed violation
    "warning":     (0,   165, 255),   # Orange — uncertain
    "compliant":   (0,   255, 0),     # Green  — no violation
    "vehicle":     (255, 255, 0),     # Cyan   — vehicle box
    "person":      (255, 0,   255),   # Purple — person box
    "plate":       (0,   255, 255),   # Yellow — license plate box
}

BOX_THICKNESS       = 2     # Bounding box line thickness in pixels
FONT_SCALE          = 0.6   # Text size for labels
FONT_THICKNESS      = 2     # Text boldness

# ============================================================
# IMAGE PREPROCESSING SETTINGS
# ============================================================
INPUT_WIDTH     = 640   # YOLOv8 default input size (do not change)
INPUT_HEIGHT    = 640
CLAHE_CLIP      = 2.0   # CLAHE contrast enhancement limit
CLAHE_GRID      = 8     # CLAHE grid tile size

# ============================================================
# TRIPLE RIDING THRESHOLD
# ============================================================
# If more than this many persons are detected on a single two-wheeler → violation
TRIPLE_RIDING_THRESHOLD = 2   # more than 2 persons = triple riding

# Person-on-vehicle containment threshold (lower = easier to match)
# Lowered from 0.25 for better detection on real photos
PERSON_ON_VEHICLE_THRESHOLD = 0.15

# ============================================================
# DATABASE SETTINGS
# ============================================================
DATABASE_URL = f"sqlite:///{DB_PATH}"

# ============================================================
# API SETTINGS
# ============================================================
API_HOST    = "0.0.0.0"
API_PORT    = 8000
API_TITLE   = "Traffic Violation Detection API"
API_VERSION = "1.0.0"

# ============================================================
# DASHBOARD SETTINGS
# ============================================================
DASHBOARD_TITLE     = "Traffic Violation Detection System"
DASHBOARD_ICON      = "🚦"
MAX_UPLOAD_SIZE_MB  = 10    # Maximum image upload size

# ============================================================
# ENSURE DIRECTORIES EXIST AT IMPORT TIME
# ============================================================
for _dir in [MODELS_DIR, DATASETS_DIR, TEST_DIR, EVIDENCE_DIR, SAMPLES_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
