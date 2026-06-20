"""
startup.py
==========
First-run initialisation script.

Called automatically by the Streamlit app on startup.
Safe to call multiple times — all operations are idempotent.

What it does:
  1. Creates required directories
  2. Downloads YOLOv8 base model if missing
  3. Warms up EasyOCR (triggers model download once)
  4. Initialises the SQLite database
  5. Prints a status summary
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def ensure_dirs() -> None:
    """Create all required runtime directories."""
    from config import MODELS_DIR, EVIDENCE_DIR, SAMPLES_DIR, TEST_DIR
    for d in [MODELS_DIR, EVIDENCE_DIR, SAMPLES_DIR, TEST_DIR,
              PROJECT_ROOT / "data" / "datasets"]:
        d.mkdir(parents=True, exist_ok=True)
    print("  [OK] Directories ready")


def download_yolo() -> bool:
    """
    Download YOLOv8s base weights if not already cached.
    Downloads the file only — does NOT load into PyTorch memory at startup.
    """
    try:
        # Use ultralytics' own download utility — finds/downloads the file,
        # returns the local path without instantiating a full YOLO model.
        from ultralytics.utils.downloads import attempt_download_asset
        path = attempt_download_asset("yolov8s.pt")
        print(f"  [OK] YOLOv8s ready  ({path})")
        return True
    except Exception:
        # Fallback: instantiate briefly just to trigger download, then free memory
        try:
            from ultralytics import YOLO
            print("  [..] Checking YOLOv8s weights ...")
            model = YOLO("yolov8s.pt")
            del model          # free PyTorch memory immediately
            print("  [OK] YOLOv8s ready")
            return True
        except Exception as e:
            print(f"  [!!] YOLOv8 download failed: {e}")
            return False


def warm_easyocr() -> bool:
    """
    Skip EasyOCR warm-up at startup — it downloads 100 MB and causes
    cloud health-check timeouts. EasyOCR loads lazily on first analysis.
    """
    print("  [--] EasyOCR: will load on first use (lazy)")
    return True


def init_database() -> bool:
    """Initialise SQLite tables (safe to call on existing DB)."""
    try:
        from core.database import init_db
        init_db()
        print("  [OK] Database initialised")
        return True
    except Exception as e:
        print(f"  [!!] Database init failed: {e}")
        return False


def check_specialized_models() -> None:
    """Report which optional specialized models are available."""
    from config import (YOLO_HELMET_MODEL, YOLO_SEATBELT_MODEL,
                        YOLO_PLATE_MODEL, MODELS_DIR)
    specs = {
        "Helmet":        YOLO_HELMET_MODEL,
        "Seatbelt":      YOLO_SEATBELT_MODEL,
        "License Plate": YOLO_PLATE_MODEL,
    }
    for name, path in specs.items():
        if Path(path).exists():
            size = Path(path).stat().st_size / 1_048_576
            print(f"  [OK] {name} model: {Path(path).name}  ({size:.1f} MB)")
        else:
            print(f"  [--] {name} model: not found — using heuristic fallback")


def run() -> None:
    """Run the full startup sequence."""
    print("\n" + "=" * 52)
    print("  TrafficAI — Startup Initialisation")
    print("=" * 52)

    ensure_dirs()
    ok_db    = init_database()
    ok_yolo  = download_yolo()
    ok_ocr   = warm_easyocr()
    check_specialized_models()

    print("=" * 52)
    if ok_yolo and ok_ocr and ok_db:
        print("  All systems ready.  Starting dashboard ...\n")
    else:
        print("  WARNING: Some components failed to initialise.")
        print("  The app will start but some features may be limited.\n")


# Allow running directly: python startup.py
if __name__ == "__main__":
    run()
