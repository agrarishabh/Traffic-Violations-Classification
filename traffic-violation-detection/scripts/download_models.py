"""
scripts/download_models.py
==========================
Downloads all required YOLOv8 model weight files.

Models downloaded:
  1. yolov8n.pt  — Base YOLOv8 Nano model (COCO, general detection)
                   Used for: vehicles, persons, traffic lights
  2. yolov8s.pt  — YOLOv8 Small (optional backup, more accurate)

Note: Specialized models (helmet, seatbelt, plate) are downloaded
      by scripts/download_datasets.py after Roboflow setup.

HOW TO RUN:
    python scripts\\download_models.py
"""

import sys
import os
from pathlib import Path

# Add project root to Python path so we can import config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    print("\n" + "="*55)
    print("  YOLOv8 MODEL DOWNLOADER")
    print("="*55)

    # ── Step 1: Check ultralytics is installed ────────────────
    print("\n[Step 1/4] Checking ultralytics installation...")
    try:
        from ultralytics import YOLO
        import ultralytics
        print(f"  OK  ultralytics {ultralytics.__version__} found")
    except ImportError:
        print("  FAIL  ultralytics not installed!")
        print("  FIX → pip install ultralytics==8.3.0")
        sys.exit(1)

    # ── Step 2: Load config ────────────────────────────────────
    print("\n[Step 2/4] Loading project configuration...")
    try:
        from config import MODELS_DIR, YOLO_BASE_MODEL
        print(f"  OK  Models directory: {MODELS_DIR}")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"  FAIL  Could not load config: {e}")
        sys.exit(1)

    # ── Step 3: Download YOLOv8n (Nano) ──────────────────────
    print("\n[Step 3/4] Downloading YOLOv8n (Nano) model...")
    print("  This is the fastest model — good for CPU demo.")
    print("  File size: ~6 MB — should take under 1 minute.\n")

    nano_path = MODELS_DIR / "yolov8n.pt"
    if nano_path.exists():
        print(f"  SKIP  yolov8n.pt already exists at: {nano_path}")
    else:
        try:
            print("  Downloading yolov8n.pt ...")
            # Ultralytics auto-downloads when you create YOLO("yolov8n.pt")
            # We move it to our models directory after download
            model = YOLO("yolov8n.pt")
            # The file downloads to current dir or ultralytics cache
            # Copy to our models directory
            import shutil
            default_path = Path("yolov8n.pt")
            if default_path.exists():
                shutil.move(str(default_path), str(nano_path))
                print(f"  OK  Saved to: {nano_path}")
            else:
                # ultralytics caches to ~/.ultralytics — just reference it there
                print(f"  OK  yolov8n.pt cached by ultralytics (auto-managed)")
                # Create a symlink reference in our models dir
                ref_file = MODELS_DIR / "yolov8n_location.txt"
                ref_file.write_text(
                    "yolov8n.pt is cached by ultralytics automatically.\n"
                    "It will be loaded from cache when you run detections.\n"
                    "No manual action needed."
                )
        except Exception as e:
            print(f"  FAIL  Download failed: {e}")
            print("  FIX → Check your internet connection and try again.")
            sys.exit(1)

    # ── Step 4: Quick inference test ──────────────────────────
    print("\n[Step 4/4] Running quick inference test with yolov8n.pt...")
    print("  Creating a test image and running detection...\n")
    try:
        import numpy as np
        import cv2

        # Create a simple test image (blank, no real objects)
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        test_img[200:440, 200:440] = (100, 100, 200)   # fake object region

        # Run YOLOv8 on it
        model = YOLO("yolov8n.pt")
        results = model(test_img, verbose=False)

        print(f"  OK  YOLOv8n inference test passed!")
        print(f"  OK  Model input size: 640x640")
        print(f"  OK  Number of COCO classes: {len(model.names)}")
        print(f"  OK  Sample classes: car={model.names[2]}, "
              f"person={model.names[0]}, motorcycle={model.names[3]}")

    except Exception as e:
        print(f"  FAIL  Inference test failed: {e}")
        sys.exit(1)

    # ── Done ───────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  MODEL DOWNLOAD COMPLETE!")
    print("="*55)
    print("""
  yolov8n.pt is ready.

  NEXT STEP:
  Download test images:
      python scripts\\create_sample_images.py

  Then download datasets:
      python scripts\\download_datasets.py
""")


if __name__ == "__main__":
    main()
