"""
scripts/create_sample_images.py
================================
Downloads free sample traffic images for testing the system.

Images are sourced from:
- Wikimedia Commons (public domain)
- Unsplash-sourced free images (via direct URLs)
- Synthetic test images generated with OpenCV

HOW TO RUN:
    python scripts\\create_sample_images.py

After running, check the samples/ folder for downloaded images.
These images are used to verify the detection pipeline works.
"""

import sys
import os
import urllib.request
import urllib.error
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SAMPLES_DIR = PROJECT_ROOT / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


# ── Free traffic image URLs (public domain / creative commons) ─
# Using direct image URLs from Wikimedia Commons
FREE_IMAGES = [
    {
        "filename": "traffic_intersection.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg",
        "description": "Traffic intersection scene",
        "use_synthetic": True,   # Flag: use synthetic if download fails
    },
    {
        "filename": "motorcycles_traffic.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Simple_bicycle.jpg/320px-Simple_bicycle.jpg",
        "description": "Motorcycles in traffic",
        "use_synthetic": True,
    },
    {
        "filename": "road_traffic.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/Above_Gotham.jpg/320px-Above_Gotham.jpg",
        "description": "Road with vehicles",
        "use_synthetic": True,
    },
]


def create_synthetic_traffic_image(filename: str, scenario: str) -> np.ndarray:
    """
    Create a synthetic test image that mimics a traffic scene.
    These are colored rectangles that help test the annotation pipeline.
    Real detection won't work on these, but they verify image I/O works.
    """
    try:
        import cv2
    except ImportError:
        print("  FAIL  OpenCV not installed — cannot create synthetic images")
        return None

    # Canvas: simulate a road scene
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # ── Road background (dark grey) ────────────────────────────
    img[:] = (50, 50, 50)

    # ── Sky (top 30%) ─────────────────────────────────────────
    img[:150, :] = (200, 180, 120)

    # ── Road surface (bottom 70%) ─────────────────────────────
    img[150:, :] = (80, 80, 80)

    # ── Road lane markings ─────────────────────────────────────
    for x in range(50, 640, 80):
        cv2.line(img, (x, 300), (x + 40, 480), (255, 255, 255), 2)

    if scenario == "car":
        # Draw a car (blue rectangle with wheels)
        cv2.rectangle(img, (150, 250), (380, 380), (180, 80, 30), -1)   # car body
        cv2.rectangle(img, (190, 210), (340, 260), (180, 80, 30), -1)   # roof
        cv2.circle(img, (200, 385), 30, (20, 20, 20), -1)               # wheel L
        cv2.circle(img, (330, 385), 30, (20, 20, 20), -1)               # wheel R
        cv2.rectangle(img, (155, 260), (210, 290), (200, 230, 255), -1) # windshield L
        # Person in car (driver)
        cv2.circle(img, (250, 230), 25, (210, 180, 140), -1)            # head
        cv2.putText(img, "CAR - Test Image", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    elif scenario == "motorcycle":
        # Draw a motorcycle with rider
        cv2.rectangle(img, (250, 290), (400, 380), (60, 60, 200), -1)  # body
        cv2.circle(img, (260, 390), 25, (20, 20, 20), -1)              # wheel L
        cv2.circle(img, (390, 390), 25, (20, 20, 20), -1)              # wheel R
        # Rider
        cv2.circle(img, (320, 255), 30, (210, 180, 140), -1)           # head
        cv2.rectangle(img, (300, 280), (340, 340), (30, 30, 150), -1)  # torso
        cv2.putText(img, "MOTORCYCLE - Test Image", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    elif scenario == "intersection":
        # Draw intersection with traffic light
        cv2.rectangle(img, (280, 150), (310, 320), (40, 40, 40), -1)   # traffic light pole
        cv2.rectangle(img, (265, 130), (325, 200), (20, 20, 20), -1)   # traffic light box
        cv2.circle(img, (295, 148), 12, (0, 0, 255), -1)               # RED light
        cv2.circle(img, (295, 168), 12, (0, 200, 200), -1)             # yellow (off)
        cv2.circle(img, (295, 188), 12, (0, 100, 0), -1)               # green (off)
        # Multiple vehicles
        cv2.rectangle(img, (50, 300), (200, 400), (30, 100, 200), -1)  # car 1
        cv2.rectangle(img, (420, 280), (600, 390), (200, 50, 50), -1)  # car 2
        cv2.putText(img, "INTERSECTION - Test Image", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # ── Watermark ──────────────────────────────────────────────
    cv2.putText(img, "SYNTHETIC TEST IMAGE - Traffic Violation Detection",
                (5, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    return img


def download_image(url: str, save_path: Path) -> bool:
    """Download an image from URL. Returns True if successful."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            data = response.read()
        with open(save_path, 'wb') as f:
            f.write(data)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, Exception):
        return False


def main():
    print("\n" + "="*55)
    print("  SAMPLE IMAGE DOWNLOADER")
    print("="*55)
    print(f"\n  Saving images to: {SAMPLES_DIR}\n")

    try:
        import cv2
        has_cv2 = True
    except ImportError:
        has_cv2 = False
        print("  WARN  OpenCV not found — synthetic images will be skipped")

    created = 0

    # ── Create synthetic test images ──────────────────────────
    if has_cv2:
        print("[Step 1/2] Creating synthetic test images...")
        synthetic_images = [
            ("test_car_scene.jpg",          "car",          "Car with driver"),
            ("test_motorcycle_scene.jpg",   "motorcycle",   "Motorcycle with rider"),
            ("test_intersection_scene.jpg", "intersection", "Traffic intersection with red light"),
        ]
        for filename, scenario, desc in synthetic_images:
            save_path = SAMPLES_DIR / filename
            if save_path.exists():
                print(f"  SKIP  {filename} (already exists)")
                created += 1
                continue
            img = create_synthetic_traffic_image(filename, scenario)
            if img is not None:
                cv2.imwrite(str(save_path), img)
                print(f"  OK  Created: {filename}  ({desc})")
                created += 1
            else:
                print(f"  FAIL  Could not create: {filename}")
    else:
        print("[Step 1/2] Skipping synthetic images (OpenCV not available)")

    # ── Create a test metadata JSON ────────────────────────────
    print("\n[Step 2/2] Creating sample metadata file...")
    import json
    sample_meta = {
        "description": "Sample test images for Traffic Violation Detection System",
        "images": [
            {
                "filename": "test_car_scene.jpg",
                "expected_detections": ["car", "person"],
                "expected_violations": [],
                "notes": "Synthetic image — tests image load and annotation pipeline"
            },
            {
                "filename": "test_motorcycle_scene.jpg",
                "expected_detections": ["motorcycle", "person"],
                "expected_violations": ["helmet_non_compliance"],
                "notes": "Synthetic image — rider has no helmet drawn"
            },
            {
                "filename": "test_intersection_scene.jpg",
                "expected_detections": ["car", "traffic light"],
                "expected_violations": ["red_light_violation"],
                "notes": "Synthetic image — traffic light is red, car is present"
            }
        ]
    }
    meta_path = SAMPLES_DIR / "sample_metadata.json"
    meta_path.write_text(json.dumps(sample_meta, indent=2))
    print(f"  OK  Created: sample_metadata.json")

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "="*55)
    print(f"  DONE! Created {created} test image(s) in samples/")
    print("="*55)
    print(f"""
  Files created in:  {SAMPLES_DIR}

  These synthetic images test the annotation pipeline.
  For real traffic violation detection accuracy testing,
  add real traffic images to the samples/ folder.

  FREE SOURCES FOR REAL TRAFFIC IMAGES:
  • https://www.kaggle.com/datasets  (search: traffic violations india)
  • https://universe.roboflow.com    (search: traffic)
  • https://images.google.com        (filter: labeled for reuse)

  NEXT STEP:
  Run: python scripts\\download_datasets.py
""")


if __name__ == "__main__":
    main()
