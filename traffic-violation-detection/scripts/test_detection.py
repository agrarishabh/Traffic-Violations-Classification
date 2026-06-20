"""
scripts/test_detection.py
==========================
Visual Detection Demo — Phase 3

Loads YOLOv8n, runs detection on sample images, draws color-coded
bounding boxes, and saves annotated results to samples/detected/.

Color scheme:
  CYAN   rectangle = vehicle (car/bike/bus/truck)
  PURPLE rectangle = person
  YELLOW rectangle = traffic light
  WHITE  rectangle = other detected object

HOW TO RUN:
    python scripts\\test_detection.py

OUTPUT:
    samples/detected/
        <filename>_detected.jpg   ← annotated image
        detection_summary.txt     ← per-image stats
"""

import sys
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.detector import VehicleDetector, DetectionResult, Detection
from config import COLORS, BOX_THICKNESS, FONT_SCALE, FONT_THICKNESS

OUTPUT_DIR = PROJECT_ROOT / "samples" / "detected"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR = PROJECT_ROOT / "samples"


# ══════════════════════════════════════════════════════════════
# ANNOTATION HELPERS
# ══════════════════════════════════════════════════════════════

def pick_color(detection: Detection) -> tuple:
    """Return BGR draw color based on detected class."""
    cid = detection.class_id
    if cid == 0:   return COLORS["person"]    # purple
    if cid in [1, 2, 3, 5, 7]: return COLORS["vehicle"]  # cyan
    if cid == 9:   return COLORS["plate"]     # yellow (traffic light)
    return (255, 255, 255)                    # white — anything else


def draw_detection(image: np.ndarray, det: Detection) -> np.ndarray:
    """Draw one bounding box + label on the image."""
    x1, y1, x2, y2 = det.bbox
    color = pick_color(det)

    # Bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, BOX_THICKNESS)

    # Label background (filled rectangle behind text)
    label = f"{det.class_name} {det.confidence:.0%}"
    (tw, th), _ = cv2.getTextSize(label,
                                   cv2.FONT_HERSHEY_SIMPLEX,
                                   FONT_SCALE, FONT_THICKNESS)
    label_y = max(y1 - 5, th + 5)
    cv2.rectangle(image,
                  (x1, label_y - th - 6),
                  (x1 + tw + 4, label_y + 2),
                  color, -1)

    # Label text (black on colored background)
    cv2.putText(image, label,
                (x1 + 2, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE, (0, 0, 0), FONT_THICKNESS)

    return image


def draw_all_detections(image: np.ndarray,
                         result: DetectionResult) -> np.ndarray:
    """Draw every detection from a DetectionResult onto the image."""
    annotated = image.copy()
    for det in result.detections:
        annotated = draw_detection(annotated, det)
    return annotated


def add_stats_panel(image: np.ndarray, result: DetectionResult) -> np.ndarray:
    """
    Add a dark stats bar at the bottom of the image showing:
      Vehicles: N  |  Persons: N  |  Traffic Lights: N  |  Time: Xms
    """
    h, w = image.shape[:2]
    panel_h = 36
    canvas = np.zeros((h + panel_h, w, 3), dtype=np.uint8)
    canvas[:h] = image
    canvas[h:] = (20, 20, 35)   # dark navy bar

    stats = (f"Vehicles: {result.vehicle_count}   "
             f"Persons: {result.person_count}   "
             f"Traffic Lights: {len(result.traffic_lights)}   "
             f"Total: {len(result.detections)}   "
             f"Inference: {result.inference_ms:.0f}ms")

    cv2.putText(canvas, stats,
                (10, h + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (200, 220, 255), 1)
    return canvas


def add_legend(image: np.ndarray) -> np.ndarray:
    """Add a small color legend in the top-right corner."""
    legend_items = [
        ("Vehicle",       COLORS["vehicle"]),
        ("Person",        COLORS["person"]),
        ("Traffic Light", COLORS["plate"]),
    ]
    x_start = image.shape[1] - 175
    y_start = 10
    for i, (label, color) in enumerate(legend_items):
        y = y_start + i * 22
        cv2.rectangle(image, (x_start, y), (x_start + 16, y + 14), color, -1)
        cv2.putText(image, label,
                    (x_start + 22, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return image


# ══════════════════════════════════════════════════════════════
# SYNTHETIC SCENE GENERATOR (fallback if no real images)
# ══════════════════════════════════════════════════════════════

def make_synthetic_scenes() -> list:
    """
    Generate synthetic traffic scenes as numpy arrays.
    YOLOv8 may not detect objects in these (they are geometric shapes,
    not real photos), but they confirm the PIPELINE works end-to-end.
    """
    scenes = []
    for name, color in [
        ("scene_car",   (30, 80, 200)),
        ("scene_multi", (80, 160, 80)),
    ]:
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        img[:320] = (200, 180, 120)   # sky
        img[320:] = (60, 62, 68)      # road
        cv2.rectangle(img, (80,  350), (350, 520), color, -1)   # vehicle body
        cv2.rectangle(img, (130, 290), (300, 360), color, -1)   # vehicle roof
        cv2.circle(img, (130, 530), 30, (15, 15, 15), -1)       # wheel L
        cv2.circle(img, (300, 530), 30, (15, 15, 15), -1)       # wheel R
        cv2.rectangle(img, (90,  300), (150, 355), (200, 225, 255), -1)  # windshield
        cv2.circle(img, (480, 320), 28, (210, 185, 140), -1)    # person head
        cv2.rectangle(img, (460, 345), (500, 500), (20, 80, 160), -1)   # torso
        # Traffic light
        cv2.rectangle(img, (570, 150), (610, 310), (20, 20, 20), -1)
        cv2.circle(img, (590, 175), 15, (0, 0, 230), -1)   # red
        cv2.line(img, (200, 580), (640, 580), (255, 255, 255), 4)  # stop line
        # Fine grid for detection confidence boost
        for y in range(0, 640, 30):
            cv2.line(img, (0, y), (640, y), (40, 42, 45), 1)
        scenes.append((name, img))
    return scenes


# ══════════════════════════════════════════════════════════════
# MAIN DEMO
# ══════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 58)
    print("  VEHICLE DETECTION DEMO — Phase 3")
    print("=" * 58)

    # ── Step 1: Load model ────────────────────────────────────
    print("\n[Step 1/3] Loading YOLOv8n model ...")
    detector = VehicleDetector(model_name="yolov8n.pt")
    detector.load_model()
    print("  Model ready\n")

    # ── Step 2: Collect images ────────────────────────────────
    print("[Step 2/3] Finding images ...")
    real_images = list(SAMPLES_DIR.glob("*.jpg")) + \
                  list(SAMPLES_DIR.glob("*.png"))
    real_images = [p for p in real_images
                   if "detected" not in str(p) and "comparison" not in str(p)]

    images_to_process = []

    if real_images:
        print(f"  Found {len(real_images)} real image(s) in samples/")
        for p in real_images:
            img = cv2.imread(str(p))
            if img is not None:
                images_to_process.append((p.stem, img))
    else:
        print("  No real images found — using synthetic scenes")
        print("  NOTE: YOLOv8 detects real objects. Synthetic shapes may not trigger")
        print("        detections but the pipeline will still run successfully.")

    # Always include synthetic scenes
    for name, img in make_synthetic_scenes():
        images_to_process.append((name, img))

    print(f"  Processing {len(images_to_process)} image(s)\n")

    # ── Step 3: Detect + Annotate + Save ─────────────────────
    print("[Step 3/3] Running detection on each image ...")
    print("-" * 58)

    summary_lines = ["DETECTION SUMMARY\n" + "=" * 58 + "\n"]
    total_vehicles = 0
    total_persons  = 0

    for name, image in images_to_process:
        print(f"\n  Image: {name}")

        # Run detection
        result = detector.detect(image, conf_threshold=0.35)

        # Annotate
        annotated = draw_all_detections(image, result)
        annotated = add_legend(annotated)
        annotated = add_stats_panel(annotated, result)

        # Save
        out_path = OUTPUT_DIR / f"{name}_detected.jpg"
        cv2.imwrite(str(out_path), annotated)

        # Print stats
        print(f"  Vehicles      : {result.vehicle_count}")
        print(f"  Persons       : {result.person_count}")
        print(f"  Traffic lights: {len(result.traffic_lights)}")
        print(f"  Total objects : {len(result.detections)}")
        print(f"  Inference     : {result.inference_ms:.1f}ms")
        print(f"  Saved         : {out_path.name}")

        # Per-detection breakdown
        if result.detections:
            print("  Detections:")
            for det in result.detections:
                v_type = detector.classify_vehicle_type(det)
                print(f"    - {det.class_name:<14} conf={det.confidence:.2f}"
                      f"  bbox={det.bbox}  type={v_type}")

            # Person-on-vehicle analysis
            for vehicle in result.vehicles:
                if detector.is_two_wheeler(vehicle):
                    riders = detector.get_persons_on_vehicle(
                        vehicle, result.persons, containment_threshold=0.25)
                    if riders:
                        print(f"  Two-wheeler has {len(riders)} rider(s)")
                        if len(riders) > 2:
                            print("  !! POTENTIAL TRIPLE RIDING !!")
        else:
            print("  No objects detected above confidence threshold")
            print("  (Normal for synthetic geometric images)")

        # Summary record
        total_vehicles += result.vehicle_count
        total_persons  += result.person_count
        summary_lines.append(
            f"Image: {name}\n"
            f"  Vehicles: {result.vehicle_count}  "
            f"Persons: {result.person_count}  "
            f"Lights: {len(result.traffic_lights)}  "
            f"Time: {result.inference_ms:.0f}ms\n"
        )

    # ── Save summary ─────────────────────────────────────────
    summary_path = OUTPUT_DIR / "detection_summary.txt"
    summary_lines.append(f"\nTOTAL: {total_vehicles} vehicles, {total_persons} persons")
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    # ── Final output ─────────────────────────────────────────
    print("\n" + "=" * 58)
    print("  DETECTION DEMO COMPLETE")
    print("=" * 58)
    print(f"""
  Annotated images saved to:
  {OUTPUT_DIR}

  HOW TO VIEW:
  Open File Explorer -> samples -> detected
  Each *_detected.jpg shows bounding boxes:
    CYAN   = vehicle
    PURPLE = person
    YELLOW = traffic light

  STATS:
  Total vehicles detected : {total_vehicles}
  Total persons detected  : {total_persons}

  NEXT: Run the unit tests:
      python -m pytest tests\\test_detection.py -v
""")


if __name__ == "__main__":
    main()
