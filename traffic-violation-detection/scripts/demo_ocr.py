"""
scripts/demo_ocr.py
====================
License Plate Recognition Demo — Phase 5

Creates synthetic vehicle images with realistic plate rectangles,
runs the full OCR pipeline, and saves annotated outputs.

HOW TO RUN:
    python scripts\\demo_ocr.py

OUTPUT:
    samples/ocr/
        plate_<name>_detected.jpg   <- annotated image
        ocr_summary.txt             <- extracted plate text report
"""

import sys
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.ocr import LicensePlateOCR, PlateResult

OUTPUT_DIR = PROJECT_ROOT / "samples" / "ocr"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# SCENE BUILDERS
# ══════════════════════════════════════════════════════════════

def draw_vehicle(img, x1, y1, x2, y2, color=(30, 80, 200)):
    """Draw a simplified car shape."""
    h = y2 - y1; w = x2 - x1
    cv2.rectangle(img, (x1, y1 + h//3), (x2, y2),    color, -1)  # body
    cv2.rectangle(img, (x1+w//6, y1),   (x2-w//6, y1+h//3+5), color, -1)  # roof
    cv2.rectangle(img, (x1+5, y1+h//3-20), (x1+w//4, y1+h//3+8),
                  (200, 225, 255), -1)  # windshield


def draw_plate_rect(img, x1, y1, x2, y2,
                    text: str = "MH12AB1234") -> None:
    """Draw a white plate rectangle with text — mimics real Indian plate."""
    cv2.rectangle(img, (x1, y1), (x2, y2), (245, 245, 240), -1)  # off-white
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)          # border

    plate_h = y2 - y1
    scale   = plate_h / 52.0
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    tx = x1 + max(2, (x2-x1-tw)//2)
    ty = y2 - max(4, (y2-y1-th)//2)
    cv2.putText(img, text, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (10, 10, 10), 2)


def make_scene(plate_text: str = "MH12AB1234",
               light: bool = True) -> np.ndarray:
    """640x480 road scene with one car and a readable plate."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:240] = (180, 160, 120) if light else (30, 30, 40)   # sky
    img[240:] = (75, 76, 82)                                  # road

    # Road markings
    for x in range(0, 640, 80):
        cv2.rectangle(img, (x, 268), (x+40, 276), (200, 200, 200), -1)

    # Car body
    draw_vehicle(img, 160, 270, 490, 430, color=(30, 70, 180))

    # Plate: 200×46px → aspect ≈ 4.35 (valid range 1.8–7.0)
    px1, py1, px2, py2 = 220, 395, 420, 441
    draw_plate_rect(img, px1, py1, px2, py2, plate_text)

    return img


def make_dark_scene(plate_text: str = "DL01CD5678") -> np.ndarray:
    """Night scene — tests preprocessing handles low light."""
    img = make_scene(plate_text, light=False)
    return (img.astype(np.float32) * 0.35).clip(0, 255).astype(np.uint8)


def make_multi_vehicle_scene() -> np.ndarray:
    """Scene with two vehicles — tests multiple plate detection."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:240] = (160, 140, 100)
    img[240:] = (72, 73, 78)

    # Vehicle 1
    draw_vehicle(img, 20, 270, 300, 430, color=(40, 90, 200))
    draw_plate_rect(img, 60, 395, 256, 435, "MH12AB1234")

    # Vehicle 2
    draw_vehicle(img, 330, 280, 620, 440, color=(160, 40, 40))
    draw_plate_rect(img, 370, 400, 580, 440, "DL01CD5678")

    return img


# ══════════════════════════════════════════════════════════════
# ANNOTATION
# ══════════════════════════════════════════════════════════════

def annotate_plates(image: np.ndarray,
                    plates: list) -> np.ndarray:
    """Draw bounding boxes and extracted text on the image."""
    out = image.copy()
    for plate in plates:
        x1, y1, x2, y2 = plate.bbox
        # Box color: green if valid format, orange if uncertain
        color = (0, 200, 0) if plate.is_valid_format else (0, 165, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label: plate text + confidence
        label = f"{plate.plate_text_clean}  {plate.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                       0.55, 1)
        ly = max(y1 - 8, th + 8)
        cv2.rectangle(out, (x1, ly-th-6), (x1+tw+4, ly+2), color, -1)
        cv2.putText(out, label, (x1+2, ly-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

        # Mark if valid Indian format
        if plate.is_valid_format:
            cv2.putText(out, "VALID", (x1, y2+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
    return out


def add_result_panel(image: np.ndarray, plates: list,
                     title: str) -> np.ndarray:
    """Add dark panel at bottom summarising results."""
    h, w = image.shape[:2]
    panel = np.zeros((55, w, 3), dtype=np.uint8)
    panel[:] = (20, 20, 35)

    if plates:
        texts = ", ".join(
            [f"{p.plate_text_clean}({p.confidence:.0%})" for p in plates])
        line1 = f"  {title}"
        line2 = f"  Plates: {texts}"
    else:
        line1 = f"  {title}"
        line2 = "  No plates detected (normal for synthetic images)"

    cv2.putText(panel, line1, (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 220, 255), 1)
    cv2.putText(panel, line2, (6, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 200, 180), 1)
    return np.vstack([image, panel])


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*58)
    print("  LICENSE PLATE RECOGNITION DEMO — Phase 5")
    print("="*58)

    print("\n[Step 1/3] Loading EasyOCR ...")
    print("  NOTE: First run downloads ~100MB of language models.")
    print("  This takes 1-3 minutes. Subsequent runs are instant.\n")
    ocr = LicensePlateOCR(use_gpu=False)
    ocr.load_models()

    scenes = [
        ("MH12AB1234",     make_scene("MH12AB1234"),          "plate_normal"),
        ("DL01CD5678 dark",make_dark_scene("DL01CD5678"),     "plate_dark"),
        ("Multi-vehicle",  make_multi_vehicle_scene(),         "plate_multi"),
    ]

    # Also try real images from samples/ if available
    from config import SAMPLES_DIR
    real_imgs = list(SAMPLES_DIR.glob("*.jpg")) + list(SAMPLES_DIR.glob("*.png"))
    real_imgs = [p for p in real_imgs
                 if "detected" not in str(p) and "demo" not in str(p)]
    for p in real_imgs[:3]:
        img = cv2.imread(str(p))
        if img is not None:
            scenes.append((p.stem, img, f"plate_real_{p.stem}"))

    print(f"[Step 2/3] Processing {len(scenes)} image(s) ...\n")
    print("-"*58)

    summary_lines = ["LICENSE PLATE OCR SUMMARY\n" + "="*58 + "\n"]
    total_plates = 0

    for title, image, filename in scenes:
        print(f"\n  Image: {title}")
        plates = ocr.extract(image)

        annotated = annotate_plates(image, plates)
        annotated = add_result_panel(annotated, plates, title)

        out_path = OUTPUT_DIR / f"{filename}_detected.jpg"
        cv2.imwrite(str(out_path), annotated)

        if plates:
            for p in plates:
                print(f"  Plate text  : {p.plate_text_clean}")
                print(f"  Confidence  : {p.confidence:.0%}")
                print(f"  Valid format: {p.is_valid_format}")
                print(f"  Method      : {p.detection_method}")
                total_plates += 1
        else:
            print("  No plate detected above threshold")
            print("  (Contour detection works better on real traffic photos)")
        print(f"  Saved: {out_path.name}")

        summary_lines.append(
            f"Image   : {title}\n"
            f"Plates  : {[p.plate_text_clean for p in plates]}\n"
            f"Valid   : {[p.is_valid_format  for p in plates]}\n"
            f"Conf    : {[round(p.confidence,2) for p in plates]}\n"
        )

    (OUTPUT_DIR / "ocr_summary.txt").write_text(
        "\n".join(summary_lines), encoding="utf-8")

    print("\n" + "="*58)
    print("  OCR DEMO COMPLETE")
    print("="*58)
    print(f"""
  Total plates found : {total_plates}
  Output folder      : {OUTPUT_DIR}

  GREEN box = valid Indian plate format (XX 00 XX 0000)
  ORANGE box = text extracted but format uncertain

  HOW TO IMPROVE OCR ACCURACY:
  1. Download the Roboflow license plate dataset:
         python scripts\\download_datasets.py
  2. Real traffic photographs give much better results
     than synthetic images because EasyOCR was trained
     on real-world text.

  NEXT STEP: Phase 6 - Evidence Generation
      (tell Kiro: "start phase 6")
""")


if __name__ == "__main__":
    main()
