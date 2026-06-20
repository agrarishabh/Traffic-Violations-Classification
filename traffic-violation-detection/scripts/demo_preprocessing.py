"""
scripts/demo_preprocessing.py
==============================
Visual Demo of the Image Preprocessing Pipeline.

Creates 4 test images (dark, blurry, noisy, hazy), runs the
preprocessor on each one, saves side-by-side before/after comparisons,
and prints a detailed quality report for each image.

HOW TO RUN:
    python scripts\\demo_preprocessing.py

OUTPUT:
    samples/preprocessed/
        demo_dark_comparison.jpg
        demo_blurry_comparison.jpg
        demo_noisy_comparison.jpg
        demo_hazy_comparison.jpg
        demo_normal_comparison.jpg
        demo_report.txt
"""

import sys
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.preprocessor import ImagePreprocessor

OUTPUT_DIR = PROJECT_ROOT / "samples" / "preprocessed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# SYNTHETIC IMAGE GENERATORS
# ══════════════════════════════════════════════════════════════

def make_base_traffic_scene(h=480, w=640) -> np.ndarray:
    """Generate a realistic-looking synthetic traffic scene."""
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Sky gradient
    for y in range(int(h * 0.35)):
        blue = int(255 * (1 - y / (h * 0.35)))
        img[y, :] = (blue, int(blue * 0.7), int(blue * 0.4))

    # Road surface
    img[int(h * 0.35):, :] = (60, 62, 65)

    # Lane markings
    road_top = int(h * 0.45)
    for x in range(50, w, 100):
        cv2.rectangle(img, (x, road_top), (x + 50, road_top + 8),
                      (220, 220, 220), -1)

    # Car 1 (blue sedan)
    cv2.rectangle(img, (80, 290), (260, 390), (160, 60, 20), -1)   # body
    cv2.rectangle(img, (110, 250), (230, 295), (160, 60, 20), -1)  # roof
    cv2.circle(img, (115, 395), 22, (15, 15, 15), -1)              # wheel L
    cv2.circle(img, (225, 395), 22, (15, 15, 15), -1)              # wheel R
    cv2.rectangle(img, (90, 260), (145, 292), (200, 230, 255), -1) # windshield

    # Car 2 (red car)
    cv2.rectangle(img, (360, 300), (560, 400), (30, 30, 180), -1)
    cv2.rectangle(img, (390, 265), (530, 305), (30, 30, 180), -1)
    cv2.circle(img, (380, 405), 22, (15, 15, 15), -1)
    cv2.circle(img, (540, 405), 22, (15, 15, 15), -1)
    cv2.rectangle(img, (370, 270), (425, 302), (200, 230, 255), -1)

    # Motorcycle with rider
    cv2.rectangle(img, (280, 330), (340, 390), (80, 80, 80), -1)
    cv2.circle(img, (284, 398), 16, (15, 15, 15), -1)
    cv2.circle(img, (336, 398), 16, (15, 15, 15), -1)
    cv2.circle(img, (310, 305), 18, (210, 185, 145), -1)          # rider head
    cv2.rectangle(img, (298, 318), (322, 355), (0, 80, 180), -1)  # torso

    # Traffic light pole
    cv2.rectangle(img, (590, 200), (608, 420), (40, 40, 40), -1)
    cv2.rectangle(img, (575, 180), (625, 260), (20, 20, 20), -1)
    cv2.circle(img, (600, 197), 11, (0, 0, 230), -1)   # red
    cv2.circle(img, (600, 217), 11, (30, 30, 30), -1)  # yellow (off)
    cv2.circle(img, (600, 237), 11, (30, 30, 30), -1)  # green (off)

    # Stop line
    cv2.line(img, (0, 430), (w, 430), (255, 255, 255), 4)

    return img


def apply_dark(img: np.ndarray) -> np.ndarray:
    return (img.astype(np.float32) * 0.18).clip(0, 255).astype(np.uint8)


def apply_blur(img: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(img, (25, 25), sigmaX=8.0)


def apply_noise(img: np.ndarray) -> np.ndarray:
    noise = np.random.normal(0, 38, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_haze(img: np.ndarray) -> np.ndarray:
    return ((img.astype(np.float32) / 255.0) * 50 + 170).clip(0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════
# SIDE-BY-SIDE COMPARISON BUILDER
# ══════════════════════════════════════════════════════════════

def make_comparison(original: np.ndarray,
                    processed: np.ndarray,
                    title: str,
                    report_text: str) -> np.ndarray:
    """
    Builds a side-by-side before/after image with:
      LEFT  : original (degraded) image with "BEFORE" label
      RIGHT : processed image with "AFTER" label
      BOTTOM: report summary text panel
    """
    target_h, target_w = 400, 640

    # Resize both to same display size (not letterbox — just for display)
    left  = cv2.resize(original,  (target_w, target_h))
    right = cv2.resize(processed, (target_w, target_h))

    # Draw "BEFORE" / "AFTER" labels
    for panel, label, color in [(left, "BEFORE", (0, 0, 220)),
                                 (right, "AFTER",  (0, 180, 0))]:
        cv2.rectangle(panel, (0, 0), (120, 40), (0, 0, 0), -1)
        cv2.putText(panel, label, (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Combine left and right side by side
    combined = np.hstack([left, right])

    # Add a separator line between panels
    cv2.line(combined, (target_w, 0), (target_w, target_h), (255, 255, 255), 2)

    # Title bar (top)
    title_bar = np.zeros((50, target_w * 2, 3), dtype=np.uint8)
    title_bar[:] = (30, 30, 30)
    cv2.putText(title_bar, title, (10, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Report panel (bottom) — white text on dark background
    lines = report_text.split("\n")
    line_h = 26
    panel_h = max(80, len(lines) * line_h + 20)
    report_panel = np.zeros((panel_h, target_w * 2, 3), dtype=np.uint8)
    report_panel[:] = (20, 20, 35)

    for i, line in enumerate(lines):
        y = 22 + i * line_h
        # Colour key metrics
        color = (180, 255, 180) if "Applied" in line else (200, 200, 200)
        cv2.putText(report_panel, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)

    return np.vstack([title_bar, combined, report_panel])


# ══════════════════════════════════════════════════════════════
# MAIN DEMO
# ══════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 58)
    print("  IMAGE PREPROCESSING PIPELINE — VISUAL DEMO")
    print("=" * 58)
    print(f"  Output folder: {OUTPUT_DIR}\n")

    preprocessor = ImagePreprocessor()
    base = make_base_traffic_scene()

    # Define test scenarios
    scenarios = [
        ("Normal (Good Quality)",  base,                   "demo_normal"),
        ("Dark / Night Scene",     apply_dark(base),       "demo_dark"),
        ("Motion Blur",            apply_blur(base),       "demo_blurry"),
        ("Heavy Noise / Rain",     apply_noise(base),      "demo_noisy"),
        ("Haze / Fog",             apply_haze(base),       "demo_hazy"),
    ]

    all_reports = []

    for title, degraded_img, filename in scenarios:
        print(f"  Processing: {title} ...")

        # Run preprocessing
        processed, report = preprocessor.process(degraded_img.copy())

        # Build report text for the image panel
        r = report.to_dict()
        report_lines = [
            f"  Brightness: {r['brightness_before']:.1f} -> {r['brightness_after']:.1f}   "
            f"Blur score: {r['blur_score_before']:.1f} -> {r['blur_score_after']:.1f}   "
            f"Time: {r['processing_ms']:.1f}ms",
            f"  Issues detected: "
            + ("Low-light " if r['was_low_light'] else "")
            + ("Blurry "    if r['was_blurry']    else "")
            + ("Noisy "     if r['was_noisy']     else "")
            + ("Shadows "   if r['had_shadows']   else "")
            + ("Haze "      if r['was_hazy']      else "")
            + ("None"       if not any([r['was_low_light'], r['was_blurry'],
                                        r['was_noisy'], r['had_shadows'], r['was_hazy']]) else ""),
            f"  Applied: {report.summary()}",
        ]
        report_text = "\n".join(report_lines)

        # Resize processed back to 640×480 for comparison (undo letterbox for display)
        processed_display = cv2.resize(processed, (640, 480))

        # Build and save comparison image
        comparison = make_comparison(degraded_img, processed_display,
                                     title, report_text)
        out_path = OUTPUT_DIR / f"{filename}_comparison.jpg"
        cv2.imwrite(str(out_path), comparison)

        print(f"    Saved: {out_path.name}")
        print(f"    Issues: "
              f"{'low-light ' if report.was_low_light else ''}"
              f"{'blurry '    if report.was_blurry    else ''}"
              f"{'noisy '     if report.was_noisy     else ''}"
              f"{'shadows '   if report.had_shadows   else ''}"
              f"{'hazy '      if report.was_hazy      else ''}"
              f"{'none'       if not report.steps_applied[:-1] else ''}")
        print(f"    Steps:  {report.summary()}")
        print(f"    Time:   {report.processing_ms:.1f}ms\n")

        all_reports.append((title, report.to_dict()))

    # ── Save text report ─────────────────────────────────────
    report_path = OUTPUT_DIR / "demo_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("IMAGE PREPROCESSING PIPELINE - DEMO REPORT\n")
        f.write("=" * 60 + "\n\n")
        for title, r in all_reports:
            f.write(f"Scenario: {title}\n")
            f.write(f"  Original shape:  {r['original_shape']}\n")
            f.write(f"  Final shape:     {r['final_shape']}\n")
            f.write(f"  Brightness:      {r['brightness_before']:.1f} -> {r['brightness_after']:.1f}\n")
            f.write(f"  Blur score:      {r['blur_score_before']:.1f} -> {r['blur_score_after']:.1f}\n")
            f.write(f"  Processing time: {r['processing_ms']:.1f}ms\n")
            f.write(f"  Steps applied:   {', '.join(r['steps_applied'])}\n")
            f.write("-" * 60 + "\n")

    print(f"  Text report saved: {report_path.name}")

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 58)
    print("  DEMO COMPLETE")
    print("=" * 58)
    print(f"""
  {len(scenarios)} comparison images saved to:
  {OUTPUT_DIR}

  HOW TO VIEW:
  Open any *_comparison.jpg file in File Explorer.
  Left panel = BEFORE (degraded), Right panel = AFTER (fixed).

  NEXT STEP:
  Run the unit tests to verify all methods work correctly:
      python -m pytest tests\\test_preprocessing.py -v
""")


if __name__ == "__main__":
    main()
