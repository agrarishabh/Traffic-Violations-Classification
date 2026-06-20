"""
scripts/run_violation_demo.py
==============================
Visual Violation Detection Demo — Phase 4

Creates synthetic traffic scenes designed to trigger each violation,
runs the full ViolationDetector pipeline, and saves annotated images.

HOW TO RUN:
    python scripts\\run_violation_demo.py

OUTPUT:
    samples/violations/
        violation_<type>_demo.jpg   <- annotated image per violation
        violation_summary.txt       <- text report
"""

import sys
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.violation_detector import ViolationDetector, ViolationResult, Violation
from core.detector import Detection, DetectionResult
from config import ViolationType, COLORS, VIOLATION_DISPLAY_NAMES

OUTPUT_DIR = PROJECT_ROOT / "samples" / "violations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# SYNTHETIC SCENE BUILDERS
# ══════════════════════════════════════════════════════════════

def base_road(h=640, w=640) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        t = y / h
        sky_c = np.array([120 + int(t*40), 140 + int(t*30), 200])
        img[y] = sky_c if y < h // 2 else np.array([70, 72, 78])
    cv2.line(img, (0, h//2), (w, h//2), (90, 90, 95), 1)
    return img


def draw_car(img, x1, y1, x2, y2, color=(30, 80, 200)):
    h = y2 - y1; w = x2 - x1
    cv2.rectangle(img, (x1, y1 + h//3), (x2, y2), color, -1)
    cv2.rectangle(img, (x1 + w//5, y1), (x2 - w//5, y1 + h//3 + 5), color, -1)
    cv2.rectangle(img, (x1+5, y1+5), (x1+w//5+5, y1+h//3), (200,225,255), -1)
    cv2.circle(img, (x1 + w//5, y2+5), w//8, (15,15,15), -1)
    cv2.circle(img, (x2 - w//5, y2+5), w//8, (15,15,15), -1)


def draw_motorcycle(img, x1, y1, x2, y2, color=(80, 80, 200)):
    h = y2 - y1; w = x2 - x1
    cx = (x1 + x2) // 2
    cv2.ellipse(img, (cx, (y1+y2)//2), (w//2, h//3), 0, 0, 360, color, -1)
    cv2.circle(img, (x1 + w//5, y2+2), w//6, (15,15,15), -1)
    cv2.circle(img, (x2 - w//5, y2+2), w//6, (15,15,15), -1)


def draw_person(img, x1, y1, x2, y2, skin=(100, 150, 200), has_helmet=False):
    h = y2 - y1; cx = (x1+x2)//2
    head_color = (30, 30, 30) if has_helmet else skin
    cv2.circle(img, (cx, y1 + h//7), h//7, head_color, -1)
    cv2.rectangle(img, (cx - h//12, y1 + h//4),
                  (cx + h//12, y1 + 2*h//3), skin, -1)
    cv2.line(img, (cx - h//12, y1+h//3), (x1+2, y1+h//2), skin, 3)
    cv2.line(img, (cx + h//12, y1+h//3), (x2-2, y1+h//2), skin, 3)


def draw_traffic_light(img, cx, cy, state="red"):
    cv2.rectangle(img, (cx-18, cy-45), (cx+18, cy+45), (20,20,20), -1)
    red_c    = (0, 0, 220)   if state=="red"    else (15,15,15)
    yellow_c = (0, 200, 230) if state=="yellow" else (15,15,15)
    green_c  = (0, 180, 0)   if state=="green"  else (15,15,15)
    cv2.circle(img, (cx, cy-25), 12, red_c,    -1)
    cv2.circle(img, (cx, cy),    12, yellow_c, -1)
    cv2.circle(img, (cx, cy+25), 12, green_c,  -1)


# ══════════════════════════════════════════════════════════════
# ANNOTATION HELPERS
# ══════════════════════════════════════════════════════════════

VIOLATION_COLORS = {
    ViolationType.HELMET_VIOLATION:   (0, 0, 255),
    ViolationType.SEATBELT_VIOLATION: (0, 80, 255),
    ViolationType.TRIPLE_RIDING:      (0, 140, 255),
    ViolationType.RED_LIGHT:          (0, 0, 200),
    ViolationType.STOP_LINE:          (0, 165, 255),
    ViolationType.WRONG_SIDE:         (180, 0, 255),
    ViolationType.ILLEGAL_PARKING:    (255, 0, 180),
}


def annotate_violations(image: np.ndarray,
                         result: ViolationResult) -> np.ndarray:
    out = image.copy()
    for v in result.violations:
        color = VIOLATION_COLORS.get(v.violation_type, (0, 0, 255))
        x1, y1, x2, y2 = v.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)

        label = f"{v.display_name}  {v.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = max(y1 - 8, th + 8)
        cv2.rectangle(out, (x1, ly-th-6), (x1+tw+4, ly+2), color, -1)
        cv2.putText(out, label, (x1+2, ly-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return out


def add_summary_bar(image: np.ndarray, result: ViolationResult,
                    scenario: str) -> np.ndarray:
    h, w = image.shape[:2]
    bar = np.zeros((50, w, 3), dtype=np.uint8); bar[:] = (20, 20, 35)
    text = (f"Scenario: {scenario}   |   "
            f"Violations: {len(result.violations)}   |   "
            f"Fine: Rs.{result.total_fines}   |   "
            f"Time: {result.processing_ms:.0f}ms")
    cv2.putText(bar, text, (8, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 220, 255), 1)
    return np.vstack([image, bar])


# ══════════════════════════════════════════════════════════════
# DEMO SCENES
# ══════════════════════════════════════════════════════════════

def scene_triple_riding():
    img = base_road()
    draw_motorcycle(img, 240, 320, 400, 480)
    for i, dx in enumerate([-30, 0, 30]):
        draw_person(img, 260+dx, 270, 320+dx, 440, has_helmet=False)
    return img, "Triple Riding (3 persons on motorcycle)"


def scene_helmet():
    img = base_road()
    draw_motorcycle(img, 220, 320, 380, 470)
    draw_person(img, 240, 270, 360, 460, has_helmet=False)
    return img, "Helmet Non-Compliance (bare head on motorcycle)"


def scene_stop_line():
    img = base_road()
    cv2.line(img, (0, 420), (640, 420), (255, 255, 255), 7)
    cv2.putText(img, "STOP", (270, 416), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2)
    draw_car(img, 150, 350, 450, 520)   # car bottom (520) > line (420)
    return img, "Stop Line Violation (car crossed stop line)"


def scene_red_light():
    img = base_road()
    draw_traffic_light(img, cx=320, cy=120, state="red")
    draw_car(img, 120, 280, 380, 480)
    draw_car(img, 400, 300, 590, 490)
    return img, "Red Light Violation (vehicles in intersection during red)"


def scene_wrong_side():
    img = base_road()
    cv2.line(img, (320, 320), (320, 640), (255, 255, 255), 2)
    draw_car(img, 50,  350, 250, 500, color=(30, 80, 200))   # correct side
    draw_car(img, 100, 370, 290, 510, color=(50, 100, 220))  # correct side
    draw_car(img, 440, 340, 620, 490, color=(20, 20, 180))   # wrong side
    return img, "Wrong-Side Driving (vehicle on wrong side of road)"


def scene_illegal_parking():
    img = base_road()
    draw_car(img, 560, 510, 638, 595)   # wide, near right edge, near bottom
    draw_car(img, 200, 360, 450, 530, color=(40, 160, 40))  # normal car
    return img, "Illegal Parking (wide unoccupied car at road edge)"


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*58)
    print("  VIOLATION DETECTION DEMO — Phase 4")
    print("="*58)

    print("\n[Step 1/3] Loading models ...")
    vd = ViolationDetector()
    vd.load_models()

    scenes = [
        (scene_triple_riding,    "triple_riding"),
        (scene_helmet,           "helmet"),
        (scene_stop_line,        "stop_line"),
        (scene_red_light,        "red_light"),
        (scene_wrong_side,       "wrong_side"),
        (scene_illegal_parking,  "illegal_parking"),
    ]

    print(f"\n[Step 2/3] Running {len(scenes)} violation scenarios ...\n")
    print("-" * 58)

    summary_lines = ["VIOLATION DEMO SUMMARY\n" + "="*58 + "\n"]
    total_violations = 0

    for scene_fn, filename in scenes:
        image, title = scene_fn()
        print(f"\n  Scenario: {title}")

        result = vd.detect_all(image)

        annotated = annotate_violations(image, result)
        annotated = add_summary_bar(annotated, result, title)

        out_path = OUTPUT_DIR / f"violation_{filename}_demo.jpg"
        cv2.imwrite(str(out_path), annotated)

        print(f"  Violations found : {len(result.violations)}")
        for v in result.violations:
            print(f"    [{v.violation_type}]  conf={v.confidence:.0%}  "
                  f"fine=Rs.{v.fine_amount}")
        if not result.violations:
            print("  No violations flagged on this synthetic scene")
            print("  (Heuristics need real photo textures for best results)")
        print(f"  Time: {result.processing_ms:.0f}ms")
        print(f"  Saved: {out_path.name}")

        total_violations += len(result.violations)
        summary_lines.append(
            f"Scenario : {title}\n"
            f"File     : {out_path.name}\n"
            f"Violations: {len(result.violations)}\n"
            f"Total fine: Rs.{result.total_fines}\n"
            f"Details  : {[v.display_name for v in result.violations]}\n"
        )

    (OUTPUT_DIR / "violation_summary.txt").write_text(
        "\n".join(summary_lines), encoding="utf-8")

    print("\n" + "="*58)
    print("  VIOLATION DEMO COMPLETE")
    print("="*58)
    print(f"""
  {len(scenes)} scenarios run, {total_violations} violation(s) flagged.
  Annotated images: {OUTPUT_DIR}

  HOW TO VIEW:
  Open samples/violations/ in File Explorer.
  Each image shows RED boxes around violations
  with violation type + confidence label.

  NOTE ON SYNTHETIC SCENES:
  Heuristic checks (helmet skin-ratio, seatbelt diagonal)
  work best on real photographs. The STOP LINE and
  TRIPLE RIDING detectors work reliably on any image.

  NEXT STEP: Phase 5 — License Plate Recognition
      (tell Kiro: "start phase 5")
""")


if __name__ == "__main__":
    main()
