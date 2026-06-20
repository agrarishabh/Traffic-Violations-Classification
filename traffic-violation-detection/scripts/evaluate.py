"""
scripts/evaluate.py
====================
Performance Evaluation — Phase 9

Evaluates the violation detection LOGIC directly by injecting synthetic
DetectionResult objects with known positions, bypassing YOLOv8 (which
needs real photos). This mirrors the approach used in unit tests and
gives meaningful precision/recall numbers for the rule-based checkers.

Metrics computed:
  Per-class : Precision, Recall, F1-score
  Overall   : mAP, weighted F1, Accuracy
  Speed     : avg inference time per violation check

HOW TO RUN:
    python scripts\\evaluate.py

OUTPUT:
    Console table + samples/evaluation/eval_report.txt
"""

import sys
import cv2
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "samples" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# HELPERS — build mock objects without running YOLOv8
# ══════════════════════════════════════════════════════════════

def det(class_id, bbox, conf=0.88):
    from config import COCO_CLASSES
    from core.detector import Detection
    return Detection(class_id=class_id,
                     class_name=COCO_CLASSES.get(class_id, "unknown"),
                     confidence=conf, bbox=bbox)


def det_result(vehicles=None, persons=None, traffic_lights=None, shape=(640,640,3)):
    from core.detector import DetectionResult
    r = DetectionResult(image_path="test.jpg")
    r.vehicles       = vehicles       or []
    r.persons        = persons        or []
    r.traffic_lights = traffic_lights or []
    r.detections     = r.vehicles + r.persons + r.traffic_lights
    r.image_shape    = shape
    return r


def blank_road(h=640, w=640):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:h//2] = (170, 155, 110)
    img[h//2:] = (72, 73, 80)
    return img


def road_with_stop_line(y=400):
    img = blank_road()
    cv2.line(img, (0, y), (640, y), (255, 255, 255), 7)
    cv2.putText(img, "STOP", (275, y-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def road_with_red_light(cx=320, cy=100):
    img = blank_road()
    cv2.rectangle(img, (cx-18, cy-45), (cx+18, cy+45), (20,20,20), -1)
    cv2.circle(img, (cx, cy-25), 12, (0,0,220), -1)   # red ON
    cv2.circle(img, (cx, cy),    12, (15,15,15), -1)
    cv2.circle(img, (cx, cy+25), 12, (15,15,15), -1)
    return img


# ══════════════════════════════════════════════════════════════
# TEST CASE DEFINITION
# ══════════════════════════════════════════════════════════════

@dataclass
class EvalCase:
    name:     str
    expected: List[str]   # expected violation_type strings
    image_fn: object      # () -> np.ndarray
    dr_fn:    object      # () -> DetectionResult


# ── TRIPLE RIDING ──────────────────────────────────────────────
def dr_triple():
    moto = det(3, [50,50,550,550])
    p1 = det(0,[100,80,400,500]); p2 = det(0,[150,90,420,490])
    p3 = det(0,[200,100,440,480])
    return det_result(vehicles=[moto], persons=[p1,p2,p3])

def dr_two_riders():
    moto = det(3, [50,50,550,550])
    p1 = det(0,[100,80,400,500]); p2 = det(0,[200,100,440,480])
    return det_result(vehicles=[moto], persons=[p1,p2])

def dr_four_riders():
    moto = det(3, [0,0,640,640])
    riders = [det(0,[50+i*30,50,200+i*30,580]) for i in range(4)]
    return det_result(vehicles=[moto], persons=riders)

# ── STOP LINE ──────────────────────────────────────────────────
def dr_car_over_line():    # car bottom 500 > line 400
    return det_result(vehicles=[det(2,[120,310,430,500])])

def dr_car_behind_line():  # car bottom 380 < line 400
    return det_result(vehicles=[det(2,[120,200,430,380])])

def dr_no_car():
    return det_result()

# ── ILLEGAL PARKING ────────────────────────────────────────────
def dr_parked():           # wide, near bottom-right edge, no persons
    return det_result(vehicles=[det(2,[490,520,638,575])])

def dr_occupied_car():     # car with person inside → not parked
    car = det(2,[490,520,638,575])
    person = det(0,[495,525,630,570])
    return det_result(vehicles=[car], persons=[person])

def dr_normal_car():       # car in middle of road, occupied
    car = det(2,[150,300,420,450])
    person = det(0,[200,310,370,430])
    return det_result(vehicles=[car], persons=[person])

# ── HELMET ─────────────────────────────────────────────────────
def bare_head_image():
    img = blank_road()
    img[300:380, 240:360] = (100, 155, 205)   # skin patch
    return img

def helmeted_image():
    img = blank_road()
    img[300:380, 240:360] = (20, 20, 20)      # dark = helmet
    return img

def dr_rider():
    moto = det(3,[200,280,450,460])
    rider= det(0,[250,260,400,450])
    return det_result(vehicles=[moto], persons=[rider])


EVAL_CASES: List[EvalCase] = [
    # Triple riding
    EvalCase("triple_3_riders",       ["triple_riding"],  blank_road, dr_triple),
    EvalCase("triple_4_riders",       ["triple_riding"],  blank_road, dr_four_riders),
    EvalCase("triple_only_2_riders",  [],                 blank_road, dr_two_riders),

    # Stop line
    EvalCase("stop_car_over_line",    ["stop_line_violation"], road_with_stop_line, dr_car_over_line),
    EvalCase("stop_car_behind_line",  [],                      road_with_stop_line, dr_car_behind_line),
    EvalCase("stop_no_car",           [],                      road_with_stop_line, dr_no_car),

    # Illegal parking
    EvalCase("parking_wide_edge",     ["illegal_parking"], blank_road, dr_parked),
    EvalCase("parking_occupied",      [],                  blank_road, dr_occupied_car),
    EvalCase("parking_normal_car",    [],                  blank_road, dr_normal_car),

    # Helmet (heuristic)
    EvalCase("helmet_bare_head",      ["helmet_non_compliance"], bare_head_image, dr_rider),
    EvalCase("helmet_dark_head",      [],                        helmeted_image,  dr_rider),
]


# ══════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════

@dataclass
class ClassMetrics:
    tp: int = 0; fp: int = 0; fn: int = 0; tn: int = 0
    times_ms: List[float] = field(default_factory=list)

    @property
    def precision(self): return self.tp / max(1, self.tp + self.fp)
    @property
    def recall(self):    return self.tp / max(1, self.tp + self.fn)
    @property
    def f1(self):
        p, r = self.precision, self.recall
        return 2*p*r / max(1e-9, p+r)
    @property
    def accuracy(self):
        total = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / max(1, total)


def main():
    print("\n" + "="*62)
    print("  TRAFFIC VIOLATION DETECTION — PERFORMANCE EVALUATION")
    print("="*62)
    print(f"  Test cases: {len(EVAL_CASES)}")
    print("  Method: direct violation checker calls with mock detections")
    print("  (bypasses YOLOv8 — tests the rule-based logic accurately)")
    print("="*62 + "\n")

    print("  Loading ViolationDetector ...")
    from core.violation_detector import ViolationDetector
    vd = ViolationDetector()
    vd.load_models()

    per_class: Dict[str, ClassMetrics] = defaultdict(ClassMetrics)
    log = []

    all_violation_types = sorted({vt for tc in EVAL_CASES
                                   for vt in tc.expected})

    print(f"\n  {'Test Case':<30} {'Expected':<28} {'Detected':<28} Result")
    print("  " + "-"*92)

    for tc in EVAL_CASES:
        image = tc.image_fn()
        dr    = tc.dr_fn()
        t0    = time.perf_counter()

        # Run only the relevant violation checkers
        detected = []
        detected += vd.check_triple_riding(dr)
        detected += vd.check_stop_line(image, dr)
        detected += vd.check_illegal_parking(image, dr)
        detected += vd.check_helmet(image, dr)
        detected += vd.check_wrong_side(dr)

        ms = (time.perf_counter() - t0) * 1000
        detected_types = list({v.violation_type for v in detected})

        exp_set = set(tc.expected)
        det_set = set(detected_types)

        # Update per-class metrics
        for cls in all_violation_types:
            m = per_class[cls]
            m.times_ms.append(ms)
            if cls in exp_set and cls in det_set:   m.tp += 1
            elif cls in det_set:                     m.fp += 1
            elif cls in exp_set:                     m.fn += 1
            else:                                    m.tn += 1

        tp = len(exp_set & det_set)
        fp = len(det_set - exp_set)
        fn = len(exp_set - det_set)

        status = "PASS" if fp == 0 and fn == 0 else "FAIL"
        exp_str = (", ".join(tc.expected) or "none")[:26]
        det_str = (", ".join(detected_types) or "none")[:26]
        icon    = "OK" if status == "PASS" else "!!"
        print(f"  [{icon}] {tc.name:<28} {exp_str:<28} {det_str:<28} {status}")
        log.append({"name": tc.name, "expected": tc.expected,
                    "detected": detected_types, "tp": tp, "fp": fp,
                    "fn": fn, "ms": ms, "status": status})

    # ── Aggregate ─────────────────────────────────────────────
    total_tp = sum(r["tp"] for r in log)
    total_fp = sum(r["fp"] for r in log)
    total_fn = sum(r["fn"] for r in log)
    overall_p  = total_tp / max(1, total_tp + total_fp)
    overall_r  = total_tp / max(1, total_tp + total_fn)
    overall_f1 = 2*overall_p*overall_r / max(1e-9, overall_p+overall_r)
    mean_ap    = float(np.mean([m.precision for m in per_class.values()
                                if m.tp + m.fn > 0] or [0]))
    avg_ms     = float(np.mean([r["ms"] for r in log]))
    pass_count = sum(1 for r in log if r["status"] == "PASS")

    # ── Per-class table ───────────────────────────────────────
    print(f"\n  {'Violation Class':<35} {'Prec':>6} {'Rec':>6} {'F1':>6}"
          f" {'Acc':>6} {'TP':>3} {'FP':>3} {'FN':>3}")
    print("  " + "-"*72)
    for cls in sorted(per_class):
        m = per_class[cls]
        print(f"  {cls:<35} {m.precision:>6.3f} {m.recall:>6.3f}"
              f" {m.f1:>6.3f} {m.accuracy:>6.3f}"
              f" {m.tp:>3} {m.fp:>3} {m.fn:>3}")

    print("\n" + "="*62)
    print("  OVERALL RESULTS")
    print("="*62)
    print(f"  Test Cases     : {len(EVAL_CASES)} ({pass_count} PASS / {len(EVAL_CASES)-pass_count} FAIL)")
    print(f"  Precision      : {overall_p:.3f}  ({overall_p*100:.1f}%)")
    print(f"  Recall         : {overall_r:.3f}  ({overall_r*100:.1f}%)")
    print(f"  F1 Score       : {overall_f1:.3f}  ({overall_f1*100:.1f}%)")
    print(f"  mAP            : {mean_ap:.3f}  ({mean_ap*100:.1f}%)")
    print(f"  Avg Check Time : {avg_ms:.1f}ms / image")
    print("="*62)

    # ── Save report ───────────────────────────────────────────
    report_path = OUTPUT_DIR / "eval_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("TRAFFIC VIOLATION DETECTION SYSTEM\n")
        f.write("Performance Evaluation Report\n")
        f.write("=" * 62 + "\n\n")
        f.write(f"Methodology   : Direct violation checker with mock detections\n")
        f.write(f"Test Cases    : {len(EVAL_CASES)}\n")
        f.write(f"Pass / Fail   : {pass_count} / {len(EVAL_CASES)-pass_count}\n")
        f.write(f"Precision     : {overall_p:.3f}\n")
        f.write(f"Recall        : {overall_r:.3f}\n")
        f.write(f"F1 Score      : {overall_f1:.3f}\n")
        f.write(f"mAP           : {mean_ap:.3f}\n")
        f.write(f"Avg Check Time: {avg_ms:.1f}ms\n\n")
        f.write("Per-Class Results:\n" + "-"*62 + "\n")
        for cls in sorted(per_class):
            m = per_class[cls]
            f.write(f"  {cls:<38} P={m.precision:.3f}  "
                    f"R={m.recall:.3f}  F1={m.f1:.3f}  Acc={m.accuracy:.3f}\n")
        f.write("\nDetailed Test Cases:\n" + "-"*62 + "\n")
        for r in log:
            f.write(f"  [{r['status']}] {r['name']:<30} "
                    f"exp={r['expected']}  det={r['detected']}\n")

    print(f"\n  Report saved: {report_path}")
    print("""
  BENCHMARK CONTEXT (from benchmark.py):
  - YOLOv8n detection: ~88ms on CPU
  - Violation checks : <40ms
  - Full pipeline    : ~1.6s (OCR dominates)
  - GPU would give   : ~5-10x speedup on detection
""")


if __name__ == "__main__":
    main()
