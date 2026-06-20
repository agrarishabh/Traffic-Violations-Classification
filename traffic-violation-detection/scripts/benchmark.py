"""
scripts/benchmark.py
=====================
Pipeline Speed Benchmark — Phase 9

Measures average time for each stage of the pipeline:
  1. Image preprocessing
  2. Vehicle/person detection (YOLOv8)
  3. Violation checking (all 7 types)
  4. License plate OCR
  5. Evidence generation
  6. Total end-to-end

Reports: avg ms / image, FPS, percentile breakdown

HOW TO RUN:
    python scripts\\benchmark.py
"""

import sys, time, cv2, numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

N_WARMUP = 2   # warm-up runs (not counted)
N_RUNS   = 10  # benchmark runs per image


def make_test_image(size=640) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:size//2] = (170, 155, 110)
    img[size//2:] = (72, 73, 80)
    cv2.rectangle(img, (100, 320), (400, 500), (30, 70, 180), -1)
    cv2.rectangle(img, (150, 265), (350, 325), (30, 70, 180), -1)
    cv2.circle(img, (155, 508), 28, (15,15,15), -1)
    cv2.circle(img, (345, 508), 28, (15,15,15), -1)
    cv2.circle(img, (260, 250), 22, (100,150,200), -1)
    cv2.rectangle(img, (247,268),(273,330),(100,150,200),-1)
    cv2.line(img, (0, 400), (640, 400), (255, 255, 255), 6)
    return img


def timer(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - t0) * 1000


def fmt_row(label, times_ms):
    avg = np.mean(times_ms)
    p50 = np.percentile(times_ms, 50)
    p95 = np.percentile(times_ms, 95)
    fps = 1000.0 / avg if avg > 0 else 0
    return (f"  {label:<30} {avg:>8.1f}ms  "
            f"p50:{p50:>7.1f}ms  p95:{p95:>7.1f}ms  "
            f"{fps:>6.1f} FPS")


def main():
    print("\n" + "="*65)
    print("  TRAFFIC VIOLATION DETECTION — PIPELINE BENCHMARK")
    print("="*65)
    print(f"  Warm-up runs: {N_WARMUP}  |  Benchmark runs: {N_RUNS}")
    print("="*65)

    # ── Load models ───────────────────────────────────────────
    print("\n[Step 1/3] Loading models ...")
    from core.preprocessor       import ImagePreprocessor
    from core.violation_detector import ViolationDetector
    from core.ocr                import LicensePlateOCR
    from core.evidence_generator import EvidenceGenerator

    preprocessor = ImagePreprocessor()
    vd  = ViolationDetector()
    vd.load_models()
    ocr = LicensePlateOCR()
    ocr.load_models()
    gen = EvidenceGenerator(save_to_db=False)

    test_image = make_test_image(640)
    print("  Models ready\n")

    # ── Warm-up ───────────────────────────────────────────────
    print("[Step 2/3] Warming up ...")
    for _ in range(N_WARMUP):
        vd.detect_all(test_image)
    print(f"  {N_WARMUP} warm-up run(s) done\n")

    # ── Benchmark ─────────────────────────────────────────────
    print(f"[Step 3/3] Running {N_RUNS} benchmark iterations ...\n")

    times: dict = defaultdict(list)

    for i in range(N_RUNS):
        img = test_image.copy()
        sys.stdout.write(f"\r  Run {i+1}/{N_RUNS} ...")
        sys.stdout.flush()

        # Stage 1: Preprocessing
        processed, _ = timer(preprocessor.process, img)
        proc_img, report = processed
        times["1_preprocess"].append(report.processing_ms)

        # Stage 2+3: Detection + violation checking (combined in detect_all)
        viol_result, ms_full = timer(vd.detect_all, img)
        times["2_detect_all"].append(ms_full)

        # Isolate just vehicle detection
        if viol_result.detection_result:
            times["2a_vehicle_detect"].append(
                viol_result.detection_result.inference_ms)

        # Violation checking only (pipeline ms minus detection)
        det_ms  = viol_result.detection_result.inference_ms \
                  if viol_result.detection_result else 0
        viol_ms = max(0, ms_full - det_ms)
        times["2b_violation_check"].append(viol_ms)

        # Stage 3: OCR
        _, ms_ocr = timer(ocr.extract, img)
        times["3_ocr"].append(ms_ocr)

        # Stage 4: Evidence generation
        _, ms_ev = timer(
            gen.generate,
            image            = img,
            image_path       = "bench.jpg",
            detection_result = viol_result.detection_result,
            violation_result = viol_result,
            plate_results    = [],
        )
        times["4_evidence"].append(ms_ev)

        # Total (preprocess + detect_all + ocr + evidence)
        total = (report.processing_ms + ms_full + ms_ocr + ms_ev)
        times["5_total"].append(total)

    print("\n")

    # ── Results table ─────────────────────────────────────────
    labels = {
        "1_preprocess":       "1. Image Preprocessing",
        "2a_vehicle_detect":  "2a. YOLOv8 Vehicle Detection",
        "2b_violation_check": "2b. Violation Rule Checking",
        "2_detect_all":       "2.  Detection + Violations",
        "3_ocr":              "3. License Plate OCR",
        "4_evidence":         "4. Evidence Generation",
        "5_total":            "TOTAL Pipeline",
    }

    print(f"  {'Stage':<30} {'Avg':>10}  {'p50':>11}  {'p95':>11}  {'FPS':>8}")
    print("  " + "-"*75)

    for key, label in labels.items():
        if key in times and times[key]:
            print(fmt_row(label, times[key]))
        if key == "2_detect_all":
            print("  " + "-"*75)

    avg_total = np.mean(times["5_total"])
    fps_total = 1000.0 / avg_total if avg_total > 0 else 0

    print("\n" + "="*65)
    print(f"  END-TO-END THROUGHPUT: {fps_total:.2f} FPS "
          f"({avg_total:.0f}ms / image)")
    print("="*65)

    # ── Save to file ──────────────────────────────────────────
    out_dir = PROJECT_ROOT / "samples" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_path = out_dir / "benchmark_results.txt"
    with open(bench_path, "w", encoding="utf-8") as f:
        f.write("PIPELINE BENCHMARK RESULTS\n" + "="*65 + "\n\n")
        f.write(f"Runs per stage : {N_RUNS}\n")
        f.write(f"Image size     : 640x640\n")
        f.write(f"GPU available  : {_check_gpu()}\n\n")
        for key, label in labels.items():
            if key in times and times[key]:
                avg = np.mean(times[key])
                fps = 1000.0 / avg if avg > 0 else 0
                f.write(f"{label:<35} avg={avg:.1f}ms  fps={fps:.1f}\n")
        f.write(f"\nTotal FPS: {fps_total:.2f}\n")
    print(f"\n  Benchmark saved: {bench_path}")
    print("""
  INTERPRETATION:
  - YOLOv8n (nano) on CPU: ~25–80ms detection
  - OCR first run is slow; subsequent runs use cache
  - GPU would reduce total time by 5-10x
  - For real-time video: use GPU or YOLOv8n + skip OCR every N frames
""")


def _check_gpu() -> str:
    try:
        import torch
        return "Yes (CUDA)" if torch.cuda.is_available() else "No (CPU)"
    except ImportError:
        return "Unknown"


if __name__ == "__main__":
    main()
