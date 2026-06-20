"""
demo.py
=======
Traffic Violation Detection System — Main Demo Script
Flipkart Gridlock Hackathon

Runs the full pipeline on every image in a folder (or samples/ by default):
  preprocess -> detect -> check violations -> OCR plates -> save evidence

HOW TO RUN:
    # Use default samples/ folder:
    python demo.py

    # Specify a custom folder:
    python demo.py --input path/to/images

    # Specify output folder:
    python demo.py --input images/ --output my_results/

OUTPUT:
    evidence/<date>/<id>/
        annotated.jpg    <- image with colored violation boxes
        metadata.json    <- full structured record
    demo_summary.json    <- aggregated results
"""

import sys
import cv2
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


# ══════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ══════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Traffic Violation Detection System Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py
  python demo.py --input samples/
  python demo.py --input images/ --output results/
  python demo.py --input images/ --conf 0.4 --no-ocr
        """
    )
    parser.add_argument("--input",   default="samples",
                        help="Input folder containing images (default: samples/)")
    parser.add_argument("--output",  default=None,
                        help="Output folder for evidence (default: evidence/)")
    parser.add_argument("--conf",    type=float, default=0.35,
                        help="Detection confidence threshold (default: 0.35)")
    parser.add_argument("--no-ocr",  action="store_true",
                        help="Skip license plate OCR (faster)")
    parser.add_argument("--no-db",   action="store_true",
                        help="Skip database storage")
    parser.add_argument("--exts",    default=".jpg,.jpeg,.png,.bmp,.webp",
                        help="Comma-separated image extensions")
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def find_images(folder: Path, extensions: list) -> list:
    images = []
    for ext in extensions:
        images.extend(folder.glob(f"*{ext}"))
        images.extend(folder.glob(f"*{ext.upper()}"))
    # Filter out already-annotated outputs
    images = [p for p in images
              if "detected" not in p.name
              and "comparison" not in p.name
              and "evidence" not in p.name.lower()]
    return sorted(set(images))


def print_header():
    print("\n" + "="*60)
    print("  TRAFFIC VIOLATION DETECTION SYSTEM")
    print("  Flipkart Gridlock Hackathon Demo")
    print("="*60)


def print_result(idx, total, path, result, package):
    vcount = len(result.violations)
    status = "VIOLATIONS FOUND" if vcount > 0 else "No violations"
    icon   = "🚨" if vcount > 0 else "✅"
    print(f"\n  [{idx}/{total}] {icon} {path.name}")
    print(f"  Status    : {status}")
    if result.violations:
        for v in result.violations:
            print(f"    - {v.display_name:<30} conf={v.confidence:.0%}"
                  f"  fine=₹{v.fine_amount}")
    print(f"  Time      : {result.processing_ms:.0f}ms")
    print(f"  Evidence  : {package.evidence_id}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    print_header()

    # ── Validate input folder ─────────────────────────────────
    input_dir = PROJECT_ROOT / args.input
    if not input_dir.exists():
        print(f"\n  ERROR: Input folder not found: {input_dir}")
        print(f"  Create it and add some traffic images, then re-run.")
        sys.exit(1)

    extensions = [e.strip() for e in args.exts.split(",")]
    images     = find_images(input_dir, extensions)

    if not images:
        print(f"\n  No images found in: {input_dir}")
        print(f"  Supported: {args.exts}")
        print(f"\n  Add traffic photos to samples/ and re-run.")
        sys.exit(0)

    print(f"\n  Input folder : {input_dir}")
    print(f"  Images found : {len(images)}")
    print(f"  Conf threshold: {args.conf}")
    print(f"  OCR enabled  : {not args.no_ocr}")
    print(f"  DB storage   : {not args.no_db}")

    # ── Load models ───────────────────────────────────────────
    print(f"\n  Loading models (first run: ~60s) ...")
    t_load = time.time()

    from core.violation_detector  import ViolationDetector
    from core.evidence_generator  import EvidenceGenerator

    vd = ViolationDetector()
    vd.load_models()

    ocr = None
    if not args.no_ocr:
        from core.ocr import LicensePlateOCR
        ocr = LicensePlateOCR()
        ocr.load_models()

    ev_dir = Path(args.output) if args.output else None
    gen = EvidenceGenerator(
        evidence_dir = ev_dir,
        save_to_db   = not args.no_db
    )

    print(f"  Models loaded in {time.time() - t_load:.1f}s")
    print("\n" + "-"*60)

    # ── Process each image ────────────────────────────────────
    all_results = []
    total_violations = 0
    total_fines      = 0
    t_start          = time.time()

    for idx, img_path in enumerate(images, 1):
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"\n  SKIP [{idx}/{len(images)}] Cannot read: {img_path.name}")
            continue

        # Run pipeline
        try:
            viol_result = vd.detect_all(image)
            det_result  = viol_result.detection_result

            plates = []
            if ocr and det_result:
                bboxes = [v.bbox for v in det_result.vehicles]
                plates = ocr.extract(image, vehicle_bboxes=bboxes)

            # Assign plates to violations
            plate_no = plates[0].plate_text_clean if plates else None
            for v in viol_result.violations:
                if v.plate_number is None and plate_no:
                    v.plate_number = plate_no

            package = gen.generate(
                image            = image,
                image_path       = str(img_path),
                detection_result = det_result,
                violation_result = viol_result,
                plate_results    = plates,
            )

            print_result(idx, len(images), img_path, viol_result, package)

            total_violations += len(viol_result.violations)
            total_fines      += viol_result.total_fines
            all_results.append({
                "filename":       img_path.name,
                "evidence_id":    package.evidence_id,
                "violations":     viol_result.summary,
                "total_fines":    viol_result.total_fines,
                "processing_ms":  viol_result.processing_ms,
                "plates":         [p.plate_text_clean for p in plates],
                "annotated_path": package.annotated_path,
            })

        except Exception as e:
            print(f"\n  ERROR [{idx}/{len(images)}] {img_path.name}: {e}")
            continue

    # ── Summary ───────────────────────────────────────────────
    elapsed = time.time() - t_start
    print("\n" + "="*60)
    print("  DEMO COMPLETE — SUMMARY")
    print("="*60)
    print(f"  Images processed  : {len(all_results)}")
    print(f"  Total violations  : {total_violations}")
    print(f"  Total fines       : ₹{total_fines:,}")
    avg_fps = len(all_results) / elapsed if elapsed > 0 else 0
    print(f"  Throughput        : {avg_fps:.2f} images/sec")
    print(f"  Total time        : {elapsed:.1f}s")
    from config import EVIDENCE_DIR
    ev_out = ev_dir or EVIDENCE_DIR
    print(f"  Evidence saved to : {ev_out}")

    # Violation breakdown
    if total_violations > 0:
        from collections import Counter
        type_counter: Counter = Counter()
        for r in all_results:
            for vtype, count in r["violations"].items():
                type_counter[vtype] += count
        print("\n  Violation breakdown:")
        for vtype, count in type_counter.most_common():
            from config import VIOLATION_DISPLAY_NAMES
            name = VIOLATION_DISPLAY_NAMES.get(vtype, vtype)
            print(f"    {name:<35} {count}")

    # Save summary JSON
    summary = {
        "generated_at":     datetime.now().isoformat(),
        "images_processed": len(all_results),
        "total_violations": total_violations,
        "total_fines_inr":  total_fines,
        "elapsed_seconds":  round(elapsed, 2),
        "results":          all_results,
    }
    summary_path = PROJECT_ROOT / "demo_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n  Summary saved to  : demo_summary.json")
    print("\n  To view results:")
    print("    streamlit run dashboard\\app.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
