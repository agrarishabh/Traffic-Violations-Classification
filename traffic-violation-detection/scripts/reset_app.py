"""
scripts/reset_app.py
=====================
Clears all synthetic/demo data from the app.

Removes:
  - violations.db          (all database records)
  - evidence/              (all annotated images + metadata JSON)
  - samples/ subfolders    (synthetic output images)
  - Synthetic test images  (test_*.jpg, demo_*.jpg, scene_*.jpg)

Keeps:
  - data/models/           (YOLOv8 weights — took time to download)
  - data/datasets/         (training datasets)
  - venv/                  (installed packages)
  - All source code files

Run this to start fresh before adding real traffic photos.
"""

import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def confirm(msg: str) -> bool:
    ans = input(f"  {msg} [y/N]: ").strip().lower()
    return ans == "y"


def remove_dir_contents(folder: Path, label: str):
    if not folder.exists():
        print(f"  SKIP  {label} (folder does not exist)")
        return 0
    items = list(folder.iterdir())
    count = 0
    for item in items:
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
            count += 1
        except Exception as e:
            print(f"  WARN  Could not remove {item.name}: {e}")
    print(f"  OK    Cleared {label} ({count} item(s) removed)")
    return count


def main():
    print("\n" + "="*55)
    print("  APP DATA RESET — Remove all synthetic/demo data")
    print("="*55)
    print("""
  This will DELETE:
    violations.db         - all database records
    evidence/             - all annotated evidence images
    samples/preprocessed/ - preprocessing comparison images
    samples/detected/     - detection output images
    samples/violations/   - violation demo images
    samples/ocr/          - OCR demo images
    samples/evidence_demo/- evidence demo images
    samples/evaluation/   - evaluation reports
    samples/*.jpg         - synthetic test images

  This will KEEP:
    data/models/          - YOLOv8 weight files
    data/datasets/        - training datasets
    All source code
    venv/ (packages)
""")

    if not confirm("Proceed with reset?"):
        print("\n  Reset cancelled. No changes made.\n")
        return

    print("\n  Resetting ...\n")
    total = 0

    # 1. Clear database
    db_path = PROJECT_ROOT / "violations.db"
    if db_path.exists():
        db_path.unlink()
        print("  OK    Deleted violations.db")
        total += 1
    else:
        print("  SKIP  violations.db (not found)")

    # Recreate fresh empty database
    try:
        from core.database import init_db
        init_db()
        print("  OK    Created fresh empty violations.db")
    except Exception as e:
        print(f"  WARN  Could not recreate database: {e}")

    # 2. Clear evidence folder
    evidence_dir = PROJECT_ROOT / "evidence"
    total += remove_dir_contents(evidence_dir, "evidence/ (annotated images + JSON)")

    # 3. Clear sample subfolders
    sample_subdirs = [
        "samples/preprocessed",
        "samples/detected",
        "samples/violations",
        "samples/ocr",
        "samples/evidence_demo",
        "samples/evaluation",
    ]
    for subdir in sample_subdirs:
        path = PROJECT_ROOT / subdir
        total += remove_dir_contents(path, subdir)

    # 4. Remove synthetic images from samples/
    samples_dir = PROJECT_ROOT / "samples"
    removed_imgs = 0
    if samples_dir.exists():
        for f in samples_dir.glob("*.jpg"):
            try:
                f.unlink()
                removed_imgs += 1
            except Exception:
                pass
        for f in samples_dir.glob("*.png"):
            try:
                f.unlink()
                removed_imgs += 1
            except Exception:
                pass
        for f in samples_dir.glob("*.json"):
            try:
                f.unlink()
                removed_imgs += 1
            except Exception:
                pass
    if removed_imgs:
        print(f"  OK    Removed {removed_imgs} synthetic image(s) from samples/")
    total += removed_imgs

    # 5. Clear demo_summary.json
    demo_summary = PROJECT_ROOT / "demo_summary.json"
    if demo_summary.exists():
        demo_summary.unlink()
        print("  OK    Deleted demo_summary.json")
        total += 1

    print("\n" + "="*55)
    print(f"  RESET COMPLETE — {total} item(s) removed")
    print("="*55)
    print("""
  Your app is now clean and ready for real traffic photos.

  HOW TO ADD YOUR REAL PHOTOS:
    1. Copy your traffic violation images to:
           traffic-violation-detection/samples/

    2. Start the dashboard:
           streamlit run dashboard\\app.py

    3. Go to 'Live Analysis' page and upload images.
       Or batch-process all images at once:
           python demo.py

  Supported formats: JPG, PNG, BMP, WebP
""")


if __name__ == "__main__":
    main()
