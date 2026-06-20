"""
scripts/download_datasets.py
============================
Downloads annotated training datasets from Roboflow Universe (FREE).

Datasets downloaded:
  1. Helmet Detection      → data/datasets/helmet/
  2. Seatbelt Detection    → data/datasets/seatbelt/
  3. License Plate (India) → data/datasets/license_plate/

You need a FREE Roboflow API key for this script.
  → Sign up at: https://app.roboflow.com  (free, no credit card)
  → After login: Settings → API Keys → Copy your key

HOW TO RUN:
    python scripts\\download_datasets.py

The script will ask for your API key interactively.
You can also set it as an environment variable:
    set ROBOFLOW_API_KEY=your_key_here
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Roboflow Dataset Definitions ─────────────────────────────
# Each entry: (display_name, workspace, project_name, version, save_folder)
# These are real public datasets on Roboflow Universe

DATASETS = [
    {
        "name":         "Helmet Detection",
        "workspace":    "joseph-nelson",
        "project":      "hard-hat-workers",
        "version":      1,
        "save_dir":     "data/datasets/helmet",
        "format":       "yolov8",
        "description":  "Detects helmets/hard hats on people",
        "classes":      ["helmet", "no-helmet", "head"],
    },
    {
        "name":         "Seatbelt Detection",
        "workspace":    "roboflow-universe-projects",
        "project":      "seat-belt-detection-lfp5n",
        "version":      2,
        "save_dir":     "data/datasets/seatbelt",
        "format":       "yolov8",
        "description":  "Detects seatbelt compliance in vehicles",
        "classes":      ["seatbelt", "no-seatbelt"],
    },
    {
        "name":         "License Plate Detection",
        "workspace":    "roboflow-universe-projects",
        "project":      "license-plate-recognition-rxg4e",
        "version":      4,
        "save_dir":     "data/datasets/license_plate",
        "format":       "yolov8",
        "description":  "Detects license plate regions in images",
        "classes":      ["license-plate"],
    },
]


def get_api_key() -> str:
    """Get Roboflow API key from env var or user input."""
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if key:
        print(f"  OK  API key loaded from environment variable.")
        return key

    print("""
  ──────────────────────────────────────────────────
  ROBOFLOW API KEY REQUIRED
  ──────────────────────────────────────────────────
  1. Open your browser and go to: https://app.roboflow.com
  2. Sign up for a FREE account (no credit card needed)
  3. After login, click on your profile picture (top right)
  4. Click 'Settings'
  5. Click 'Roboflow API' in the left menu
  6. Copy your 'Private API Key'
  7. Paste it below when prompted
  ──────────────────────────────────────────────────
""")
    key = input("  Paste your Roboflow API key and press Enter: ").strip()
    if not key:
        print("  FAIL  No API key entered. Exiting.")
        sys.exit(1)
    return key


def download_dataset(api_key: str, dataset_info: dict, base_dir: Path) -> bool:
    """Download a single dataset from Roboflow."""
    name     = dataset_info["name"]
    workspace = dataset_info["workspace"]
    project  = dataset_info["project"]
    version  = dataset_info["version"]
    fmt      = dataset_info["format"]
    save_dir = base_dir / dataset_info["save_dir"]

    print(f"\n  Downloading: {name}")
    print(f"  Roboflow:    {workspace}/{project} (v{version})")
    print(f"  Save to:     {save_dir}")

    try:
        from roboflow import Roboflow

        rf = Roboflow(api_key=api_key)
        project_obj = rf.workspace(workspace).project(project)
        dataset = project_obj.version(version).download(
            model_format=fmt,
            location=str(save_dir),
            overwrite=False
        )
        print(f"  OK  {name} downloaded successfully!")
        print(f"  Location: {save_dir}")
        return True

    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "403" in error_msg or "authentication" in error_msg.lower():
            print(f"  FAIL  Authentication error — check your API key.")
        elif "404" in error_msg:
            print(f"  FAIL  Dataset not found — it may have moved.")
            print(f"  TRY  Search manually at: https://universe.roboflow.com")
        elif "quota" in error_msg.lower():
            print(f"  FAIL  Download quota exceeded — try again tomorrow (free tier limit).")
        else:
            print(f"  FAIL  Download error: {error_msg}")
        return False


def check_existing_datasets(base_dir: Path) -> None:
    """Report which datasets are already downloaded."""
    print("\n  Checking existing datasets...")
    all_present = True
    for ds in DATASETS:
        path = base_dir / ds["save_dir"]
        # Dataset exists if folder has YOLO yaml + images
        yaml_files = list(path.glob("*.yaml")) if path.exists() else []
        if yaml_files:
            print(f"  SKIP  {ds['name']} already exists: {path}")
        else:
            print(f"  NEED  {ds['name']}")
            all_present = False
    return all_present


def show_manual_download_instructions():
    """Show manual download instructions if automated download fails."""
    print("""
  ══════════════════════════════════════════════════════════
  MANUAL DOWNLOAD INSTRUCTIONS (if automated download fails)
  ══════════════════════════════════════════════════════════

  1. HELMET DATASET:
     URL: https://universe.roboflow.com/joseph-nelson/hard-hat-workers
     → Click 'Download Dataset' → Choose 'YOLOv8' format
     → Extract ZIP to:  data/datasets/helmet/

  2. SEATBELT DATASET:
     URL: https://universe.roboflow.com/roboflow-universe-projects/seat-belt-detection-lfp5n
     → Click 'Download Dataset' → Choose 'YOLOv8' format
     → Extract ZIP to:  data/datasets/seatbelt/

  3. LICENSE PLATE DATASET:
     URL: https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e
     → Click 'Download Dataset' → Choose 'YOLOv8' format
     → Extract ZIP to:  data/datasets/license_plate/

  EXPECTED FOLDER STRUCTURE AFTER DOWNLOAD:
     data/datasets/helmet/
         ├── train/
         │     ├── images/      ← training images (.jpg)
         │     └── labels/      ← YOLO label files (.txt)
         ├── valid/
         │     ├── images/
         │     └── labels/
         └── data.yaml          ← dataset config file
  ══════════════════════════════════════════════════════════
""")


def main():
    print("\n" + "="*55)
    print("  ROBOFLOW DATASET DOWNLOADER")
    print("="*55)

    base_dir = PROJECT_ROOT

    # ── Check roboflow package ─────────────────────────────────
    print("\n[Step 1/3] Checking roboflow package...")
    try:
        import roboflow
        print(f"  OK  roboflow {roboflow.__version__} found")
    except ImportError:
        print("  FAIL  roboflow not installed")
        print("  FIX → pip install roboflow==1.1.48")
        sys.exit(1)

    # ── Check existing datasets ────────────────────────────────
    print("\n[Step 2/3] Checking existing datasets...")
    all_present = check_existing_datasets(base_dir)

    if all_present:
        print("\n  All datasets already downloaded. Nothing to do!")
        print("  Delete dataset folders and re-run to re-download.")
        sys.exit(0)

    # ── Get API key and download ───────────────────────────────
    print("\n[Step 3/3] Downloading datasets...")
    api_key = get_api_key()

    success_count = 0
    fail_count    = 0

    for dataset in DATASETS:
        ds_path = base_dir / dataset["save_dir"]
        yaml_files = list(ds_path.glob("*.yaml")) if ds_path.exists() else []
        if yaml_files:
            print(f"\n  SKIP  {dataset['name']} (already exists)")
            success_count += 1
            continue

        success = download_dataset(api_key, dataset, base_dir)
        if success:
            success_count += 1
        else:
            fail_count += 1

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "="*55)
    print(f"  DOWNLOAD SUMMARY: {success_count} OK, {fail_count} failed")
    print("="*55)

    if fail_count > 0:
        show_manual_download_instructions()
    else:
        print("""
  All datasets downloaded!

  NEXT STEP:
  Phase 2 — Image Preprocessing Pipeline
  (wait for Kiro to start Phase 2)
""")


if __name__ == "__main__":
    main()
