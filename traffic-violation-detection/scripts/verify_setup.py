"""
scripts/verify_setup.py
=======================
Beginner-Friendly Setup Verification Script

Run this after setup_windows.bat to confirm everything installed correctly.
Each check prints [  OK  ] or [ FAIL ] so you know exactly what works.

HOW TO RUN (from the traffic-violation-detection folder):
    python scripts/verify_setup.py

If something fails, the script tells you exactly how to fix it.
"""

import sys
import os
from pathlib import Path

# ── Helpers ─────────────────────────────────────────────────────────────────
OK   = "[  OK  ]"
FAIL = "[ FAIL ]"
WARN = "[ WARN ]"
INFO = "[ INFO ]"

passed = 0
failed = 0
warnings = 0

def check(label: str, condition: bool, fix_hint: str = ""):
    global passed, failed
    if condition:
        print(f"  {OK}  {label}")
        passed += 1
    else:
        print(f"  {FAIL}  {label}")
        if fix_hint:
            print(f"         FIX → {fix_hint}")
        failed += 1

def warn(label: str, message: str = ""):
    global warnings
    print(f"  {WARN}  {label}")
    if message:
        print(f"         NOTE → {message}")
    warnings += 1

def section(title: str):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


# ════════════════════════════════════════════════════════════
print("\n" + "═"*55)
print("  TRAFFIC VIOLATION DETECTION SYSTEM")
print("  Setup Verification")
print("═"*55)

# ── Section 1: Python Version ────────────────────────────────
section("1/6  PYTHON VERSION")
major, minor = sys.version_info.major, sys.version_info.minor
print(f"  {INFO}  Detected: Python {major}.{minor}.{sys.version_info.micro}")
check(
    f"Python version is 3.10+",
    major == 3 and minor >= 10,
    fix_hint="Install Python 3.10+ from https://www.python.org/downloads/"
)

# ── Section 2: Virtual Environment ───────────────────────────
section("2/6  VIRTUAL ENVIRONMENT")
in_venv = (
    hasattr(sys, 'real_prefix') or
    (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
)
check(
    "Running inside virtual environment (venv)",
    in_venv,
    fix_hint="Run: venv\\Scripts\\activate.bat  THEN  python scripts\\verify_setup.py"
)

# ── Section 3: Core Package Imports ──────────────────────────
section("3/6  REQUIRED PACKAGES")
print("  Testing each package import (this may take 10–30 seconds)...\n")

# List of (display_name, import_name, fix_hint)
packages = [
    ("ultralytics (YOLOv8)",      "ultralytics",       "pip install ultralytics==8.3.0"),
    ("OpenCV (cv2)",               "cv2",               "pip install opencv-python==4.10.0.84"),
    ("Pillow (PIL)",               "PIL",               "pip install Pillow==10.4.0"),
    ("NumPy",                      "numpy",             "pip install numpy==1.26.4"),
    ("EasyOCR",                    "easyocr",           "pip install easyocr==1.7.2"),
    ("FastAPI",                    "fastapi",           "pip install fastapi==0.115.0"),
    ("Uvicorn",                    "uvicorn",           "pip install uvicorn[standard]==0.30.0"),
    ("Streamlit",                  "streamlit",         "pip install streamlit==1.39.0"),
    ("Pandas",                     "pandas",            "pip install pandas==2.2.3"),
    ("Plotly",                     "plotly",            "pip install plotly==5.24.1"),
    ("Matplotlib",                 "matplotlib",        "pip install matplotlib==3.9.2"),
    ("Scikit-learn",               "sklearn",           "pip install scikit-learn==1.5.2"),
    ("SQLAlchemy",                 "sqlalchemy",        "pip install sqlalchemy==2.0.36"),
    ("Pydantic",                   "pydantic",          "pip install pydantic==2.9.2"),
    ("Requests",                   "requests",          "pip install requests==2.32.3"),
    ("TQDM (progress bars)",       "tqdm",              "pip install tqdm==4.66.6"),
    ("PyTorch (torch)",            "torch",             "pip install torch  (auto-installed with ultralytics)"),
    ("Torchvision",                "torchvision",       "pip install torchvision  (auto-installed with ultralytics)"),
    ("Roboflow",                   "roboflow",          "pip install roboflow==1.1.48"),
    ("Aiofiles",                   "aiofiles",          "pip install aiofiles==24.1.0"),
    ("Python-dotenv",              "dotenv",            "pip install python-dotenv==1.0.1"),
    ("Pytest",                     "pytest",            "pip install pytest==8.3.3"),
]

for display_name, import_name, fix in packages:
    try:
        __import__(import_name)
        check(f"{display_name}", True)
    except ImportError:
        check(f"{display_name}", False, fix_hint=fix)

# ── Section 4: Package Versions ──────────────────────────────
section("4/6  KEY PACKAGE VERSIONS")
try:
    import torch
    print(f"  {INFO}  PyTorch version: {torch.__version__}")
    print(f"  {INFO}  CUDA available:  {torch.cuda.is_available()} "
          f"{'(GPU will be used!)' if torch.cuda.is_available() else '(CPU mode — OK for demo)'}")
except ImportError:
    warn("PyTorch not importable — check ultralytics installation")

try:
    import ultralytics
    print(f"  {INFO}  Ultralytics (YOLOv8) version: {ultralytics.__version__}")
except ImportError:
    pass

try:
    import cv2
    print(f"  {INFO}  OpenCV version: {cv2.__version__}")
except ImportError:
    pass

try:
    import streamlit
    print(f"  {INFO}  Streamlit version: {streamlit.__version__}")
except ImportError:
    pass

# ── Section 5: Project Directory Structure ───────────────────
section("5/6  PROJECT FOLDERS")
BASE = Path(__file__).resolve().parent.parent

required_dirs = [
    ("data/models",              "YOLOv8 weight files go here"),
    ("data/datasets/helmet",     "Helmet dataset goes here"),
    ("data/datasets/seatbelt",   "Seatbelt dataset goes here"),
    ("data/datasets/license_plate", "License plate dataset goes here"),
    ("data/test",                "Test images go here"),
    ("evidence",                 "Annotated outputs saved here"),
    ("samples",                  "Demo images go here"),
    ("core",                     "CV pipeline modules"),
    ("app",                      "FastAPI backend"),
    ("dashboard",                "Streamlit frontend"),
    ("scripts",                  "Utility scripts"),
]

for rel_path, desc in required_dirs:
    full_path = BASE / rel_path
    check(
        f"{rel_path:<30} ({desc})",
        full_path.exists(),
        fix_hint=f"mkdir {rel_path}"
    )

# ── Section 6: Project Config & Core Files ───────────────────
section("6/6  PROJECT FILES")
required_files = [
    ("config.py",                      "Main configuration"),
    ("requirements.txt",               "Package list"),
    ("core/preprocessor.py",           "Image preprocessing module"),
    ("core/detector.py",               "Vehicle detection module"),
    ("core/violation_detector.py",     "Violation detection module"),
    ("core/ocr.py",                    "License plate OCR module"),
    ("core/evidence_generator.py",     "Evidence generation module"),
    ("core/__init__.py",               "Core package init"),
    ("app/__init__.py",                "App package init"),
    ("dashboard/__init__.py",          "Dashboard package init"),
]

for rel_path, desc in required_files:
    full_path = BASE / rel_path
    check(
        f"{rel_path:<40} ({desc})",
        full_path.exists(),
        fix_hint=f"File missing — re-run project scaffolding"
    )

# ── Quick OpenCV Image Test ───────────────────────────────────
section("BONUS  QUICK OPENCV TEST")
try:
    import cv2
    import numpy as np
    # Create a tiny test image in memory (no file needed)
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    test_img[:] = (0, 128, 255)                          # Fill with orange
    cv2.rectangle(test_img, (10, 10), (90, 90), (255, 255, 255), 2)  # White box
    # Encode to JPEG and decode back (tests full encode/decode pipeline)
    _, encoded = cv2.imencode('.jpg', test_img)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    check("OpenCV can create, encode, and decode images", decoded is not None)
except Exception as e:
    check(f"OpenCV image test", False, fix_hint=str(e))

# ── Final Summary ─────────────────────────────────────────────
print("\n" + "═"*55)
print(f"  VERIFICATION SUMMARY")
print(f"  Passed  : {passed}")
print(f"  Failed  : {failed}")
print(f"  Warnings: {warnings}")
print("═"*55)

if failed == 0:
    print("""
  ✔  ALL CHECKS PASSED!
  You are ready to proceed.

  NEXT STEPS:
  1.  python scripts\\download_models.py
      (downloads YOLOv8 weight files)

  2.  python scripts\\create_sample_images.py
      (downloads free test images)

  3.  python scripts\\download_datasets.py
      (downloads training datasets from Roboflow)
""")
else:
    print(f"""
  ✘  {failed} CHECK(S) FAILED.
  Fix the issues shown above, then re-run:
      python scripts\\verify_setup.py

  MOST COMMON FIX:
  Make sure your virtual environment is active:
      venv\\Scripts\\activate.bat
  Then reinstall:
      pip install -r requirements.txt
""")

sys.exit(0 if failed == 0 else 1)
