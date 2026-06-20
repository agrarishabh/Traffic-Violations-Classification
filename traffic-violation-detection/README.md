# Traffic Violation Detection System
### Flipkart Gridlock Hackathon — AI-Based Traffic Monitoring

A computer vision system that automatically detects, classifies, and documents
traffic violations from photographs using YOLOv8 and OpenCV.

---

## What This System Can Detect

| Violation | Method |
|---|---|
| Helmet Non-Compliance | Specialized YOLOv8 model |
| Seatbelt Non-Compliance | Specialized YOLOv8 model |
| Triple Riding | Person count on two-wheelers |
| Wrong-Side Driving | Lane direction analysis |
| Stop-Line Violation | Geometric bounding box logic |
| Red-Light Violation | Traffic light color + vehicle position |
| Illegal Parking | Zone detection |

---

## Tech Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv8 (Ultralytics) |
| Image Processing | OpenCV + Pillow |
| License Plate OCR | EasyOCR |
| Backend API | FastAPI |
| Dashboard | Streamlit |
| Database | SQLite via SQLAlchemy |
| Language | Python 3.10+ |

---

## Project Structure

```
traffic-violation-detection/
│
├── core/                          ← CV Pipeline (main logic)
│   ├── preprocessor.py            → Image enhancement (low light, blur fix)
│   ├── detector.py                → Vehicle & person detection (YOLOv8)
│   ├── violation_detector.py      → Violation checking logic
│   ├── ocr.py                     → License plate OCR
│   ├── evidence_generator.py      → Annotated image + JSON output
│   └── database.py                → SQLite database CRUD
│
├── app/                           ← FastAPI REST API
│   ├── main.py                    → API entry point
│   ├── models/schemas.py          → Request/response schemas
│   └── routers/                   → Route handlers
│
├── dashboard/                     ← Streamlit Web Dashboard
│   └── app.py                     → Dashboard entry point
│
├── data/
│   ├── models/                    → YOLOv8 .pt weight files
│   ├── datasets/                  → Training datasets (Roboflow)
│   │   ├── helmet/
│   │   ├── seatbelt/
│   │   └── license_plate/
│   └── test/                      → Test images for evaluation
│
├── evidence/                      → Output: annotated images + JSON
├── samples/                       → Demo/test images
├── scripts/                       → Setup and utility scripts
│
├── config.py                      ← ALL settings in one place
├── requirements.txt               ← Python dependencies
└── setup_windows.bat              ← One-click Windows setup
```

---

## SETUP GUIDE (Step by Step for Beginners)

### Prerequisites (Do these BEFORE anything else)

1. **Install Python 3.10+**
   - Download from: https://www.python.org/downloads/
   - During installation, **check the box that says "Add Python to PATH"**
   - To verify: open Command Prompt, type `python --version`
   - You should see: `Python 3.10.x` or higher

2. **Install Git** (optional but recommended)
   - Download from: https://git-scm.com/download/win

3. **Create a Roboflow account** (needed for datasets)
   - Go to: https://app.roboflow.com
   - Sign up for FREE (no credit card needed)

4. **Ensure you have 8 GB free disk space**

---

### Step 1 — Run the Setup Script

Open **Command Prompt** (press `Win + R`, type `cmd`, press Enter).

Navigate to the project folder:
```cmd
cd path\to\traffic-violation-detection
```
*(Replace `path\to` with the actual path on your computer)*

Run the setup script:
```cmd
setup_windows.bat
```

This will:
- Check Python is installed
- Create a virtual environment (isolated Python environment)
- Install all required packages from requirements.txt
- Takes **5–15 minutes** depending on internet speed

> **What is a virtual environment?**
> It's like a clean Python installation just for this project.
> It prevents package conflicts with other Python projects.

---

### Step 2 — Activate the Virtual Environment

**Every time you open a new Command Prompt**, you must activate:
```cmd
venv\Scripts\activate.bat
```

Your prompt will change to show `(venv)` at the start:
```
(venv) C:\...\traffic-violation-detection>
```
This means the virtual environment is active. Good to go.

---

### Step 3 — Verify Everything Works

```cmd
python scripts\verify_setup.py
```

Expected output — all checks should show `[  OK  ]`:
```
  [  OK  ]  Python version is 3.10+
  [  OK  ]  Running inside virtual environment (venv)
  [  OK  ]  ultralytics (YOLOv8)
  [  OK  ]  OpenCV (cv2)
  ...
  VERIFICATION SUMMARY
  Passed  : 30+
  Failed  : 0
```

If any check fails, the script shows you exactly how to fix it.

---

### Step 4 — Download YOLOv8 Models

```cmd
python scripts\download_models.py
```

This downloads `yolov8n.pt` (~6 MB) — the core detection model.
Requires internet connection. Takes 1–2 minutes.

---

### Step 5 — Download Test Images

```cmd
python scripts\create_sample_images.py
```

Creates synthetic test images in the `samples/` folder.
These are used to verify the annotation pipeline works.

---

### Step 6 — Download Training Datasets (from Roboflow)

```cmd
python scripts\download_datasets.py
```

The script will ask for your Roboflow API key.

**How to get your Roboflow API key:**
1. Log into https://app.roboflow.com
2. Click your profile picture (top-right corner)
3. Click **Settings**
4. Click **Roboflow API** in the left menu
5. Copy the **Private API Key**
6. Paste it into the script when asked

If automated download fails, the script shows you manual download links.

---

### Step 7 — Run the Database Test

```cmd
python core\database.py
```

Expected output:
```
  OK  Inserted violation: ID=1
  OK  Query returned 1 record(s)
  OK  Stats: {'total_images': 0, ...}
  OK  Test data cleaned up
  Database test PASSED!
```

---

## Running the System (After All Phases Are Complete)

### Start the API server:
```cmd
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Open: http://localhost:8000/docs (interactive API documentation)

### Start the Dashboard:
```cmd
streamlit run dashboard\app.py
```
Open: http://localhost:8501

---

## Common Errors and Fixes

| Error | Fix |
|---|---|
| `python` not recognized | Reinstall Python, check "Add to PATH" |
| `pip` not found | Run `python -m pip install --upgrade pip` |
| `ModuleNotFoundError` | Activate venv: `venv\Scripts\activate.bat` then `pip install -r requirements.txt` |
| Download timeout | Check internet connection, try again |
| Roboflow 401 error | Check your API key, re-login to roboflow.com |
| `CUDA not available` | Normal on CPU — the system works without GPU |
| Port 8000 already in use | Change port: `uvicorn app.main:app --port 8001` |

---

## Performance Evaluation Metrics

The system is evaluated using:
- **mAP** (mean Average Precision) — overall detection accuracy
- **Precision** — of all violations flagged, how many were real
- **Recall** — of all real violations, how many were found
- **F1-Score** — harmonic mean of Precision and Recall
- **FPS** — frames processed per second (speed benchmark)

---

## Quick Start — Run Everything

### Option A: Streamlit Dashboard (recommended for demo)
```cmd
venv\Scripts\activate.bat
streamlit run dashboard\app.py
```
Open http://localhost:8501 — upload a traffic image on the Live Analysis page.

### Option B: Demo Script (batch process a folder)
```cmd
venv\Scripts\activate.bat
python demo.py --input samples\
```

### Option C: REST API
```cmd
venv\Scripts\activate.bat
uvicorn app.main:app --reload --port 8000
```
Open http://localhost:8000/docs — interactive Swagger UI.

### Run Evaluation
```cmd
python scripts\evaluate.py
```

### Run Benchmark
```cmd
python scripts\benchmark.py
```

---

## Architecture

```
Input Image
    ↓
[ImagePreprocessor]   — CLAHE, deblur, denoise, letterbox resize
    ↓
[VehicleDetector]     — YOLOv8n (COCO): cars, bikes, persons, traffic lights
    ↓
[ViolationDetector]   — 7 rule-based + heuristic checks
    ↓
[LicensePlateOCR]     — contour detection + EasyOCR
    ↓
[EvidenceGenerator]   — annotated JPG + metadata JSON + SQLite record
    ↓
[FastAPI / Streamlit] — REST API + web dashboard
```

---

## Phase Progress

| Phase | Description | Status |
|---|---|---|
| 1 | Project Scaffolding & Setup | ✅ Complete |
| 2 | Image Preprocessing Pipeline | ✅ Complete |
| 3 | Vehicle & Road User Detection | ✅ Complete |
| 4 | Traffic Violation Detection | ✅ Complete |
| 5 | License Plate Recognition | ✅ Complete |
| 6 | Evidence Generation | ✅ Complete |
| 7 | Backend API | ✅ Complete |
| 8 | Analytics Dashboard | ✅ Complete |
| 9 | Evaluation & Testing | ✅ Complete |
| 10 | Polish & Demo Prep | ✅ Complete |
