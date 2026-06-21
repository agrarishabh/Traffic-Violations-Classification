# 🚦 TrafficAI — Automated Traffic Violation Detection

**AI-powered computer vision that turns raw traffic photos into classified, timestamped, evidence-ready violation records — automatically.**

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white">
  <img alt="YOLOv8" src="https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF">
  <img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-CV-5C3EE8?logo=opencv&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="Docker" src="https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white">
</p>

> 🔴 **Live Demo:** [agrarishabh-trafficai.hf.space](https://agrarishabh-trafficai.hf.space) — running on Hugging Face Spaces

> 📂 The application lives in [`traffic-violation-detection/`](traffic-violation-detection). The `hf-space/` folder is a deployment mirror for the live Hugging Face Space.

---

## Overview

Traffic surveillance cameras generate huge volumes of images every day, but violations are still reviewed manually — slow, labour-intensive, inconsistent, and impossible to scale.

**TrafficAI** automates the entire workflow. Upload one traffic image and the system detects vehicles and road users, identifies **seven categories** of traffic violations, reads license plates with OCR, and produces annotated evidence with metadata and timestamps — in seconds.

---

## ✨ Key Features

- **End-to-end pipeline** — preprocess → detect → classify → OCR → annotated evidence → searchable record.
- **Custom-trained models** — a dedicated YOLOv8 model per violation, not a generic off-the-shelf detector *(see below)*.
- **Robust preprocessing** — auto-fixes low light, blur, noise, shadows, and haze, only when detected.
- **License plate recognition** — plate detection + multi-pass EasyOCR for registration text.
- **Evidence generation** — annotated images + JSON metadata, stored with timestamp and fine amount.
- **Interactive dashboard** — live analysis, searchable records, analytics & trends, one-click reports.
- **Deploy anywhere** — Docker + cloud-aware config; runs on **CPU**, no GPU required.

---

## 🎯 Our Differentiator — Custom-Trained Models

Rather than relying on a single generic model, we **trained a dedicated YOLOv8 model for each violation type** — helmet, seatbelt, triple riding, license plate, and more.

- 🗂️ **Datasets** curated, annotated & augmented on **Roboflow** for real road conditions
- ⚙️ **Trained** via transfer learning on **Google Colab** GPUs, exported to deployable `.pt` weights
- 📈 **Higher accuracy** than the generic baseline — fewer missed violations and fewer false alarms
- 🧩 **Modular** — any model can be retrained or swapped independently

---

## 🚓 Violations Detected

| Violation | Detection Method | Fine (₹) |
|---|---|---|
| Helmet Non-Compliance | Custom YOLOv8 model | 1,000 |
| Seatbelt Non-Compliance | Custom YOLOv8 model | 1,000 |
| Triple Riding | Person count on two-wheelers | 2,000 |
| Wrong-Side Driving | Lane direction analysis | 5,000 |
| Stop-Line Violation | Geometric bounding-box logic | 500 |
| Red-Light Violation | Traffic-light state + vehicle position | 1,000 |
| Illegal Parking | Score-based roadside heuristic | 500 |

Plus **license plate recognition (OCR)** on every detected offence.

---

## 🔬 How It Works

```
Input Image
    ↓
[ ImagePreprocessor ]   CLAHE low-light · NLM denoise · unsharp deblur · shadow/haze fix · letterbox 640×640
    ↓
[ VehicleDetector ]     YOLOv8 → vehicles, persons/riders, pedestrians, traffic lights (+ NMS)
    ↓
[ ViolationDetector ]   7 rule-based + heuristic checks with confidence scores
    ↓
[ LicensePlateOCR ]     plate region detection + multi-pass EasyOCR
    ↓
[ EvidenceGenerator ]   annotated JPG + metadata JSON + SQLite record (timestamped)
    ↓
[ FastAPI / Streamlit ] decoupled REST API + web dashboard
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Object Detection | YOLOv8 (Ultralytics) |
| Image Processing | OpenCV + Pillow |
| License Plate OCR | EasyOCR |
| Backend API | FastAPI + Uvicorn |
| Dashboard | Streamlit (multi-page) |
| Database | SQLite via SQLAlchemy |
| Evaluation | scikit-learn (Precision, Recall, F1, mAP) |
| Packaging | Docker |
| Language | Python 3.10+ |

---

## 📁 Repository Structure

```
Traffic-Violations-Classification/
├── traffic-violation-detection/    # ← main application
│   ├── core/                       #   CV pipeline (preprocess, detect, OCR, evidence)
│   ├── app/                        #   FastAPI REST API
│   ├── dashboard/                  #   Streamlit dashboard (st.navigation)
│   ├── data/models/                #   YOLOv8 .pt weights (downloaded at runtime)
│   ├── config.py                   #   all settings in one place
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
└── hf-space/                       # deployment mirror for Hugging Face Spaces
```

---

## 🚀 Getting Started (Local)

### Prerequisites
- Python 3.10+ (check **"Add Python to PATH"** during install)
- ~8 GB free disk space

```bash
# 1. Enter the application folder
cd traffic-violation-detection

# 2. Create & activate a virtual environment
python -m venv venv
# Windows: venv\Scripts\activate    |    macOS/Linux: source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

> YOLOv8 base weights and EasyOCR models download automatically on first run.

Windows users can instead run the one-click `setup_windows.bat` inside `traffic-violation-detection/`.

---

## ▶️ Running

All commands run from inside `traffic-violation-detection/`.

**Dashboard (recommended for demo):**
```bash
streamlit run dashboard/app.py
```
Open http://localhost:8501 and upload a traffic image on the **Live Analysis** page.

**REST API:**
```bash
uvicorn app.main:app --reload --port 8000
```
Open http://localhost:8000/docs for interactive Swagger docs.

**Batch demo:**
```bash
python demo.py --input samples/
```

---

## 🐳 Deployment

**Docker:**
```bash
cd traffic-violation-detection
docker build -t trafficai .
docker run -p 8501:8501 trafficai
```

**Hugging Face Spaces:** the app auto-detects the cloud environment (`SPACE_ID`) and routes writable paths accordingly. Also portable to **Railway**, **Render**, or any VPS with no code change.

---

## 📊 Performance Evaluation

Evaluated with standard detection metrics: **mAP**, **Precision**, **Recall**, **F1-Score**, and **FPS** (throughput).

```bash
python scripts/evaluate.py     # accuracy metrics
python scripts/benchmark.py    # speed benchmark
```

---

## 👥 Team

**Flipkart Gridlock Hackathon**

- **Rishabh Agrahari**
- **Ayush Bind**
- **Shreyas Mohan**

---

## 🙏 Acknowledgements

Built with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), [OpenCV](https://opencv.org/), [EasyOCR](https://github.com/JaidedAI/EasyOCR), [FastAPI](https://fastapi.tiangolo.com/), and [Streamlit](https://streamlit.io/). Datasets managed with [Roboflow](https://roboflow.com/); models trained on Google Colab.
