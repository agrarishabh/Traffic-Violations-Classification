# ============================================================
# Dockerfile  —  Traffic Violation Detection System
# Targets: Railway · Render · Hugging Face Spaces · any VPS
#
# Build:  docker build -t trafficai .
# Run:    docker run -p 8501:8501 trafficai
# ============================================================

FROM python:3.10-slim

# ── System dependencies needed by OpenCV + EasyOCR ───────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        wget \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies first (cached layer) ─────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy project source ───────────────────────────────────────
COPY . .

# ── Create runtime directories ────────────────────────────────
RUN mkdir -p \
    data/models \
    data/datasets \
    evidence \
    samples

# ── Environment variables ─────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    # Tells the app it is running in cloud mode
    TRAFFICAI_CLOUD=1

# ── Models download at runtime (not build time) ───────────────
# Keeping build fast — YOLOv8 and EasyOCR download on first startup

# ── Expose Streamlit port ─────────────────────────────────────
EXPOSE 8501

# ── Health check ─────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:${PORT:-8501}/_stcore/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────
# Use $PORT from Railway/Render env, fallback to 8501 for local
CMD ["sh", "-c", "python startup.py && exec streamlit run dashboard/app.py \
     --server.port=${PORT:-8501} \
     --server.address=0.0.0.0 \
     --server.headless=true \
     --server.maxUploadSize=10 \
     --server.enableCORS=false \
     --server.enableXsrfProtection=false"]
