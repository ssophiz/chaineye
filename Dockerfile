# =============================================================================
# ChainEye (체인아이) — 2026 금융 AI Challenge
# Single-image deployment: builds the React frontend, then serves it together
# with the FastAPI backend from one Python process.
#
# Target: Hugging Face Spaces (Docker SDK) — listens on $PORT (default 7860),
# binds 0.0.0.0. Also runs locally with `docker run -p 7860:7860 chaineye`.
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1 — build the Vite/React frontend into app/frontend/dist
# -----------------------------------------------------------------------------
FROM node:24-slim AS frontend

WORKDIR /build/frontend

# Install deps first (better layer caching) using the committed lockfile.
COPY app/frontend/package.json app/frontend/package-lock.json ./
RUN npm ci

# Then bring in the sources and build the static bundle.
COPY app/frontend/ ./
RUN npm run build
# Result: /build/frontend/dist

# -----------------------------------------------------------------------------
# Stage 2 — Python runtime serving API + static frontend
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# libgomp1 is required by LightGBM (OpenMP runtime) at import/predict time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

# HF Spaces runs the container as a non-root user (uid 1000). Create it so the
# app is writable/consistent locally and on Spaces.
RUN useradd -m -u 1000 appuser

WORKDIR /app

# --- Python dependencies (own layer for caching) ---
COPY deploy/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# --- Application code + model artifacts ---
# ML module + model artifacts (feature_table.parquet is ~58MB — expected).
# The gcn/ subfolder and raw datasets are excluded via .dockerignore.
COPY app/ml/ /app/app/ml/
# Backend service.
COPY app/backend/ /app/app/backend/
# Built frontend from stage 1 -> where the backend serves it (app/frontend/dist).
COPY --from=frontend /build/frontend/dist /app/app/frontend/dist

RUN chown -R appuser:appuser /app
USER appuser

# Launch as a package so imports behave consistently in tests and production.
WORKDIR /app

EXPOSE 7860

# Bind 0.0.0.0 and honour the platform-provided $PORT (default 7860 for Spaces).
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '7860') + '/health', timeout=4)" || exit 1

CMD ["sh", "-c", "python -m uvicorn app.backend.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
