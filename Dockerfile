# Stage 1: build React + TypeScript GUI (Open Source, no runtime Node needed)
FROM node:20-alpine AS frontend-builder
WORKDIR /web
COPY frontend/web/package*.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/web/ ./
RUN npm run build

# Stage 2: CPU-only Python runtime with Tesseract OCR (`pol`) + FastAPI + CLI
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_THREAD_LIMIT=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-pol \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY ocr/ ./ocr/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY dane/ ./dane/
COPY README.md ./
COPY --from=frontend-builder /web/dist ./frontend/web/dist

RUN pip install --no-cache-dir --no-deps -e . && \
    python scripts/classify_documents.py --train && \
    mkdir -p /app/input /app/output

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
