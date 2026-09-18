FROM python:3.12-slim

# tesseract = OCR for scanned PDFs and images; libgomp1 is needed by onnxruntime (FastEmbed)
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/app \
    MODEL_CACHE_DIR=/app/models \
    QDRANT_PATH=/app/storage/qdrant \
    PORT=8000

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY data ./data
COPY scripts ./scripts
COPY eval ./eval

# Download the embedding models at build time so the first request is fast and startup needs no download.
RUN python -c "from fastembed import TextEmbedding, SparseTextEmbedding; \
TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/app/models'); \
SparseTextEmbedding('Qdrant/bm25', cache_dir='/app/models')"

# Hugging Face Spaces runs containers as uid 1000; keep writable dirs owned by that user.
RUN useradd -m -u 1000 appuser && mkdir -p /app/storage && chown -R appuser /app
USER appuser

EXPOSE 8000
# One worker: embedded Qdrant is a single-process store and conversation memory is in-process.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
