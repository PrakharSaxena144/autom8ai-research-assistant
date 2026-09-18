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

# Ingest the sample corpus at build time: the container starts ready, with no external
# vector database and no parsing/OCR work at startup. Needs no API key.
RUN python -c "from app.vectorstore import get_kb; from app.ingestion.pipeline import seed_directory; \
kb = get_kb(); \
[print(r.status, r.source, r.chunks, r.error or '') for r in seed_directory(kb, 'data/corpus')]; \
print('chunks in collection:', kb.count())"

# Hugging Face Spaces runs containers as uid 1000; keep writable dirs owned by that user.
RUN useradd -m -u 1000 appuser && mkdir -p /app/storage && chown -R appuser /app
USER appuser

EXPOSE 8000
# One worker: embedded Qdrant is a single-process store and conversation memory is in-process.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
