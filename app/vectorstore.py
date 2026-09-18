"""Knowledge base on Qdrant with hybrid retrieval.

* Dense vectors: BAAI/bge-small-en-v1.5 via FastEmbed (ONNX, CPU, free)
* Sparse vectors: Qdrant/bm25 via FastEmbed, with Qdrant's IDF modifier
* Fusion: Reciprocal Rank Fusion inside Qdrant (langchain-qdrant RetrievalMode.HYBRID)

Hybrid matters for this corpus: codes like "SUP-117", "LX-9" or "IR-2025-031" are matched far better
by BM25 than by a small dense model, while paraphrased questions need the dense side.

Storage: embedded on-disk Qdrant by default (zero setup). Set QDRANT_URL + QDRANT_API_KEY to use a
free Qdrant Cloud cluster so the knowledge base survives redeploys.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models

from .config import settings

log = logging.getLogger(__name__)

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"


class FastEmbedDense(Embeddings):
    """Minimal LangChain Embeddings wrapper around fastembed.TextEmbedding."""

    def __init__(self, model_name: str, cache_dir: str | None = None):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._model.embed(list(texts), batch_size=32)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


class KnowledgeBase:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._titles_cache: list[str] | None = None
        self.collection = settings.collection_name
        Path(settings.model_cache_dir).mkdir(parents=True, exist_ok=True)

        if settings.qdrant_url:
            self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None, timeout=60)
            self.backend = "qdrant-cloud"
        else:
            Path(settings.qdrant_path).mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=settings.qdrant_path)
            self.backend = "qdrant-embedded"

        self.dense = FastEmbedDense(settings.embedding_model, cache_dir=settings.model_cache_dir)
        self.sparse = FastEmbedSparse(model_name=settings.sparse_model, cache_dir=settings.model_cache_dir)
        self._ensure_collection(len(self.dense.embed_query("dimension probe")))

        self.store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection,
            embedding=self.dense,
            sparse_embedding=self.sparse,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name=DENSE_VECTOR,
            sparse_vector_name=SPARSE_VECTOR,
        )

    # ---- setup -------------------------------------------------------------------------
    def _ensure_collection(self, dim: int) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={DENSE_VECTOR: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
                sparse_vectors_config={SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)},
            )
            log.info("created collection %s (dim=%s)", self.collection, dim)
        if self.backend == "qdrant-cloud":
            # Qdrant Cloud may refuse filters on unindexed payload fields (strict mode).
            for key in ("metadata.doc_id",):
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection, field_name=key,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                except Exception as exc:  # already exists
                    log.debug("payload index %s: %s", key, exc)

    @staticmethod
    def _doc_filter(doc_id: str) -> models.Filter:
        return models.Filter(must=[models.FieldCondition(key="metadata.doc_id", match=models.MatchValue(value=doc_id))])

    # ---- writes ------------------------------------------------------------------------
    def add_chunks(self, docs: list[Document], ids: list[str]) -> None:
        with self._lock:
            self.store.add_documents(docs, ids=ids, batch_size=64)
            self._titles_cache = None

    def delete_document(self, doc_id: str) -> bool:
        if not self.get_document(doc_id):
            return False
        with self._lock:
            self.client.delete(
                collection_name=self.collection,
                points_selector=models.FilterSelector(filter=self._doc_filter(doc_id)),
            )
            self._titles_cache = None
        return True

    # ---- reads -------------------------------------------------------------------------
    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        with self._lock:
            return self.store.similarity_search_with_score(query, k=k)

    def titles(self) -> list[str]:
        """Document titles, cached until the next add/delete (used to give the planner context)."""
        if self._titles_cache is None:
            self._titles_cache = [d.get("title") or d.get("source") or "" for d in self.list_documents()]
        return self._titles_cache

    def count(self) -> int:
        with self._lock:
            return self.client.count(collection_name=self.collection, exact=True).count

    def get_document(self, doc_id: str) -> dict | None:
        with self._lock:
            points, _ = self.client.scroll(
                collection_name=self.collection, scroll_filter=self._doc_filter(doc_id),
                limit=1, with_payload=True, with_vectors=False,
            )
        if not points:
            return None
        docs = self.list_documents(doc_id=doc_id)
        return docs[0] if docs else None

    def list_documents(self, doc_id: str | None = None) -> list[dict]:
        """Aggregate chunk payloads into one row per document (fine for small/medium knowledge bases)."""
        summary: dict[str, dict] = {}
        offset = None
        with self._lock:
            while True:
                points, offset = self.client.scroll(
                    collection_name=self.collection,
                    scroll_filter=self._doc_filter(doc_id) if doc_id else None,
                    limit=256, offset=offset, with_payload=True, with_vectors=False,
                )
                for p in points:
                    meta = (p.payload or {}).get("metadata", {})
                    did = meta.get("doc_id")
                    if not did:
                        continue
                    row = summary.setdefault(did, {
                        "doc_id": did, "title": meta.get("title"), "source": meta.get("source"),
                        "file_type": meta.get("file_type"), "added_at": meta.get("added_at"),
                        "chunks": 0, "pages": set(), "ocr": False,
                    })
                    row["chunks"] += 1
                    if meta.get("page"):
                        row["pages"].add(meta["page"])
                    row["ocr"] = row["ocr"] or bool(meta.get("ocr"))
                if offset is None:
                    break
        rows = []
        for row in summary.values():
            pages = row.pop("pages")
            row["pages"] = max(pages) if pages else None
            rows.append(row)
        return sorted(rows, key=lambda r: (r.get("added_at") or "", r.get("title") or ""))


_kb: KnowledgeBase | None = None
_kb_lock = threading.Lock()


def kb_ready() -> bool:
    """True once the knowledge base (and its embedding models) finished loading."""
    return _kb is not None


def get_kb() -> KnowledgeBase:
    global _kb
    if _kb is None:
        with _kb_lock:
            if _kb is None:
                _kb = KnowledgeBase()
    return _kb
