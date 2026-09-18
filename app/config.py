"""All configuration comes from environment variables (optionally loaded from a .env file)."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def _str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if raw is None:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    # ---- LLM (Groq free tier) ----------------------------------------------------------
    # "main" model: query analysis + final answer. "fast" model: evidence grading + fact-checking.
    # Groq rate limits are per model, so fallbacks to other models also act as extra free capacity.
    groq_api_key: str = field(default_factory=lambda: _str("GROQ_API_KEY"))
    llm_model: str = field(default_factory=lambda: _str("LLM_MODEL", "openai/gpt-oss-120b"))
    llm_fallback_models: list[str] = field(
        default_factory=lambda: _list("LLM_FALLBACK_MODELS", ["llama-3.3-70b-versatile", "openai/gpt-oss-20b"])
    )
    fast_llm_model: str = field(default_factory=lambda: _str("FAST_LLM_MODEL", "openai/gpt-oss-20b"))
    fast_llm_fallback_models: list[str] = field(
        default_factory=lambda: _list("FAST_LLM_FALLBACK_MODELS", ["llama-3.3-70b-versatile", "openai/gpt-oss-120b"])
    )
    main_max_tokens: int = field(default_factory=lambda: _int("MAIN_MAX_TOKENS", 2500))
    fast_max_tokens: int = field(default_factory=lambda: _int("FAST_MAX_TOKENS", 1500))
    llm_max_retries: int = field(default_factory=lambda: _int("LLM_MAX_RETRIES", 2))
    llm_timeout_s: int = field(default_factory=lambda: _int("LLM_TIMEOUT_S", 60))

    # ---- Vector store: Qdrant embedded on disk by default, Qdrant Cloud when QDRANT_URL is set ----
    qdrant_url: str = field(default_factory=lambda: _str("QDRANT_URL"))
    qdrant_api_key: str = field(default_factory=lambda: _str("QDRANT_API_KEY"))
    qdrant_path: str = field(default_factory=lambda: _str("QDRANT_PATH", str(BASE_DIR / "storage" / "qdrant")))
    collection_name: str = field(default_factory=lambda: _str("COLLECTION_NAME", "knowledge_base"))

    # ---- Embeddings: local FastEmbed ONNX models, no API key needed ----
    embedding_model: str = field(default_factory=lambda: _str("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
    sparse_model: str = field(default_factory=lambda: _str("SPARSE_MODEL", "Qdrant/bm25"))
    model_cache_dir: str = field(default_factory=lambda: _str("MODEL_CACHE_DIR", str(BASE_DIR / "models")))

    # ---- Ingestion ----
    chunk_size: int = field(default_factory=lambda: _int("CHUNK_SIZE", 900))
    chunk_overlap: int = field(default_factory=lambda: _int("CHUNK_OVERLAP", 150))
    max_upload_mb: int = field(default_factory=lambda: _int("MAX_UPLOAD_MB", 25))
    seed_on_startup: bool = field(default_factory=lambda: _bool("SEED_ON_STARTUP", True))
    seed_dir: str = field(default_factory=lambda: _str("SEED_DIR", str(BASE_DIR / "data" / "corpus")))
    allow_private_urls: bool = field(default_factory=lambda: _bool("ALLOW_PRIVATE_URLS", False))

    # ---- Agent behaviour ----
    retrieval_k: int = field(default_factory=lambda: _int("RETRIEVAL_K", 6))
    max_candidates: int = field(default_factory=lambda: _int("MAX_CANDIDATES", 8))
    max_sub_questions: int = field(default_factory=lambda: _int("MAX_SUB_QUESTIONS", 3))
    max_kb_attempts: int = field(default_factory=lambda: _int("MAX_KB_ATTEMPTS", 2))
    max_revisions: int = field(default_factory=lambda: _int("MAX_REVISIONS", 1))
    max_followup_rounds: int = field(default_factory=lambda: _int("MAX_FOLLOWUP_ROUNDS", 1))
    history_turns: int = field(default_factory=lambda: _int("HISTORY_TURNS", 4))
    passage_chars: int = field(default_factory=lambda: _int("PASSAGE_CHARS", 800))

    # ---- Web fallback ----
    # auto = Tavily if TAVILY_API_KEY is set, otherwise DuckDuckGo (no key). "none" disables it.
    web_search_provider: str = field(default_factory=lambda: _str("WEB_SEARCH_PROVIDER", "auto").lower())
    tavily_api_key: str = field(default_factory=lambda: _str("TAVILY_API_KEY"))
    web_results: int = field(default_factory=lambda: _int("WEB_RESULTS", 5))

    @property
    def ocr_available(self) -> bool:
        return shutil.which("tesseract") is not None

    @property
    def llm_configured(self) -> bool:
        return bool(self.groq_api_key)


settings = Settings()
