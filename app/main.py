"""FastAPI entrypoint. Interactive API docs at /docs, a minimal test UI at /."""

from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse

from .agent import graph as agent
from .config import settings
from .ingestion.pipeline import IngestResult, ingest_bytes, ingest_text, ingest_url, seed_directory
from .llm import LLMNotConfigured
from .schemas import AskRequest, AskResponse, DocumentInfo, HealthResponse, IngestItem, IngestResponse
from .vectorstore import get_kb, kb_ready
from .websearch import provider_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

STATIC_DIR = Path(__file__).parent / "static"
_seed_state = {"status": "pending"}


def _startup_tasks() -> None:
    """Load embedding models and seed the sample corpus without blocking server start."""
    try:
        _seed_state["status"] = "loading models"
        kb = get_kb()
        if not settings.seed_on_startup:
            _seed_state["status"] = "disabled"
            return
        if kb.count() > 0:
            _seed_state["status"] = "skipped (knowledge base already has documents)"
            return
        if not Path(settings.seed_dir).is_dir():
            _seed_state["status"] = f"skipped (no folder {settings.seed_dir})"
            return
        _seed_state["status"] = "ingesting sample corpus"
        results = seed_directory(kb, settings.seed_dir)
        added = sum(r.status == "added" for r in results)
        failed = [f"{r.source}: {r.error}" for r in results if r.status == "failed"]
        _seed_state["status"] = f"done ({added} added" + (f", failed: {failed})" if failed else ")")
    except Exception as exc:  # noqa: BLE001
        log.exception("startup seeding failed")
        _seed_state["status"] = f"failed: {type(exc).__name__}: {exc}"


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=_startup_tasks, name="startup-seed", daemon=True).start()
    yield


app = FastAPI(
    title="Multi-Step Research Assistant API",
    version="1.0.0",
    description=(
        "Answers questions over a document knowledge base with a LangGraph agent: query analysis and "
        "decomposition, hybrid retrieval, evidence grading with query rewriting, web fallback, cited synthesis "
        "and a fact-check pass. Add documents with `POST /documents`, ask with `POST /ask`."
    ),
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# --------------------------------------------------------------------------------------
# UI + health
# --------------------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    backend = documents = chunks = None
    try:
        if kb_ready():
            kb = get_kb()
            backend = kb.backend
            chunks = kb.count()
            documents = len(kb.titles())
    except Exception as exc:  # noqa: BLE001
        backend = f"error: {exc}"
    return HealthResponse(
        status="ok",
        llm_configured=settings.llm_configured,
        llm_models={"main": [settings.llm_model, *settings.llm_fallback_models],
                    "fast": [settings.fast_llm_model, *settings.fast_llm_fallback_models]},
        vector_store=backend,
        documents=documents,
        chunks=chunks,
        web_search=provider_name(),
        ocr_available=settings.ocr_available,
        seeding=_seed_state["status"],
    )


@app.get("/graph", response_class=PlainTextResponse, tags=["system"],
         summary="Mermaid diagram of the agent graph")
def graph_diagram() -> str:
    return agent.mermaid_diagram()


# --------------------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------------------
def _split_urls(values: list[str] | None) -> list[str]:
    urls: list[str] = []
    for v in values or []:
        urls.extend(u for u in re.split(r"[\s,]+", v or "") if u)
    return urls


@app.post("/documents", response_model=IngestResponse, tags=["documents"],
          summary="Add documents (any mix of files, URLs and raw text) through one ingestion path")
def add_documents(
    files: list[UploadFile] | None = File(default=None, description="PDF, DOCX, HTML, MD, TXT, CSV/TSV, JSON, PNG/JPG/TIFF (OCR)"),
    urls: list[str] | None = Form(default=None, description="Web pages or direct links to files"),
    text: str | None = Form(default=None, description="Raw text to add as a document"),
    title: str | None = Form(default=None, description="Title for the raw text document"),
) -> IngestResponse:
    kb = get_kb()
    results: list[IngestResult] = []
    for upload in files or []:
        if not getattr(upload, "filename", None):
            continue
        data = upload.file.read()
        results.append(ingest_bytes(kb, data, filename=upload.filename, content_type=upload.content_type))
    for url in _split_urls(urls):
        results.append(ingest_url(kb, url))
    if text and text.strip():
        results.append(ingest_text(kb, text, title))
    if not results:
        raise HTTPException(status_code=400, detail="Provide at least one of: files, urls, text")
    items = [IngestItem(**r.__dict__) for r in results]
    return IngestResponse(
        results=items,
        added=sum(i.status == "added" for i in items),
        duplicates=sum(i.status == "duplicate" for i in items),
        failed=sum(i.status == "failed" for i in items),
    )


@app.get("/documents", response_model=list[DocumentInfo], tags=["documents"])
def list_documents() -> list[DocumentInfo]:
    return [DocumentInfo(**d) for d in get_kb().list_documents()]


@app.delete("/documents/{doc_id}", tags=["documents"])
def delete_document(doc_id: str) -> dict:
    if not get_kb().delete_document(doc_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": doc_id}


# --------------------------------------------------------------------------------------
# Questions
# --------------------------------------------------------------------------------------
def _response(state: dict, conversation_id: str, question: str, mode: str) -> AskResponse:
    return AskResponse(
        conversation_id=conversation_id,
        question=question,
        standalone_question=state.get("standalone_question") or question,
        answer=state.get("answer", ""),
        answer_type=state.get("answer_type", "answer"),
        confidence=state.get("confidence", "low"),
        grounding=state.get("grounding", "none"),
        assumptions=state.get("assumptions", ""),
        sources=state.get("sources", []),
        sub_questions=[
            {k: sq.get(k) for k in ("id", "question", "origin", "status", "finding", "queries_tried", "used_web")}
            for sq in state.get("sub_questions", [])
        ],
        reasoning_trace=state.get("trace", []),
        mode=mode,
    )


@app.post("/ask", response_model=AskResponse, tags=["questions"],
          summary="Ask a question; returns the answer, cited sources, sub-questions and the reasoning trace")
def ask(req: AskRequest) -> AskResponse:
    if not settings.llm_configured:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys")
    conversation_id = req.conversation_id or str(uuid.uuid4())
    try:
        if req.mode == "naive":
            state = agent.run_naive(req.question)
        else:
            state = agent.run_agent(req.question, conversation_id, use_web=req.use_web)
    except LLMNotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _response(state, conversation_id, req.question, req.mode)


@app.post("/ask/stream", tags=["questions"],
          summary="Same as /ask but streams reasoning steps as NDJSON, ending with the full response")
def ask_stream(req: AskRequest) -> StreamingResponse:
    if not settings.llm_configured:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys")
    conversation_id = req.conversation_id or str(uuid.uuid4())

    def events():
        try:
            for event in agent.stream_agent(req.question, conversation_id, use_web=req.use_web):
                if event["event"] == "final":
                    body = _response(event["state"], conversation_id, req.question, "agent")
                    yield json.dumps({"event": "final", "response": jsonable_encoder(body)}) + "\n"
                else:
                    yield json.dumps(jsonable_encoder(event)) + "\n"
        except Exception as exc:  # noqa: BLE001
            log.exception("stream failed")
            yield json.dumps({"event": "error", "detail": f"{type(exc).__name__}: {exc}"}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.get("/conversations/{conversation_id}", tags=["questions"])
def get_conversation(conversation_id: str) -> dict:
    return {"conversation_id": conversation_id, "turns": agent.conversation_history(conversation_id)}


@app.delete("/conversations/{conversation_id}", tags=["questions"])
def clear_conversation(conversation_id: str) -> dict:
    return {"conversation_id": conversation_id, "cleared": agent.delete_conversation(conversation_id)}
