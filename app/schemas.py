from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000, examples=[
        "Which country is the supplier of Atlas's lidar based in, and when does that contract end?"])
    conversation_id: str | None = Field(default=None, description="Reuse to ask follow-up questions; omitted = new conversation")
    use_web: bool = Field(default=True, description="Allow web search when the knowledge base is not enough")
    mode: Literal["agent", "naive"] = Field(default="agent", description="'naive' = single retrieve+generate baseline, for comparison")


class Source(BaseModel):
    n: int = Field(description="Citation number used in the answer text, e.g. [2]")
    type: Literal["kb", "web"]
    cited: bool = Field(description="False when the answer had no explicit citation and all evidence is listed")
    title: str | None = None
    source: str | None = Field(default=None, description="File name, URL or text label")
    url: str | None = None
    doc_id: str | None = None
    chunk_id: str | None = None
    page: int | None = None
    section: str | None = None
    snippet: str


class SubQuestionOut(BaseModel):
    id: int
    question: str
    origin: str = "plan"
    status: str
    finding: str = ""
    queries_tried: list[str] = []
    used_web: bool = False


class TraceStep(BaseModel):
    step: str
    detail: str
    data: dict | None = None


class AskResponse(BaseModel):
    conversation_id: str
    question: str
    standalone_question: str
    answer: str
    answer_type: Literal["answer", "clarification", "direct", "not_found"]
    confidence: str
    grounding: Literal["knowledge_base", "web", "mixed", "none"]
    assumptions: str = ""
    sources: list[Source]
    sub_questions: list[SubQuestionOut]
    reasoning_trace: list[TraceStep]
    mode: str = "agent"


class IngestItem(BaseModel):
    status: Literal["added", "duplicate", "failed"]
    source: str
    doc_id: str | None = None
    title: str | None = None
    file_type: str | None = None
    chunks: int = 0
    ocr_pages: list[int] = []
    error: str | None = None


class IngestResponse(BaseModel):
    results: list[IngestItem]
    added: int
    duplicates: int
    failed: int


class DocumentInfo(BaseModel):
    doc_id: str
    title: str | None
    source: str | None
    file_type: str | None
    chunks: int
    pages: int | None = None
    ocr: bool = False
    added_at: str | None = None


class HealthResponse(BaseModel):
    status: str
    llm_configured: bool
    llm_models: dict
    vector_store: str | None
    documents: int | None
    chunks: int | None
    web_search: str
    ocr_available: bool
    seeding: str
