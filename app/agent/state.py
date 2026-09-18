"""LangGraph state for the research agent.

Only `history` accumulates across turns (via a reducer). Every other field is per-turn and is reset
explicitly at the start of each question, because the checkpointer keeps the previous turn's values.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class Evidence(TypedDict, total=False):
    id: str              # "kb:<point id>" or "web:<url hash>"
    type: str            # "kb" | "web"
    title: str
    source: str          # file name / URL
    url: str | None
    doc_id: str | None
    page: int | None
    section: str | None
    text: str
    score: float | None


class SubQuestion(TypedDict, total=False):
    id: int
    question: str
    search_query: str
    depends_on_previous: bool
    origin: str          # "plan" | "follow-up"
    status: str          # pending | answered | partial | not_found
    finding: str
    missing: str
    queries_tried: list[str]
    evidence_ids: list[str]
    used_web: bool


class AgentState(TypedDict, total=False):
    # input
    question: str
    use_web: bool

    # conversation memory (persists across turns through the checkpointer)
    history: Annotated[list[dict], operator.add]

    # query understanding
    standalone_question: str
    intent: str          # research | clarify | direct
    assumptions: str
    reply: str

    # research loop
    sub_questions: list[SubQuestion]
    current: int
    current_query: str
    kb_attempts: int
    web_tried: bool
    web_error: str
    candidates: list[Evidence]
    last_grade: dict
    evidence: dict[str, Evidence]

    # answer + verification
    evidence_order: list[str]
    draft: str
    feedback: str
    revisions: int
    followup_rounds: int
    verification: dict

    # output
    answer: str
    answer_type: str     # answer | clarification | direct | not_found
    confidence: str      # high | medium | low
    grounding: str       # knowledge_base | web | mixed | none
    sources: list[dict]
    trace: list[dict]


def new_turn(question: str, use_web: bool) -> dict:
    """Input for a new turn: resets every per-turn field (history is left to its reducer)."""
    return {
        "question": question,
        "use_web": use_web,
        "standalone_question": question,
        "intent": "research",
        "assumptions": "",
        "reply": "",
        "sub_questions": [],
        "current": 0,
        "current_query": "",
        "kb_attempts": 0,
        "web_tried": False,
        "web_error": "",
        "candidates": [],
        "last_grade": {},
        "evidence": {},
        "evidence_order": [],
        "draft": "",
        "feedback": "",
        "revisions": 0,
        "followup_rounds": 0,
        "verification": {},
        "answer": "",
        "answer_type": "answer",
        "confidence": "low",
        "grounding": "none",
        "sources": [],
        "trace": [],
    }
