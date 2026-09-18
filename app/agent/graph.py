"""Builds the LangGraph research agent and exposes run / stream helpers."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

try:  # langgraph >= 0.2 names it InMemorySaver; MemorySaver is the older alias
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:  # pragma: no cover
    from langgraph.checkpoint.memory import MemorySaver as InMemorySaver

from ..config import settings
from ..llm import clean_text, text_llm
from ..vectorstore import get_kb
from . import nodes, prompts
from .state import AgentState, new_turn

log = logging.getLogger(__name__)

_checkpointer = InMemorySaver()
_thread_locks: dict[str, threading.Lock] = {}
_thread_locks_guard = threading.Lock()


@lru_cache(maxsize=1)
def build_graph():
    g = StateGraph(AgentState)

    g.add_node("analyze", nodes.analyze)
    g.add_node("respond", nodes.respond)
    g.add_node("prepare_step", nodes.prepare_step)
    g.add_node("retrieve_kb", nodes.retrieve_kb)
    g.add_node("grade_evidence", nodes.grade_evidence)
    g.add_node("rewrite_query", nodes.rewrite_query)
    g.add_node("web_search", nodes.web_search)
    g.add_node("record_step", nodes.record_step)
    g.add_node("synthesize", nodes.synthesize)
    g.add_node("verify", nodes.verify)
    g.add_node("plan_followups", nodes.plan_followups)
    g.add_node("no_answer", nodes.no_answer)
    g.add_node("finalize", nodes.finalize)

    g.add_edge(START, "analyze")
    g.add_conditional_edges("analyze", nodes.route_after_analyze,
                            {"respond": "respond", "prepare_step": "prepare_step"})
    g.add_edge("respond", "finalize")

    g.add_edge("prepare_step", "retrieve_kb")
    g.add_edge("retrieve_kb", "grade_evidence")
    g.add_conditional_edges("grade_evidence", nodes.route_after_grade,
                            {"record_step": "record_step", "rewrite_query": "rewrite_query", "web_search": "web_search"})
    g.add_edge("rewrite_query", "retrieve_kb")
    g.add_edge("web_search", "grade_evidence")
    g.add_conditional_edges("record_step", nodes.route_after_record,
                            {"prepare_step": "prepare_step", "synthesize": "synthesize", "no_answer": "no_answer"})

    g.add_edge("synthesize", "verify")
    g.add_conditional_edges("verify", nodes.route_after_verify,
                            {"synthesize": "synthesize", "plan_followups": "plan_followups", "finalize": "finalize"})
    g.add_edge("plan_followups", "prepare_step")
    g.add_edge("no_answer", "finalize")
    g.add_edge("finalize", END)

    return g.compile(checkpointer=_checkpointer)


def _config(conversation_id: str) -> dict:
    return {"configurable": {"thread_id": conversation_id}, "recursion_limit": 120}


def _lock_for(conversation_id: str) -> threading.Lock:
    with _thread_locks_guard:
        return _thread_locks.setdefault(conversation_id, threading.Lock())


def run_agent(question: str, conversation_id: str, use_web: bool = True) -> dict:
    graph = build_graph()
    with _lock_for(conversation_id):  # turns in one conversation must not interleave
        return graph.invoke(new_turn(question, use_web), _config(conversation_id))


def stream_agent(question: str, conversation_id: str, use_web: bool = True) -> Iterator[dict]:
    """Yields {"event": "step", ...} for every new trace entry, then {"event": "final", "state": ...}."""
    graph = build_graph()
    with _lock_for(conversation_id):
        seen = 0
        for update in graph.stream(new_turn(question, use_web), _config(conversation_id), stream_mode="updates"):
            for node_name, delta in update.items():
                trace = (delta or {}).get("trace") if isinstance(delta, dict) else None
                if trace and len(trace) > seen:
                    for entry in trace[seen:]:
                        yield {"event": "step", "node": node_name, **entry}
                    seen = len(trace)
        yield {"event": "final", "state": graph.get_state(_config(conversation_id)).values}


def conversation_history(conversation_id: str) -> list[dict]:
    snapshot = build_graph().get_state(_config(conversation_id))
    return list((snapshot.values or {}).get("history", [])) if snapshot else []


def delete_conversation(conversation_id: str) -> bool:
    try:
        _checkpointer.delete_thread(conversation_id)
        return True
    except Exception as exc:  # older checkpointer versions
        log.warning("could not delete thread %s: %s", conversation_id, exc)
        return False


def mermaid_diagram() -> str:
    return build_graph().get_graph().draw_mermaid()


# --------------------------------------------------------------------------------------
# Naive single-pass baseline, exposed only so the difference can be measured
# --------------------------------------------------------------------------------------
def run_naive(question: str) -> dict:
    hits = get_kb().search(question, k=5)
    context_items = []
    sources = []
    for n, (doc, _score) in enumerate(hits, start=1):
        meta = doc.metadata or {}
        body = doc.page_content.split("\n", 1)[1] if doc.page_content.startswith("Document:") else doc.page_content
        context_items.append(f"[{n}] {body[: settings.passage_chars]}")
        sources.append({"n": n, "type": "kb", "cited": False, "title": meta.get("title"), "source": meta.get("source"),
                        "url": None, "doc_id": meta.get("doc_id"), "chunk_id": str(meta.get("_id")),
                        "page": meta.get("page"), "section": meta.get("section"), "snippet": body[:400]})
    msg = (prompts.NAIVE | text_llm("main")).invoke({"context": "\n\n".join(context_items) or "(empty)", "question": question})
    return {
        "answer": clean_text(msg.content),
        "answer_type": "answer",
        "confidence": "unknown",
        "grounding": "knowledge_base" if sources else "none",
        "sources": sources,
        "sub_questions": [],
        "trace": [{"step": "naive", "detail": "Single retrieval (top 5) + single LLM call, no planning or checks."}],
        "standalone_question": question,
        "assumptions": "",
    }
