"""Nodes of the research graph. Each node does one job and records a trace entry.

analyze -> (clarify/direct -> respond) | research loop:
  prepare_step -> retrieve_kb -> grade_evidence -> [rewrite_query -> retrieve_kb]* -> [web_search -> grade_evidence]
  -> record_step -> next sub-question ... -> synthesize -> verify -> [synthesize (revision) | plan_followups] -> finalize
"""

from __future__ import annotations

import hashlib
import logging
import re

from ..config import settings
from ..llm import clean_text, structured_llm, text_llm
from ..vectorstore import get_kb
from ..websearch import search_web, web_enabled
from . import prompts
from .state import AgentState, Evidence, SubQuestion

log = logging.getLogger(__name__)

CITATION_RE = re.compile(r"\[(\d{1,3})\]")
MAX_EVIDENCE_FOR_ANSWER = 12  # keeps synthesis/verify prompts well under Groq free-tier TPM


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _trace(state: AgentState, step: str, detail: str, **data) -> list[dict]:
    entry: dict = {"step": step, "detail": detail}
    if data:
        entry["data"] = data
    log.info("[%s] %s", step, detail)
    return [*state.get("trace", []), entry]


def _body(ev: Evidence) -> str:
    """Chunk text without the contextual header line added at ingestion."""
    text = ev.get("text", "")
    if ev.get("type") == "kb" and text.startswith("Document:"):
        text = text.split("\n", 1)[1] if "\n" in text else text
    return text.strip()


def _label(ev: Evidence) -> str:
    if ev.get("type") == "web":
        return f"WEB | {ev.get('title') or ev.get('url')} | {ev.get('url')}"
    parts = [f"KB | {ev.get('title') or ev.get('source')}"]
    if ev.get("section"):
        parts.append(f"section: {ev['section']}")
    if ev.get("page"):
        parts.append(f"page {ev['page']}")
    return " | ".join(parts)


def _format_passages(items: list[Evidence]) -> str:
    limit = settings.passage_chars
    return "\n\n".join(f"[{i}] ({_label(ev)})\n{_body(ev)[:limit]}" for i, ev in enumerate(items, start=1))


def _format_history(history: list[dict]) -> str:
    turns = history[-settings.history_turns:]
    if not turns:
        return "(no earlier messages)"
    lines = []
    for t in turns:
        lines.append(f"User: {t.get('question', '')}")
        answer = (t.get("answer") or "").strip()
        if len(answer) > 700:
            answer = answer[:700] + " ..."
        lines.append(f"Assistant: {answer}")
        if t.get("source_titles"):
            lines.append(f"(sources used: {', '.join(t['source_titles'])})")
    return "\n".join(lines)


def _current(state: AgentState) -> tuple[int, SubQuestion, list[SubQuestion]]:
    subs = [dict(s) for s in state.get("sub_questions", [])]
    idx = state.get("current", 0)
    return idx, subs[idx], subs


def _kb_titles() -> str:
    try:
        titles = get_kb().titles()
    except Exception:
        titles = []
    return "; ".join(titles[:40]) if titles else "(knowledge base is empty)"


# --------------------------------------------------------------------------------------
# 1. analyze
# --------------------------------------------------------------------------------------
def analyze(state: AgentState) -> dict:
    question = state["question"]
    max_sub = settings.max_sub_questions
    result: prompts.QueryAnalysis | None = None
    try:
        result = (prompts.ANALYZE | structured_llm("main", prompts.QueryAnalysis)).invoke({
            "question": question,
            "history": _format_history(state.get("history", [])),
            "kb_titles": _kb_titles(),
            "max_sub": max_sub,
        })
    except Exception as exc:
        log.warning("analyze failed, using the raw question: %s", exc)

    if result is None:
        result = prompts.QueryAnalysis(intent="research", standalone_question=question,
                                       sub_questions=[prompts.PlannedSubQuestion(question=question, search_query=question)])

    standalone = result.standalone_question.strip() or question
    planned = result.sub_questions[:max_sub] or [prompts.PlannedSubQuestion(question=standalone, search_query=standalone)]
    subs: list[SubQuestion] = []
    for i, p in enumerate(planned):
        subs.append({
            "id": i + 1, "question": p.question.strip() or standalone,
            "search_query": p.search_query.strip() or p.question.strip() or standalone,
            "depends_on_previous": bool(p.depends_on_previous and i > 0), "origin": "plan",
            "status": "pending", "finding": "", "missing": "", "queries_tried": [], "evidence_ids": [], "used_web": False,
        })

    intent = result.intent
    if intent in {"clarify", "direct"} and not result.reply.strip():
        intent = "research"  # nothing to say without research

    if intent == "research":
        detail = f"Interpreted as: {standalone}. Planned {len(subs)} sub-question(s)."
        if result.assumptions:
            detail += f" Assumption: {result.assumptions}"
    else:
        detail = f"Intent: {intent}."
    return {
        "standalone_question": standalone,
        "intent": intent,
        "assumptions": result.assumptions.strip(),
        "reply": result.reply.strip(),
        "sub_questions": subs if intent == "research" else [],
        "current": 0,
        "trace": _trace(state, "analyze", detail, intent=intent,
                        sub_questions=[{"q": s["question"], "depends_on_previous": s["depends_on_previous"]} for s in subs]
                        if intent == "research" else []),
    }


def route_after_analyze(state: AgentState) -> str:
    return "respond" if state.get("intent") in {"clarify", "direct"} else "prepare_step"


def respond(state: AgentState) -> dict:
    kind = "clarification" if state.get("intent") == "clarify" else "direct"
    return {
        "answer": state.get("reply", ""),
        "answer_type": kind,
        "confidence": "high" if kind == "direct" else "low",
        "trace": _trace(state, "respond", "Asked a clarifying question." if kind == "clarification" else "Replied directly."),
    }


# --------------------------------------------------------------------------------------
# 2. research loop
# --------------------------------------------------------------------------------------
def prepare_step(state: AgentState) -> dict:
    idx, sq, subs = _current(state)
    detail = f"Step {sq['id']}: {sq['question']}"
    if sq.get("depends_on_previous"):
        findings = "\n".join(
            f"- {s['question']} -> {s['finding'] or 'not found'}" for s in subs[:idx] if s.get("status") != "pending"
        )
        if findings:
            try:
                step = (prompts.REWRITE_STEP | structured_llm("fast", prompts.StepQuery)).invoke({
                    "question": state.get("standalone_question", state["question"]),
                    "findings": findings, "sub_question": sq["question"],
                })
                if step and step.question.strip():
                    sq["question"] = step.question.strip()
                    sq["search_query"] = (step.search_query or step.question).strip()
                    detail = f"Step {sq['id']} (uses earlier findings): {sq['question']}"
            except Exception as exc:
                log.warning("dependent-step rewrite failed: %s", exc)
    sq["status"] = "in_progress"
    subs[idx] = sq
    return {
        "sub_questions": subs,
        "current_query": sq["search_query"],
        "kb_attempts": 0,
        "web_tried": False,
        "web_error": "",
        "candidates": [],
        "last_grade": {},
        "trace": _trace(state, "plan_step", detail),
    }


def retrieve_kb(state: AgentState) -> dict:
    idx, sq, subs = _current(state)
    attempt = state.get("kb_attempts", 0) + 1
    queries = [state.get("current_query") or sq["question"]]
    if attempt == 1 and sq["question"].strip().lower() != queries[0].strip().lower():
        queries.append(sq["question"])  # keyword query + natural-language question

    kept = set(sq.get("evidence_ids", []))
    fused: dict[str, float] = {}
    found: dict[str, Evidence] = {}
    error = None
    try:
        kb = get_kb()
        for q in queries:
            for rank, (doc, score) in enumerate(kb.search(q, k=settings.retrieval_k)):
                meta = doc.metadata or {}
                point_id = meta.get("_id") or f"{meta.get('doc_id')}-{meta.get('chunk_index')}"
                eid = f"kb:{point_id}"
                if eid in kept:
                    continue
                fused[eid] = fused.get(eid, 0.0) + 1.0 / (60 + rank)  # reciprocal rank fusion across queries
                found.setdefault(eid, {
                    "id": eid, "type": "kb", "title": meta.get("title"), "source": meta.get("source"), "url": None,
                    "doc_id": meta.get("doc_id"), "page": meta.get("page"), "section": meta.get("section"),
                    "text": doc.page_content, "score": float(score) if score is not None else None,
                })
    except Exception as exc:
        log.exception("knowledge base search failed")
        error = f"{type(exc).__name__}: {exc}"

    ranked = sorted(found.values(), key=lambda e: fused[e["id"]], reverse=True)[: settings.max_candidates]
    sq["queries_tried"] = [*sq.get("queries_tried", []), *queries]
    subs[idx] = sq
    titles = sorted({e.get("title") or "" for e in ranked})
    detail = (f"Search {attempt} for '{queries[0]}': {len(ranked)} new passage(s) from {len(titles)} document(s)."
              if not error else f"Search failed: {error}")
    return {
        "sub_questions": subs,
        "candidates": ranked,
        "kb_attempts": attempt,
        "trace": _trace(state, "retrieve", detail, queries=queries, documents=titles),
    }


def web_search(state: AgentState) -> dict:
    idx, sq, subs = _current(state)
    query = sq["question"]
    results, error = search_web(query)
    candidates: list[Evidence] = [
        {
            "id": "web:" + hashlib.sha1(r.url.encode()).hexdigest()[:12], "type": "web", "title": r.title,
            "source": r.url, "url": r.url, "doc_id": None, "page": None, "section": None,
            "text": r.content[:2000], "score": None,
        }
        for r in results
    ]
    sq["queries_tried"] = [*sq.get("queries_tried", []), f"web: {query}"]
    subs[idx] = sq
    detail = (f"Knowledge base was not enough; searched the web: {len(candidates)} result(s)."
              if not error else f"Web search returned nothing ({error}).")
    return {
        "sub_questions": subs,
        "candidates": candidates,
        "web_tried": True,
        "web_error": error or "",
        "trace": _trace(state, "web_search", detail, urls=[c["url"] for c in candidates]),
    }


def grade_evidence(state: AgentState) -> dict:
    idx, sq, subs = _current(state)
    evidence = dict(state.get("evidence", {}))
    kept = [evidence[e] for e in sq.get("evidence_ids", []) if e in evidence]
    candidates = list(state.get("candidates", []))
    passages = kept + candidates

    if not passages:
        grade = {"sufficient": False, "refined_query": "", "relevant": 0, "missing": "no passages found"}
        return {"last_grade": grade, "trace": _trace(state, "grade", "Nothing to grade: no passages found.")}

    result: prompts.EvidenceGrade | None = None
    try:
        result = (prompts.GRADE | structured_llm("fast", prompts.EvidenceGrade)).invoke({
            "question": state.get("standalone_question", state["question"]),
            "sub_question": sq["question"],
            "passages": _format_passages(passages),
        })
    except Exception as exc:
        log.warning("grading failed: %s", exc)

    if result is None:
        # Grader unavailable: accept the top passages and let synthesis + verification be the safety net.
        chosen = list(range(1, min(len(passages), 4) + 1))
        result = prompts.EvidenceGrade(relevant_passages=chosen, sufficient=True, finding="")
        note = " (grader unavailable; kept top passages)"
    else:
        note = ""

    relevant_idx = sorted({i for i in result.relevant_passages if 1 <= i <= len(passages)})
    ids = list(sq.get("evidence_ids", []))
    for i in relevant_idx:
        ev = passages[i - 1]
        evidence[ev["id"]] = ev
        if ev["id"] not in ids:
            ids.append(ev["id"])
    sq["evidence_ids"] = ids
    if result.finding.strip():
        sq["finding"] = result.finding.strip()
    sq["missing"] = result.missing.strip()
    sufficient = bool(result.sufficient and ids)
    subs[idx] = sq

    verdict = "enough to answer" if sufficient else f"not enough ({result.missing or 'answer not stated'})"
    detail = f"{len(relevant_idx)} of {len(passages)} passage(s) relevant; {verdict}.{note}"
    return {
        "sub_questions": subs,
        "evidence": evidence,
        "last_grade": {"sufficient": sufficient, "refined_query": result.refined_query.strip(),
                       "relevant": len(relevant_idx), "missing": result.missing.strip()},
        "trace": _trace(state, "grade", detail),
    }


def route_after_grade(state: AgentState) -> str:
    grade = state.get("last_grade", {})
    if grade.get("sufficient"):
        return "record_step"
    if not state.get("web_tried") and state.get("kb_attempts", 0) < settings.max_kb_attempts:
        return "rewrite_query"
    if state.get("use_web", True) and web_enabled() and not state.get("web_tried"):
        return "web_search"
    return "record_step"


def rewrite_query(state: AgentState) -> dict:
    idx, sq, _ = _current(state)
    grade = state.get("last_grade", {})
    tried = {q.lower() for q in sq.get("queries_tried", [])}
    query = grade.get("refined_query", "")
    if not query or query.lower() in tried:
        missing = grade.get("missing", "")
        query = f"{sq['question']} {missing}".strip() if missing else f"{sq['question']} details"
    return {
        "current_query": query,
        "trace": _trace(state, "rewrite_query", f"Evidence incomplete; trying a different search: '{query}'."),
    }


def record_step(state: AgentState) -> dict:
    idx, sq, subs = _current(state)
    evidence = state.get("evidence", {})
    ids = sq.get("evidence_ids", [])
    if state.get("last_grade", {}).get("sufficient"):
        status = "answered"
    elif ids:
        status = "partial"
    else:
        status = "not_found"
    sq["status"] = status
    sq["used_web"] = any(evidence.get(i, {}).get("type") == "web" for i in ids)
    subs[idx] = sq
    label = {"answered": "answered", "partial": "partly answered", "not_found": "not found"}[status]
    source = " using web results" if sq["used_web"] else ""
    return {
        "sub_questions": subs,
        "current": idx + 1,
        "trace": _trace(state, "record_step", f"Step {sq['id']} {label}{source}."),
    }


def route_after_record(state: AgentState) -> str:
    if state.get("current", 0) < len(state.get("sub_questions", [])):
        return "prepare_step"
    if not state.get("evidence"):
        return "no_answer"
    return "synthesize"


# --------------------------------------------------------------------------------------
# 3. answer, verify, follow up
# --------------------------------------------------------------------------------------
def _ordered_evidence(state: AgentState) -> list[str]:
    order: list[str] = []
    evidence = state.get("evidence", {})
    for sq in state.get("sub_questions", []):
        for eid in sq.get("evidence_ids", []):
            if eid in evidence and eid not in order:
                order.append(eid)
    return order[:MAX_EVIDENCE_FOR_ANSWER]


def synthesize(state: AgentState) -> dict:
    order = _ordered_evidence(state)
    evidence = state.get("evidence", {})
    items = [evidence[e] for e in order]
    notes = "\n".join(
        f"{sq['id']}. {sq['question']} [{sq.get('status')}] {sq.get('finding') or ''}".strip()
        for sq in state.get("sub_questions", [])
    )
    feedback = state.get("feedback", "")
    revisions = state.get("revisions", 0) + (1 if feedback else 0)
    feedback_block = (
        "\nA fact-checker reviewed your previous draft and found these claims unsupported by the evidence. "
        f"Remove or correct them:\n{feedback}\n" if feedback else ""
    )
    try:
        msg = (prompts.SYNTHESIZE | text_llm("main")).invoke({
            "question": state["question"],
            "standalone": state.get("standalone_question", state["question"]),
            "assumptions": state.get("assumptions") or "none",
            "notes": notes,
            "evidence": _format_passages(items),
            "feedback": feedback_block,
        })
        draft = clean_text(msg.content)
    except Exception as exc:
        log.exception("synthesis failed")
        draft = ("I found relevant material but could not generate an answer because the language model is "
                 f"unavailable ({type(exc).__name__}). Relevant findings:\n" +
                 "\n".join(f"- {sq['question']}: {sq.get('finding') or 'see sources'}" for sq in state.get("sub_questions", [])))
    detail = "Revised the answer after fact-checking." if feedback else f"Drafted an answer from {len(items)} evidence passage(s)."
    return {
        "evidence_order": order,
        "draft": draft,
        "revisions": revisions,
        "feedback": "",
        "trace": _trace(state, "synthesize", detail),
    }


def verify(state: AgentState) -> dict:
    evidence = state.get("evidence", {})
    items = [evidence[e] for e in state.get("evidence_order", [])]
    result: prompts.Verification | None = None
    try:
        result = (prompts.VERIFY | structured_llm("fast", prompts.Verification)).invoke({
            "question": state.get("standalone_question", state["question"]),
            "evidence": _format_passages(items),
            "draft": state.get("draft", ""),
        })
    except Exception as exc:
        log.warning("verification failed: %s", exc)

    if result is None:
        verification = {"unsupported_claims": [], "missing_aspects": [], "confidence": "medium", "checked": False}
        detail = "Fact-check unavailable; answer not independently verified."
    else:
        verification = {"unsupported_claims": result.unsupported_claims, "missing_aspects": result.missing_aspects,
                        "confidence": result.confidence, "checked": True}
        problems = []
        if result.unsupported_claims:
            problems.append(f"{len(result.unsupported_claims)} unsupported claim(s)")
        if result.missing_aspects:
            problems.append(f"{len(result.missing_aspects)} unaddressed part(s)")
        detail = "Fact-check passed: every claim is backed by the sources." if not problems else "Fact-check found " + " and ".join(problems) + "."
    feedback = "\n".join(f"- {c}" for c in verification["unsupported_claims"])
    return {"verification": verification, "feedback": feedback, "trace": _trace(state, "verify", detail, **verification)}


def route_after_verify(state: AgentState) -> str:
    v = state.get("verification", {})
    if v.get("unsupported_claims") and state.get("revisions", 0) < settings.max_revisions:
        return "synthesize"
    if (v.get("missing_aspects") and state.get("followup_rounds", 0) < settings.max_followup_rounds):
        return "plan_followups"
    return "finalize"


def plan_followups(state: AgentState) -> dict:
    subs = [dict(s) for s in state.get("sub_questions", [])]
    start = len(subs)
    for aspect in state.get("verification", {}).get("missing_aspects", [])[:2]:
        subs.append({
            "id": len(subs) + 1, "question": aspect, "search_query": aspect, "depends_on_previous": False,
            "origin": "follow-up", "status": "pending", "finding": "", "missing": "", "queries_tried": [],
            "evidence_ids": [], "used_web": False,
        })
    return {
        "sub_questions": subs,
        "current": start,
        "followup_rounds": state.get("followup_rounds", 0) + 1,
        "feedback": "",
        "trace": _trace(state, "plan_followups", f"Answer missed part of the question; researching {len(subs) - start} more step(s)."),
    }


def no_answer(state: AgentState) -> dict:
    subs = state.get("sub_questions", [])
    web_note = ""
    if state.get("web_tried"):
        web_note = " or in web search results"
    elif not (state.get("use_web", True) and web_enabled()):
        web_note = " (web search was not enabled)"
    tried = sorted({q for s in subs for q in s.get("queries_tried", [])})
    answer = (f"I could not find information to answer this in the knowledge base{web_note}. "
              f"I searched for: {'; '.join(tried[:6])}. "
              "If you have a document that covers it, add it and ask again.")
    return {
        "answer": answer,
        "answer_type": "not_found",
        "confidence": "low",
        "trace": _trace(state, "no_answer", "No supporting evidence found anywhere; not guessing."),
    }


def finalize(state: AgentState) -> dict:
    answer_type = state.get("answer_type", "answer")
    if state.get("intent") in {"clarify", "direct"} or answer_type == "not_found":
        answer = state.get("answer", "")
        sources: list[dict] = []
        confidence = state.get("confidence", "low")
        grounding = "none"
    else:
        answer = state.get("draft", "")
        evidence = state.get("evidence", {})
        order = state.get("evidence_order", [])
        cited = []
        for m in CITATION_RE.findall(answer):
            n = int(m)
            if 1 <= n <= len(order) and n not in cited:
                cited.append(n)
        numbers = sorted(cited) if cited else list(range(1, len(order) + 1))
        sources = []
        for n in numbers:
            ev = evidence[order[n - 1]]
            body = _body(ev)
            sources.append({
                "n": n, "type": ev.get("type"), "cited": bool(cited),
                "title": ev.get("title"), "source": ev.get("source"), "url": ev.get("url"),
                "doc_id": ev.get("doc_id"), "chunk_id": ev["id"].split(":", 1)[1] if ev.get("type") == "kb" else None,
                "page": ev.get("page"), "section": ev.get("section"),
                "snippet": body[:400] + ("..." if len(body) > 400 else ""),
            })
        types = {s["type"] for s in sources}
        grounding = "mixed" if types == {"kb", "web"} else ("web" if types == {"web"} else ("knowledge_base" if types else "none"))

        v = state.get("verification", {})
        confidence = v.get("confidence", "medium")
        if v.get("unsupported_claims"):
            confidence = "low"
            answer += ("\n\nNote: an automatic fact-check could not confirm these statements against the sources: "
                       + "; ".join(v["unsupported_claims"][:3]))
        elif grounding in {"web", "mixed"} and confidence == "high":
            confidence = "medium"
        if any(s.get("status") == "not_found" for s in state.get("sub_questions", [])) and confidence == "high":
            confidence = "medium"
        answer_type = "answer"

    history_entry = {
        "question": state["question"],
        "standalone_question": state.get("standalone_question", state["question"]),
        "answer": answer,
        "answer_type": answer_type,
        "source_titles": sorted({s["title"] or s["source"] for s in sources if s.get("title") or s.get("source")}),
    }
    return {
        "answer": answer,
        "answer_type": answer_type,
        "confidence": confidence,
        "grounding": grounding,
        "sources": sources,
        "history": [history_entry],
        "trace": _trace(state, "finalize", f"Answer ready ({answer_type}, confidence {confidence}, grounding {grounding})."),
    }
