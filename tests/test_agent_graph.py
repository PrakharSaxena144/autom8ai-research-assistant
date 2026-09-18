"""End-to-end runs of the real LangGraph agent with a scripted LLM, a fake knowledge base and fake web search.

This checks the control flow the assignment cares about - decomposition, multi-hop rewriting, query
rewriting, web fallback, fact-check revision, not-found handling, clarification and conversation memory -
without network access or API keys.
"""

import re
import uuid

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_qdrant")

from langchain_core.documents import Document  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.runnables import RunnableLambda  # noqa: E402

from app.agent import graph as agent  # noqa: E402
from app.agent import nodes, prompts  # noqa: E402
from app.websearch import WebResult  # noqa: E402

CORPUS_CHUNKS = [
    ("Annual Report FY2025", "5. Risk factors", 3,
     "The LX-9 3D lidar module used in every Atlas unit is supplied by a single supplier (internal supplier "
     "reference SUP-117). A second source is under evaluation."),
    ("supplier register", "rows 1-10", None,
     "supplier_id = SUP-117; supplier_name = Lumenar Optics GmbH; country = Austria; component = 3D lidar module; "
     "component_code = LX-9; used_in = Atlas; contract_end = 2026-11-30; single_source = Yes"),
    ("Employee Handbook", "3. Leave", None,
     "Parental leave: the primary caregiver receives 16 weeks of fully paid parental leave."),
    ("Product specifications", "Autonomous mobile robots", None,
     "Rated payload: Porter (AMR-400) = 400 kg; Atlas (AMR-700) = 700 kg"),
]


class FakeKB:
    backend = "fake"

    def __init__(self, chunks=CORPUS_CHUNKS):
        self.docs = []
        for i, (title, section, page, text) in enumerate(chunks):
            header = f"Document: {title} | Section: {section}" + (f" | Page: {page}" if page else "")
            self.docs.append(Document(page_content=f"{header}\n{text}", metadata={
                "_id": f"pt-{i}", "doc_id": f"doc-{i}", "title": title, "source": f"{title}.src",
                "page": page, "section": section, "chunk_index": 0}))
        self.queries = []

    def search(self, query, k):
        self.queries.append(query)
        terms = {t for t in re.findall(r"[a-z0-9-]+", query.lower()) if len(t) > 2}
        scored = []
        for d in self.docs:
            words = set(re.findall(r"[a-z0-9-]+", d.page_content.lower()))
            overlap = len(terms & words)
            if overlap:
                scored.append((overlap, d))
        scored.sort(key=lambda x: -x[0])
        return [(d, float(s)) for s, d in scored[:k]]

    def titles(self):
        return sorted({d.metadata["title"] for d in self.docs})


class ScriptedLLM:
    """Stands in for Groq. `handlers[schema]` receives the rendered prompt text and returns a schema instance."""

    def __init__(self, handlers, answer):
        self.handlers = handlers
        self.answer = answer
        self.calls = []

    def structured(self, role, schema):
        def run(prompt_value):
            text = prompt_value.to_string()
            self.calls.append(schema.__name__)
            return self.handlers[schema](text)
        return RunnableLambda(run)

    def text(self, role):
        def run(prompt_value):
            self.calls.append("SYNTHESIZE")
            text = prompt_value.to_string()
            return AIMessage(content=self.answer(text) if callable(self.answer) else self.answer)
        return RunnableLambda(run)


def _passage_numbers(prompt_text, needle):
    """Passage numbers in a GRADE prompt whose body contains `needle`."""
    found = []
    for m in re.finditer(r"^\[(\d+)\] \((.*?)\)\n(.*?)(?=^\[\d+\] \(|\Z)", prompt_text, re.S | re.M):
        if needle.lower() in m.group(3).lower():
            found.append(int(m.group(1)))
    return found


def _sub_question(prompt_text):
    return re.search(r"Sub-question to judge against: (.*)", prompt_text).group(1)


@pytest.fixture
def wire(monkeypatch):
    def _wire(llm, kb=None, web_results=None, web_on=False):
        kb = kb or FakeKB()
        monkeypatch.setattr(nodes, "structured_llm", llm.structured)
        monkeypatch.setattr(nodes, "text_llm", llm.text)
        monkeypatch.setattr(nodes, "get_kb", lambda: kb)
        monkeypatch.setattr(nodes, "web_enabled", lambda: web_on)
        monkeypatch.setattr(nodes, "search_web", lambda q, n=None: (web_results or [], None if web_results else "none"))
        return kb
    return _wire


def _steps(state):
    return [t["step"] for t in state["trace"]]


# --------------------------------------------------------------------------------------
def test_multi_hop_question_rewrites_dependent_step_and_cites_sources(wire):
    def analyze(_):
        return prompts.QueryAnalysis(
            intent="research",
            standalone_question="Which country is the supplier of the Atlas lidar in, and when does its contract end?",
            sub_questions=[
                prompts.PlannedSubQuestion(question="Which supplier provides the lidar used in Atlas?",
                                           search_query="Atlas LX-9 lidar supplier reference"),
                prompts.PlannedSubQuestion(question="Where is that supplier based and when does its contract end?",
                                           search_query="supplier country contract_end", depends_on_previous=True),
            ])

    def grade(text):
        sub = _sub_question(text)
        if "SUP-117" in sub:
            nums = _passage_numbers(text, "Lumenar")
            return prompts.EvidenceGrade(relevant_passages=nums, sufficient=bool(nums),
                                         finding="SUP-117 is Lumenar Optics GmbH, Austria, contract ends 2026-11-30")
        nums = _passage_numbers(text, "single supplier")
        return prompts.EvidenceGrade(relevant_passages=nums, sufficient=True, finding="Supplier reference SUP-117")

    llm = ScriptedLLM({
        prompts.QueryAnalysis: analyze,
        prompts.StepQuery: lambda text: prompts.StepQuery(
            question="Where is supplier SUP-117 based and when does its contract end?",
            search_query="SUP-117 country contract_end"),
        prompts.EvidenceGrade: grade,
        prompts.Verification: lambda text: prompts.Verification(confidence="high"),
    }, answer="The LX-9 lidar supplier SUP-117 [1] is Lumenar Optics GmbH in Austria; the contract ends on 2026-11-30 [2].")
    kb = wire(llm)

    cid = str(uuid.uuid4())
    state = agent.run_agent("Where is Atlas's lidar supplier based and when does the contract end?", cid, use_web=False)

    assert state["answer_type"] == "answer"
    assert state["grounding"] == "knowledge_base"
    assert state["confidence"] == "high"
    subs = state["sub_questions"]
    assert [s["status"] for s in subs] == ["answered", "answered"]
    assert "SUP-117" in subs[1]["question"]                     # multi-hop: earlier finding substituted
    assert any("SUP-117" in q for q in kb.queries)
    assert [s["title"] for s in state["sources"]] == ["Annual Report FY2025", "supplier register"]
    assert all(s["cited"] for s in state["sources"])
    assert _steps(state)[0] == "analyze" and _steps(state)[-1] == "finalize"
    assert "StepQuery" in llm.calls and llm.calls.count("SYNTHESIZE") == 1


def test_insufficient_kb_rewrites_query_then_uses_web(wire):
    web = [WebResult(title="ISO 3691-4 overview", url="https://example.org/iso-3691-4",
                     content="ISO 3691-4 specifies safety requirements for driverless industrial trucks.")]

    def grade(text):
        nums = _passage_numbers(text, "driverless")
        if nums:
            return prompts.EvidenceGrade(relevant_passages=nums, sufficient=True, finding="Safety of driverless trucks")
        return prompts.EvidenceGrade(sufficient=False, missing="scope of the standard",
                                     refined_query="ISO 3691-4 standard scope safety")

    llm = ScriptedLLM({
        prompts.QueryAnalysis: lambda _: prompts.QueryAnalysis(
            intent="research", standalone_question="What does ISO 3691-4 cover?",
            sub_questions=[prompts.PlannedSubQuestion(question="What does ISO 3691-4 cover?",
                                                      search_query="ISO 3691-4 Atlas certification")]),
        prompts.EvidenceGrade: grade,
        prompts.Verification: lambda _: prompts.Verification(confidence="high"),
    }, answer="According to a web source, ISO 3691-4 covers safety requirements for driverless industrial trucks [1].")
    wire(llm, web_results=web, web_on=True)

    state = agent.run_agent("What does ISO 3691-4 cover?", str(uuid.uuid4()), use_web=True)
    steps = _steps(state)
    assert "rewrite_query" in steps and "web_search" in steps
    assert steps.index("rewrite_query") < steps.index("web_search")
    assert state["grounding"] == "web"
    assert state["confidence"] == "medium"                     # web-only answers are never "high"
    assert state["sub_questions"][0]["used_web"] is True
    assert state["sources"][0]["url"] == "https://example.org/iso-3691-4"


def test_nothing_found_returns_not_found_without_guessing(wire):
    llm = ScriptedLLM({
        prompts.QueryAnalysis: lambda _: prompts.QueryAnalysis(
            intent="research", standalone_question="What is the stock ticker?",
            sub_questions=[prompts.PlannedSubQuestion(question="What is the stock ticker?", search_query="zzqx ticker")]),
        prompts.EvidenceGrade: lambda _: prompts.EvidenceGrade(sufficient=False),
        prompts.Verification: lambda _: prompts.Verification(confidence="high"),
    }, answer="should never be used")
    wire(llm, kb=FakeKB(chunks=[]))

    state = agent.run_agent("What is Nimbus's stock ticker?", str(uuid.uuid4()), use_web=False)
    assert state["answer_type"] == "not_found"
    assert state["sources"] == [] and state["grounding"] == "none"
    assert "SYNTHESIZE" not in llm.calls
    assert "no_answer" in _steps(state)


def test_fact_check_triggers_one_revision_and_flags_remaining_claims(wire):
    drafts = iter(["Primary caregivers get 16 weeks [1] and a bonus of EUR 5,000.",
                   "Primary caregivers get 16 weeks [1] and a bonus of EUR 5,000."])
    llm = ScriptedLLM({
        prompts.QueryAnalysis: lambda _: prompts.QueryAnalysis(
            intent="research", standalone_question="How long is parental leave?",
            sub_questions=[prompts.PlannedSubQuestion(question="How long is parental leave?", search_query="parental leave weeks")]),
        prompts.EvidenceGrade: lambda text: prompts.EvidenceGrade(
            relevant_passages=_passage_numbers(text, "parental"), sufficient=True, finding="16 weeks"),
        prompts.Verification: lambda _: prompts.Verification(unsupported_claims=["bonus of EUR 5,000"], confidence="medium"),
    }, answer=lambda _: next(drafts))
    wire(llm)

    state = agent.run_agent("How long is parental leave?", str(uuid.uuid4()), use_web=False)
    assert llm.calls.count("SYNTHESIZE") == 2                  # draft + exactly one revision (MAX_REVISIONS=1)
    assert state["confidence"] == "low"
    assert "could not confirm" in state["answer"]


def test_missing_aspect_triggers_follow_up_research(wire):
    verdicts = iter([prompts.Verification(missing_aspects=["Atlas payload"], confidence="medium"),
                     prompts.Verification(confidence="high")])
    llm = ScriptedLLM({
        prompts.QueryAnalysis: lambda _: prompts.QueryAnalysis(
            intent="research", standalone_question="Parental leave and Atlas payload?",
            sub_questions=[prompts.PlannedSubQuestion(question="How long is parental leave?", search_query="parental leave")]),
        prompts.EvidenceGrade: lambda text: prompts.EvidenceGrade(
            relevant_passages=_passage_numbers(text, "parental") or _passage_numbers(text, "payload"), sufficient=True),
        prompts.Verification: lambda _: next(verdicts),
    }, answer="16 weeks [1]; Atlas carries 700 kg [2].")
    wire(llm)

    state = agent.run_agent("Parental leave length and Atlas payload?", str(uuid.uuid4()), use_web=False)
    assert "plan_followups" in _steps(state)
    assert [s["origin"] for s in state["sub_questions"]] == ["plan", "follow-up"]
    assert state["confidence"] == "high"


def test_clarification_and_conversation_memory(wire):
    seen_history = []

    def analyze(text):
        seen_history.append(text)
        if "compare" in text.lower() and "Assistant:" not in text:
            return prompts.QueryAnalysis(intent="clarify", standalone_question="How does it compare?",
                                         reply="What would you like me to compare?")
        return prompts.QueryAnalysis(intent="direct", standalone_question="thanks", reply="You're welcome!")

    llm = ScriptedLLM({prompts.QueryAnalysis: analyze}, answer="unused")
    wire(llm)
    cid = str(uuid.uuid4())

    first = agent.run_agent("How does it compare?", cid)
    assert first["answer_type"] == "clarification" and first["answer"] == "What would you like me to compare?"

    second = agent.run_agent("ok thanks", cid)
    assert second["answer_type"] == "direct"
    assert "User: How does it compare?" in seen_history[-1]    # earlier turn was given to the planner
    assert len(agent.conversation_history(cid)) == 2
    assert second["sub_questions"] == [] and second["sources"] == []  # per-turn state was reset


def test_llm_failure_degrades_gracefully(wire):
    def boom(_):
        raise RuntimeError("model unavailable")

    llm = ScriptedLLM({prompts.QueryAnalysis: boom, prompts.EvidenceGrade: boom, prompts.Verification: boom},
                      answer="Atlas carries 700 kg [1].")
    wire(llm)
    state = agent.run_agent("Atlas payload", str(uuid.uuid4()), use_web=False)
    assert state["answer_type"] == "answer"
    assert state["sources"]
    assert any("grader unavailable" in t["detail"] for t in state["trace"])
    assert any("Fact-check unavailable" in t["detail"] for t in state["trace"])


def test_stream_emits_steps_then_final_state(wire):
    llm = ScriptedLLM({prompts.QueryAnalysis: lambda _: prompts.QueryAnalysis(
        intent="direct", standalone_question="hi", reply="Hello!")}, answer="unused")
    wire(llm)
    events = list(agent.stream_agent("hi", str(uuid.uuid4())))
    assert [e["event"] for e in events][-1] == "final"
    assert [e["step"] for e in events if e["event"] == "step"] == ["analyze", "respond", "finalize"]
    assert events[-1]["state"]["answer"] == "Hello!"


def test_graph_diagram_lists_all_nodes():
    diagram = agent.mermaid_diagram()
    for node in ["analyze", "prepare_step", "retrieve_kb", "grade_evidence", "rewrite_query", "web_search",
                 "record_step", "synthesize", "verify", "plan_followups", "no_answer", "finalize"]:
        assert node in diagram


# --------------------------------------------------------------------------------------
# routers in isolation
# --------------------------------------------------------------------------------------
def test_route_after_grade(monkeypatch):
    monkeypatch.setattr(nodes, "web_enabled", lambda: True)
    assert nodes.route_after_grade({"last_grade": {"sufficient": True}}) == "record_step"
    assert nodes.route_after_grade({"last_grade": {}, "kb_attempts": 1}) == "rewrite_query"
    assert nodes.route_after_grade({"last_grade": {}, "kb_attempts": 2, "use_web": True}) == "web_search"
    assert nodes.route_after_grade({"last_grade": {}, "kb_attempts": 2, "use_web": False}) == "record_step"
    assert nodes.route_after_grade({"last_grade": {}, "kb_attempts": 2, "web_tried": True}) == "record_step"


def test_finalize_ignores_out_of_range_citations():
    ev = {"kb:1": {"id": "kb:1", "type": "kb", "title": "A", "source": "a.pdf", "text": "Document: A\nalpha"},
          "kb:2": {"id": "kb:2", "type": "kb", "title": "B", "source": "b.pdf", "text": "Document: B\nbeta"}}
    out = nodes.finalize({"question": "q", "draft": "Uses [2] and a bogus [7].", "evidence": ev,
                          "evidence_order": ["kb:1", "kb:2"], "verification": {"confidence": "high"},
                          "sub_questions": [], "trace": []})
    assert [s["n"] for s in out["sources"]] == [2]
    assert out["sources"][0]["snippet"] == "beta"
    assert out["history"][0]["source_titles"] == ["B"]
