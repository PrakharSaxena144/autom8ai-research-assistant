"""Compare the research agent with a naive single-pass RAG baseline on eval/questions.json.

Usage (from the project root, with GROQ_API_KEY set):
    python -m eval.run_eval                                  # in-process, agent + naive
    python -m eval.run_eval --base-url http://localhost:8000 # against a running server
    python -m eval.run_eval --modes agent --only ocr-auditor calc-software-growth
    python -m eval.run_eval --judge                          # add an LLM-as-judge verdict (extra Groq calls)

Scoring (automatic, deterministic):
  * correct       - every required key fact appears in the answer (each fact accepts listed alternatives);
                    for unanswerable questions: the answer declines instead of inventing something
  * fact_coverage - fraction of required key facts present
  * cited         - the answer contains at least one [n] citation that maps to a returned source
  * source_recall - fraction of expected documents among the returned sources
Results go to eval/results/<timestamp>.json and .md.

Free-tier note: the agent makes several LLM calls per question. Keep --sleep at a few seconds so Groq's
per-minute token limits are not hit constantly (fallback models absorb occasional 429s).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

QUESTIONS = Path(__file__).with_name("questions.json")
RESULTS_DIR = Path(__file__).with_name("results")

DECLINE_RE = re.compile(
    r"could(?: not|n't) find|can(?:not|'t) find|not (?:mentioned|found|available|stated|included|provided|specified)"
    r"|no (?:information|mention|ticker|data)|(?:do not|don't) (?:know|have)|does(?: not|n't) (?:mention|contain|include|say)"
    r"|is(?: not|n't) (?:in|listed|mentioned|publicly)|not (?:a )?publicly (?:listed|traded)|unable to",
    re.IGNORECASE,
)
CITATION_RE = re.compile(r"\[(\d{1,3})\]")
LLM_STEPS = {"analyze", "grade", "synthesize", "verify"}


# --------------------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------------------
def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-").replace("\u00a0", " ")
    text = text.replace("**", "").replace("&amp;", "&")
    return re.sub(r"\s+", " ", text).lower()


def score(item: dict, response: dict) -> dict:
    answer = response.get("answer", "")
    norm = normalise(answer)
    sources = response.get("sources", [])

    if item.get("expect_no_answer"):
        declined = response.get("answer_type") == "not_found" or bool(DECLINE_RE.search(answer))
        coverage = 1.0 if declined else 0.0
        correct = declined
        missing: list[str] = [] if declined else ["should decline"]
    else:
        groups = item.get("must_include", [])
        hits = [any(normalise(alt) in norm for alt in group) for group in groups]
        coverage = sum(hits) / len(groups) if groups else 1.0
        correct = all(hits)
        missing = [group[0] for group, hit in zip(groups, hits) if not hit]

    numbers = {int(n) for n in CITATION_RE.findall(answer)}
    source_numbers = {s.get("n") for s in sources}
    cited = bool(numbers & source_numbers)

    expected = [e.lower() for e in item.get("expected_sources", [])]
    titles = " || ".join(normalise(f"{s.get('title') or ''} {s.get('source') or ''}") for s in sources)
    recall = (sum(e in titles for e in expected) / len(expected)) if expected else None

    trace = response.get("reasoning_trace") or response.get("trace") or []
    llm_calls = sum(1 for t in trace if t.get("step") in LLM_STEPS and "Nothing to grade" not in t.get("detail", ""))
    llm_calls += sum(1 for t in trace if t.get("step") == "plan_step" and "uses earlier findings" in t.get("detail", ""))
    if response.get("mode") == "naive":
        llm_calls = 1

    return {"correct": correct, "fact_coverage": round(coverage, 3), "missing_facts": missing, "cited": cited,
            "source_recall": recall, "llm_calls": llm_calls}


# --------------------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------------------
class InProcess:
    name = "in-process"

    def __init__(self) -> None:
        from app.config import settings

        if not settings.llm_configured:
            sys.exit("GROQ_API_KEY is not set (see .env.example).")
        from app.agent import graph as agent
        from app.ingestion.pipeline import seed_directory
        from app.vectorstore import get_kb

        self.agent = agent
        kb = get_kb()
        if kb.count() == 0:
            print("Knowledge base empty: ingesting data/corpus ...", flush=True)
            for r in seed_directory(kb, settings.seed_dir):
                print(f"  {r.status:9} {r.source} ({r.chunks} chunks){' - ' + r.error if r.error else ''}")

    def ask(self, question: str, mode: str, conversation_id: str, use_web: bool) -> dict:
        from app.main import _response

        if mode == "naive":
            state = self.agent.run_naive(question)
        else:
            state = self.agent.run_agent(question, conversation_id, use_web=use_web)
        return json.loads(_response(state, conversation_id, question, mode).model_dump_json())


class Http:
    name = "http"

    def __init__(self, base_url: str) -> None:
        import httpx

        self.client = httpx.Client(base_url=base_url.rstrip("/"), timeout=600)
        for _ in range(120):  # wait for startup seeding
            health = self.client.get("/health").json()
            if not str(health.get("seeding", "")).startswith(("pending", "loading", "ingesting")):
                break
            print(f"waiting for server: {health.get('seeding')}", flush=True)
            time.sleep(5)
        if not health.get("llm_configured"):
            sys.exit("The server has no GROQ_API_KEY configured.")

    def ask(self, question: str, mode: str, conversation_id: str, use_web: bool) -> dict:
        r = self.client.post("/ask", json={"question": question, "mode": mode,
                                           "conversation_id": conversation_id, "use_web": use_web})
        r.raise_for_status()
        return r.json()


def judge(item: dict, answer: str) -> dict:
    """Optional LLM-as-judge against the reference answer (uses the fast model)."""
    from langchain_core.prompts import ChatPromptTemplate
    from pydantic import BaseModel, Field

    from app.llm import structured_llm

    class Verdict(BaseModel):
        correct: bool = Field(description="True if the answer is factually consistent with the reference and answers the question")
        reason: str = Field(description="One sentence")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You grade answers against a reference answer. Minor wording differences are fine. An answer that "
                   "states outdated or superseded information as current is incorrect. For unanswerable questions, "
                   "declining is correct and inventing an answer is incorrect."),
        ("human", "Question: {question}\n\nReference: {reference}\n\nAnswer to grade:\n{answer}"),
    ])
    try:
        v = (prompt | structured_llm("fast", Verdict)).invoke(
            {"question": item["question"], "reference": item["reference"], "answer": answer})
        return {"judge_correct": v.correct, "judge_reason": v.reason}
    except Exception as exc:  # noqa: BLE001
        return {"judge_correct": None, "judge_reason": f"judge failed: {type(exc).__name__}"}


# --------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------
def _pct(values: list) -> str:
    values = [v for v in values if v is not None]
    return f"{100 * sum(values) / len(values):.0f}%" if values else "-"


def _avg(values: list, fmt: str = "{:.1f}") -> str:
    values = [v for v in values if v is not None]
    return fmt.format(sum(values) / len(values)) if values else "-"


def report(rows: list[dict], modes: list[str], with_judge: bool) -> str:
    lines = ["# Evaluation: research agent vs naive RAG", "",
             f"Run: {datetime.now().isoformat(timespec='seconds')} - {len({r['id'] for r in rows})} questions", "",
             "## Overall", ""]
    columns = ["Mode", "Correct (key facts)", "Fact coverage", "Cited", "Source recall",
               "Avg latency (s)", "Avg LLM calls", "Errors"]
    if with_judge:
        columns.insert(2, "Judge correct")
    lines += ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for mode in modes:
        rs = [r for r in rows if r["mode"] == mode]
        ok = [r for r in rs if not r.get("error")]
        cells = [mode, _pct([r["correct"] for r in rs]), _pct([r["fact_coverage"] for r in ok]),
                 _pct([r["cited"] for r in ok if not r.get("expect_no_answer")]),
                 _pct([r["source_recall"] for r in ok]), _avg([r["latency_s"] for r in ok]),
                 _avg([r["llm_calls"] for r in ok]), str(len(rs) - len(ok))]
        if with_judge:
            cells.insert(2, _pct([r.get("judge_correct") for r in ok]))
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## By category (correct)", "", "| Category | " + " | ".join(modes) + " |",
              "|---|" + "---|" * len(modes)]
    for cat in sorted({r["category"] for r in rows}):
        cells = [_pct([r["correct"] for r in rows if r["category"] == cat and r["mode"] == m]) for m in modes]
        lines.append(f"| {cat} | " + " | ".join(cells) + " |")

    lines += ["", "## Per question", "", "| Question | " + " | ".join(modes) + " |", "|---|" + "---|" * len(modes)]
    for qid in dict.fromkeys(r["id"] for r in rows):
        cells = []
        for m in modes:
            r = next((x for x in rows if x["id"] == qid and x["mode"] == m), None)
            if not r:
                cells.append("-")
            elif r.get("error"):
                cells.append(f"error: {r['error'][:40]}")
            else:
                mark = "pass" if r["correct"] else "fail"
                if r["missing_facts"]:
                    mark += f" (missing: {', '.join(r['missing_facts'])})"
                cells.append(mark)
        lines.append(f"| {qid} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", help="Evaluate a running server instead of running in-process")
    parser.add_argument("--modes", nargs="+", default=["agent", "naive"], choices=["agent", "naive"])
    parser.add_argument("--only", nargs="+", help="Question ids to run")
    parser.add_argument("--sleep", type=float, default=6.0, help="Seconds to pause between questions (rate limits)")
    parser.add_argument("--judge", action="store_true", help="Also grade with an LLM judge")
    args = parser.parse_args()

    items = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    if args.only:
        items = [i for i in items if i["id"] in set(args.only)]
    backend = Http(args.base_url) if args.base_url else InProcess()

    rows: list[dict] = []
    for item in items:
        for mode in args.modes:
            cid = str(uuid.uuid4())
            row = {"id": item["id"], "category": item["category"], "mode": mode, "question": item["question"],
                   "expect_no_answer": bool(item.get("expect_no_answer"))}
            print(f"[{mode:5}] {item['id']} ...", end=" ", flush=True)
            try:
                if mode == "agent":  # the naive baseline has no memory, so it only sees the final question
                    for turn in item.get("turns", []):
                        backend.ask(turn, mode, cid, item.get("use_web", False))
                        time.sleep(args.sleep)
                start = time.perf_counter()
                response = backend.ask(item["question"], mode, cid, item.get("use_web", False))
                row["latency_s"] = round(time.perf_counter() - start, 2)
                row.update(score(item, response))
                if args.judge:
                    row.update(judge(item, response.get("answer", "")))
                row.update({"answer": response.get("answer"), "answer_type": response.get("answer_type"),
                            "confidence": response.get("confidence"), "grounding": response.get("grounding"),
                            "sources": [s.get("title") or s.get("source") for s in response.get("sources", [])]})
                print("pass" if row["correct"] else f"fail {row['missing_facts']}", f"({row['latency_s']}s)")
            except Exception as exc:  # noqa: BLE001 - keep going, report the error
                row.update({"correct": False, "fact_coverage": 0.0, "cited": False, "source_recall": None,
                            "llm_calls": None, "missing_facts": [], "error": f"{type(exc).__name__}: {exc}"})
                print(f"error {row['error'][:120]}")
            rows.append(row)
            time.sleep(args.sleep)

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (RESULTS_DIR / f"{stamp}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    md = report(rows, args.modes, args.judge)
    (RESULTS_DIR / f"{stamp}.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"Saved eval/results/{stamp}.json and .md")


if __name__ == "__main__":
    main()
