"""Structured-output schemas and prompts, one pair per reasoning step.

Prompts avoid literal curly braces because ChatPromptTemplate treats them as variables.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


# ======================================================================================
# 1. Query analysis: rewrite follow-ups, detect ambiguity, decompose
# ======================================================================================
class PlannedSubQuestion(BaseModel):
    question: str = Field(description="One self-contained question")
    search_query: str = Field(description="Keyword-rich query using names, codes and terms likely to appear in documents")
    depends_on_previous: bool = Field(
        default=False,
        description="True if this can only be searched after earlier sub-questions are answered (multi-hop)",
    )


class QueryAnalysis(BaseModel):
    intent: Literal["research", "clarify", "direct"] = Field(description="research is the default")
    standalone_question: str = Field(description="The latest message rewritten to be understandable without the conversation")
    assumptions: str = Field(default="", description="How an ambiguous or under-specified question is being interpreted; empty if none")
    reply: str = Field(default="", description="Clarifying question (intent=clarify) or short reply (intent=direct); empty for research")
    sub_questions: list[PlannedSubQuestion] = Field(default_factory=list, description="1 to N sub-questions for research")


ANALYZE = ChatPromptTemplate.from_messages([
    ("system",
     "You are the query-analysis step of a research assistant. The assistant answers questions from a private "
     "document knowledge base and can fall back to web search. You do not answer the question; you prepare it "
     "for research.\n\n"
     "Knowledge base contents (document titles): {kb_titles}\n\n"
     "Conversation so far, oldest first:\n{history}\n\n"
     "Do the following:\n"
     "1. Rewrite the latest user message as a standalone question. Resolve references such as 'it', 'that "
     "supplier', 'the second one' from the conversation, and keep every constraint the user gave (dates, "
     "versions, names, units).\n"
     "2. Choose intent. research: anything that needs facts; this is the default. direct: greetings, thanks, or "
     "questions about how to use this assistant; put a one or two sentence reply in reply. clarify: only when "
     "the message has no resolvable referent or is too vague to search at all (for example 'how does it "
     "compare?' with nothing earlier to refer to); put one short clarifying question in reply. If a reasonable "
     "default reading exists, choose research instead and state that reading in assumptions.\n"
     "3. Ambiguity. If a term could mean different things in this knowledge base (for example a name used for "
     "two different things, or 'last year' when documents use fiscal years), or if newer documents may "
     "supersede older ones, write the interpretation in assumptions. When two readings are both plausible, "
     "research both with separate sub-questions.\n"
     "4. Decompose into 1 to {max_sub} sub-questions. Use a single sub-question for a simple lookup. Give each "
     "part of a compound or comparative question its own sub-question. For multi-hop questions, where a later "
     "part needs a value found by an earlier part (for example find a supplier ID, then look that ID up in a "
     "register), order the parts and set depends_on_previous=true on the later ones. For questions about "
     "policies, dates or plans, word the search so that it also finds later updates or changes."),
    ("human", "Latest user message: {question}"),
])


# ======================================================================================
# 2. Dependent-step rewrite (multi-hop): fill in values found by earlier steps
# ======================================================================================
class StepQuery(BaseModel):
    question: str = Field(description="The sub-question rewritten with concrete names, codes or values from the findings")
    search_query: str = Field(description="Keyword-rich search query for it")


REWRITE_STEP = ChatPromptTemplate.from_messages([
    ("system",
     "You prepare the next step of a multi-step research plan. Rewrite the next sub-question so it can be "
     "searched on its own: substitute the concrete names, IDs, codes, versions or dates that the earlier findings "
     "established (for example replace 'that supplier' with 'SUP-117'). Do not answer it."),
    ("human",
     "Overall question: {question}\n\nEarlier findings:\n{findings}\n\nNext sub-question: {sub_question}"),
])


# ======================================================================================
# 3. Evidence grading: relevance + sufficiency + what to search next
# ======================================================================================
class EvidenceGrade(BaseModel):
    relevant_passages: list[int] = Field(default_factory=list, description="Numbers of passages that directly help answer the sub-question")
    sufficient: bool = Field(description="True only if the relevant passages together fully answer the sub-question")
    finding: str = Field(default="", description="Concise answer to the sub-question based only on the relevant passages; note conflicts or newer versions; empty if nothing relevant")
    missing: str = Field(default="", description="What is still missing if not sufficient")
    refined_query: str = Field(default="", description="A different search query likely to find the missing information")


GRADE = ChatPromptTemplate.from_messages([
    ("system",
     "You are the evidence-checking step of a research assistant. Judge the passages strictly.\n"
     "- A passage is relevant only if it states information that helps answer the sub-question. Being about the "
     "same topic is not enough.\n"
     "- Watch for near misses: a different product or model, a different time period or fiscal year, a "
     "superseded policy version, or a different thing with the same name.\n"
     "- If passages conflict, or one updates another, say so in the finding and say which is newer.\n"
     "- sufficient=true only if the passages actually state the answer. Never use your own background knowledge "
     "to fill gaps.\n"
     "- If not sufficient, write refined_query using different wording, synonyms, or codes and names that appear "
     "in the passages (for example an ID mentioned in one passage that should be looked up elsewhere)."),
    ("human",
     "Overall user question (context only): {question}\n\n"
     "Sub-question to judge against: {sub_question}\n\n"
     "Passages:\n{passages}"),
])


# ======================================================================================
# 4. Answer synthesis
# ======================================================================================
SYNTHESIZE = ChatPromptTemplate.from_messages([
    ("system",
     "You write the final answer of a research assistant using ONLY the numbered evidence provided.\n"
     "Rules:\n"
     "- Lead with the direct answer, then the supporting detail. Be concise. Use a short list or table only when "
     "comparing several items.\n"
     "- Cite facts with evidence numbers in square brackets, like [2] or [1][4]. Cite only numbers that exist.\n"
     "- Evidence marked WEB comes from the internet, not the knowledge base; say so when you use it.\n"
     "- If sources conflict or a newer document supersedes an older one, give the current answer and mention "
     "the change and its effective date.\n"
     "- If the question was ambiguous, state the interpretation in one short sentence first; if two readings "
     "are covered, answer both.\n"
     "- If part of the question cannot be answered from the evidence, say plainly what is missing. Do not fill "
     "gaps with general knowledge or guesses.\n"
     "- Arithmetic on cited numbers (differences, percentages) is allowed; show the inputs."),
    ("human",
     "Question: {question}\n"
     "Interpreted as: {standalone}\n"
     "Assumptions: {assumptions}\n\n"
     "Research notes per sub-question:\n{notes}\n\n"
     "Evidence:\n{evidence}\n"
     "{feedback}"),
])


# ======================================================================================
# 5. Verification: groundedness + completeness
# ======================================================================================
class Verification(BaseModel):
    unsupported_claims: list[str] = Field(default_factory=list, description="Specific claims in the draft not supported by the evidence")
    missing_aspects: list[str] = Field(default_factory=list, description="Parts of the question the draft ignores entirely")
    confidence: Literal["high", "medium", "low"] = Field(description="Overall confidence in the draft")


VERIFY = ChatPromptTemplate.from_messages([
    ("system",
     "You are the fact-checking step of a research assistant. Compare the draft answer with the evidence.\n"
     "- unsupported_claims: any specific claim (number, date, name, cause, recommendation) that is not stated in, "
     "or directly calculable from, the evidence. Statements that something could not be found are not claims.\n"
     "- missing_aspects: only parts of the user's question that the draft neither answers nor explicitly says "
     "are unavailable. Leave empty if the draft covers every part.\n"
     "- confidence: high if fully supported by knowledge-base evidence; medium if it relies on web evidence or "
     "has minor gaps; low if key parts are unsupported or missing."),
    ("human", "Question: {question}\n\nEvidence:\n{evidence}\n\nDraft answer:\n{draft}"),
])


# ======================================================================================
# Naive baseline (for comparison only): one retrieval, one prompt
# ======================================================================================
NAIVE = ChatPromptTemplate.from_messages([
    ("system",
     "Answer the question using the numbered context. Cite sources like [1]. If the context does not contain the "
     "answer, say you don't know."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])
