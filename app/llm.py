"""LLM access via LangChain + Groq (free tier).

Two roles:
  * "main" - query analysis and final answer writing (quality matters most)
  * "fast" - evidence grading, dependent-query rewriting, fact-checking (called most often)

Each role is a primary model plus fallbacks. Groq's free-tier limits are per model (for example
gpt-oss-120b: 30 requests/min, 8K tokens/min, 200K tokens/day), so when one model is rate-limited or
retired, LangChain's `with_fallbacks` moves on to the next model, which has its own budget.

Structured output is done in two stages per model, because tool-calling structured output is not
reliable on every Groq model:
  1. Groq Structured Outputs (`response_format=json_schema`, best-effort mode)
  2. plain completion with the JSON schema in the prompt, parsed and validated with Pydantic
A rate-limit error skips stage 2 and goes straight to the next model.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from typing import Literal, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_groq import ChatGroq
from pydantic import BaseModel, ValidationError

from .config import settings

log = logging.getLogger(__name__)

Role = Literal["main", "fast"]
T = TypeVar("T", bound=BaseModel)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class LLMNotConfigured(RuntimeError):
    pass


class StructuredOutputError(ValueError):
    pass


def _models_for(role: Role) -> list[str]:
    if role == "main":
        names = [settings.llm_model, *settings.llm_fallback_models]
    else:
        names = [settings.fast_llm_model, *settings.fast_llm_fallback_models]
    seen: list[str] = []
    for n in names:
        if n and n not in seen:
            seen.append(n)
    return seen


def _max_tokens(role: Role) -> int:
    return settings.main_max_tokens if role == "main" else settings.fast_max_tokens


def _chat(model: str, max_tokens: int) -> BaseChatModel:
    kwargs: dict = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "max_retries": settings.llm_max_retries,  # the Groq SDK honours retry-after on 429s
        "timeout": settings.llm_timeout_s,
    }
    if model.startswith("openai/gpt-oss"):
        # GPT-OSS models reason before answering; reasoning tokens count against max_tokens.
        kwargs["reasoning_effort"] = "low"
    return ChatGroq(**kwargs)


def _require_key() -> None:
    if not settings.llm_configured:
        raise LLMNotConfigured("GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys")


def is_rate_limit(exc: BaseException) -> bool:
    if getattr(exc, "status_code", None) == 429:
        return True
    name = type(exc).__name__.lower()
    return "ratelimit" in name or "rate limit" in str(exc).lower()


# --------------------------------------------------------------------------------------
# text output
# --------------------------------------------------------------------------------------
@lru_cache(maxsize=16)
def text_llm(role: Role) -> Runnable:
    """Chat model (messages in, AIMessage out) with fallbacks."""
    _require_key()
    chain = [_chat(m, _max_tokens(role)) for m in _models_for(role)]
    head, *rest = chain
    return head.with_fallbacks(rest) if rest else head


def clean_text(content: object) -> str:
    """Normalise an AIMessage content payload to plain text and strip any leaked <think> blocks."""
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        content = "".join(parts)
    text = str(content or "")
    return _THINK_RE.sub("", text).strip()


# --------------------------------------------------------------------------------------
# structured output
# --------------------------------------------------------------------------------------
def parse_json_model(text: str, schema: type[T]) -> T:
    """Extract the first JSON object in `text` that validates against `schema`."""
    text = _FENCE_RE.sub("", clean_text(text).strip())
    candidates: list[object] = []
    try:
        candidates.append(json.loads(text))
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text):
            try:
                obj, _ = decoder.raw_decode(text[match.start():])
                candidates.append(obj)
            except json.JSONDecodeError:
                continue
    errors: list[str] = []
    for obj in candidates:
        if isinstance(obj, dict) and len(obj) == 1 and isinstance(next(iter(obj.values())), dict):
            # some models wrap the object: {"QueryAnalysis": {...}}
            inner = next(iter(obj.values()))
            try:
                return schema.model_validate(inner)
            except ValidationError:
                pass
        try:
            return schema.model_validate(obj)
        except ValidationError as exc:
            errors.append(str(exc).splitlines()[0])
    raise StructuredOutputError(f"no valid {schema.__name__} JSON in model output" +
                                (f" ({errors[0]})" if errors else ""))


def _as_messages(value: object) -> list[BaseMessage]:
    if isinstance(value, PromptValue):
        return value.to_messages()
    if isinstance(value, list):
        return list(value)
    return [HumanMessage(content=str(value))]


def _json_instruction(schema: type[BaseModel]) -> str:
    return (
        "Respond with a single JSON object and nothing else (no prose, no code fences). "
        f"It must validate against this JSON schema:\n{json.dumps(schema.model_json_schema())}"
    )


def _structured_for_model(model: str, schema: type[T], max_tokens: int) -> Runnable:
    chat = _chat(model, max_tokens)
    try:
        native = chat.with_structured_output(schema, method="json_schema")
    except Exception as exc:  # older langchain-groq without json_schema support
        log.debug("json_schema structured output unavailable for %s: %s", model, exc)
        native = None

    def call(value: object) -> T:
        if native is not None:
            try:
                result = native.invoke(value)
                if isinstance(result, schema):
                    return result
                if isinstance(result, dict):
                    return schema.model_validate(result)
            except Exception as exc:
                if is_rate_limit(exc) or isinstance(exc, LLMNotConfigured):
                    raise  # let the next model (separate rate-limit budget) handle it
                log.info("structured output via json_schema failed on %s (%s); retrying with JSON prompt",
                         model, type(exc).__name__)
        messages = [*_as_messages(value), HumanMessage(content=_json_instruction(schema))]
        msg = chat.invoke(messages)
        return parse_json_model(msg.content, schema)

    return RunnableLambda(call, name=f"structured[{model}]")


@lru_cache(maxsize=32)
def structured_llm(role: Role, schema: type[T]) -> Runnable:
    """Runnable returning an instance of `schema`, with per-model fallbacks."""
    _require_key()
    chain = [_structured_for_model(m, schema, _max_tokens(role)) for m in _models_for(role)]
    head, *rest = chain
    return head.with_fallbacks(rest) if rest else head
