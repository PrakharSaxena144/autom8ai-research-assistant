"""Parsing of model output. No model is called."""

import pytest

pytest.importorskip("langchain_groq")

from pydantic import BaseModel  # noqa: E402

from app.llm import StructuredOutputError, clean_text, is_rate_limit, parse_json_model  # noqa: E402


class Grade(BaseModel):
    relevant_passages: list[int] = []
    sufficient: bool


def test_parse_plain_json():
    assert parse_json_model('{"relevant_passages": [1, 3], "sufficient": true}', Grade).relevant_passages == [1, 3]


def test_parse_fenced_json_with_reasoning_and_prose():
    text = '<think>let me see</think>Here you go:\n```json\n{"relevant_passages": [2], "sufficient": false}\n```'
    assert parse_json_model(text, Grade) == Grade(relevant_passages=[2], sufficient=False)


def test_parse_skips_invalid_objects_and_unwraps_named_wrapper():
    text = 'Example {"foo": 1} then the real one {"Grade": {"sufficient": true}}'
    assert parse_json_model(text, Grade).sufficient is True


def test_parse_raises_when_nothing_validates():
    with pytest.raises(StructuredOutputError):
        parse_json_model("no json here", Grade)


def test_clean_text_handles_content_blocks():
    content = [{"type": "text", "text": "<think>x</think>Answer "}, {"type": "reasoning", "text": "hidden"}, "[1]"]
    assert clean_text(content) == "Answer [1]"


def test_is_rate_limit():
    class RateLimitError(Exception):
        status_code = 429

    assert is_rate_limit(RateLimitError("slow down"))
    assert not is_rate_limit(ValueError("bad schema"))
