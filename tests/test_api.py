"""HTTP layer without starting the lifespan (so no embedding models are loaded)."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("langgraph")
pytest.importorskip("langchain_qdrant")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)  # not used as a context manager -> lifespan (model loading, seeding) does not run


def test_ui_is_served():
    r = client.get("/")
    assert r.status_code == 200 and "<html" in r.text.lower()


def test_health_does_not_block_on_model_loading():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["llm_configured"] is False
    assert body["vector_store"] is None and body["web_search"] == "none"


def test_ask_without_key_returns_503_with_instructions():
    for path in ("/ask", "/ask/stream"):
        r = client.post(path, json={"question": "What is Atlas?"})
        assert r.status_code == 503
        assert "console.groq.com" in r.json()["detail"]


def test_ask_validates_input():
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "x", "mode": "other"}).status_code == 422


def test_graph_endpoint_returns_mermaid():
    r = client.get("/graph")
    assert r.status_code == 200 and "grade_evidence" in r.text


def test_openapi_lists_endpoints():
    paths = client.get("/openapi.json").json()["paths"]
    for p in ("/documents", "/documents/{doc_id}", "/ask", "/ask/stream", "/conversations/{conversation_id}"):
        assert p in paths
