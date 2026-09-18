"""Chunking, dedup and the SSRF guard. No embedding model or vector store is loaded."""

import pytest

pytest.importorskip("langchain_text_splitters")
pytest.importorskip("httpx")

from app.ingestion import pipeline  # noqa: E402
from app.ingestion.parsers import Block, ParsedDocument  # noqa: E402
from tests.conftest import CORPUS  # noqa: E402


class FakeKB:
    def __init__(self):
        self.docs = {}
        self.added = []

    def get_document(self, doc_id):
        return self.docs.get(doc_id)

    def add_chunks(self, chunks, ids):
        assert len(chunks) == len(ids)
        self.added.extend(zip(ids, chunks))
        meta = chunks[0].metadata
        self.docs[meta["doc_id"]] = {"title": meta["title"], "file_type": meta["file_type"], "chunks": len(chunks)}


def test_chunks_carry_context_header_and_metadata():
    parsed = ParsedDocument(file_type="pdf", blocks=[
        Block(text="LX-9 lidar is single-sourced from SUP-117. " * 3, page=3, section="5. Risk factors"),
    ], ocr_pages=[3])
    docs, ids = pipeline.build_chunks(parsed, doc_id="abc", title="Annual Report", source="report.pdf")
    assert docs and len(docs) == len(ids)
    first = docs[0]
    assert first.page_content.startswith("Document: Annual Report | Section: 5. Risk factors | Page: 3\n")
    assert first.metadata["doc_id"] == "abc" and first.metadata["page"] == 3 and first.metadata["ocr"] is True


def test_chunk_ids_are_deterministic():
    parsed = ParsedDocument(file_type="text", blocks=[Block(text="Some reasonably long text block. " * 50)])
    _, ids1 = pipeline.build_chunks(parsed, "doc1", "T", "t.txt")
    _, ids2 = pipeline.build_chunks(parsed, "doc1", "T", "t.txt")
    assert ids1 == ids2 and len(set(ids1)) == len(ids1)


def test_long_blocks_are_split_with_limit():
    parsed = ParsedDocument(file_type="text", blocks=[Block(text="word " * 2000, section="Long")])
    docs, _ = pipeline.build_chunks(parsed, "d", "T", "t.txt")
    header_len = len("Document: T | Section: Long\n")
    assert len(docs) > 1
    assert all(len(d.page_content) - header_len <= pipeline.settings.chunk_size for d in docs)


def test_small_sections_on_same_page_are_packed():
    blocks = [Block(text="Short A.", page=1, section="A"), Block(text="Short B.", page=1, section="B"),
              Block(text="Other page.", page=2, section="C")]
    packed = pipeline._pack_blocks(blocks, 900)
    assert len(packed) == 2
    assert "B\nShort B." in packed[0].text


def test_ingest_bytes_adds_then_detects_duplicate():
    kb = FakeKB()
    data = (CORPUS / "Policy_Update_Memo_2025.md").read_bytes()
    first = pipeline.ingest_bytes(kb, data, "Policy_Update_Memo_2025.md")
    assert first.status == "added" and first.chunks > 0 and first.file_type == "markdown"
    again = pipeline.ingest_bytes(kb, data, "renamed-copy.md")
    assert again.status == "duplicate" and again.doc_id == first.doc_id
    assert len(kb.added) == first.chunks  # nothing re-embedded


def test_ingest_bytes_reports_failures_without_raising():
    kb = FakeKB()
    assert pipeline.ingest_bytes(kb, bytes(range(256)) * 4, "blob.bin").status == "failed"
    assert pipeline.ingest_bytes(kb, b"", "empty.txt").status == "failed"


def test_ingest_text_uses_title():
    kb = FakeKB()
    result = pipeline.ingest_text(kb, "The cafeteria opens at 8:00 and closes at 15:00 on weekdays.", "Cafeteria hours")
    assert result.status == "added" and result.title == "Cafeteria hours"


@pytest.mark.parametrize("address,blocked", [
    ("127.0.0.1", True), ("10.1.2.3", True), ("192.168.0.10", True), ("169.254.169.254", True),
    ("::1", True), ("fe80::1%eth0", True), ("::ffff:127.0.0.1", True), ("0.0.0.0", True),
    ("8.8.8.8", False), ("2606:4700:4700::1111", False),
])
def test_is_blocked_ip(address, blocked):
    assert pipeline.is_blocked_ip(address) is blocked


def test_url_guard_rejects_bad_schemes_and_private_hosts():
    with pytest.raises(pipeline.IngestionError):
        pipeline._assert_public_url("file:///etc/passwd")
    with pytest.raises(pipeline.IngestionError):
        pipeline._assert_public_url("http://127.0.0.1:8000/admin")  # IP literal: no DNS needed
    result = pipeline.ingest_url(FakeKB(), "http://169.254.169.254/latest/meta-data/")
    assert result.status == "failed" and "private" in result.error
