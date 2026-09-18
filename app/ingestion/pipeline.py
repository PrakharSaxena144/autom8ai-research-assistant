"""One ingestion path for every source: uploaded file, URL, or raw text.

    bytes -> parse_document (format detection + parser) -> blocks -> section-aware chunks -> Qdrant

Chunks carry a contextual header ("Document | Section | Page") inside the embedded text, so both dense
and BM25 retrieval can match on document and section names, not just the chunk body.
"""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import socket
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import settings
from .parsers import Block, ParsedDocument, parse_document

log = logging.getLogger(__name__)

CHUNK_NAMESPACE = uuid.UUID("5b0c7b1e-6f0e-4a44-9a57-2a3a5c0f1d10")


class IngestionError(ValueError):
    pass


@dataclass
class IngestResult:
    status: str  # "added" | "duplicate" | "failed"
    source: str
    doc_id: str | None = None
    title: str | None = None
    file_type: str | None = None
    chunks: int = 0
    ocr_pages: list[int] = field(default_factory=list)
    error: str | None = None


# --------------------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------------------
def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )


def _pack_blocks(blocks: list[Block], limit: int) -> list[Block]:
    """Merge small neighbouring sections on the same page so short sections keep their context."""
    packed: list[Block] = []
    for b in blocks:
        prev = packed[-1] if packed else None
        if (
            prev is not None
            and prev.page == b.page
            and len(prev.text) + len(b.text) + 40 <= limit
            and (len(prev.text) < limit // 2 or len(b.text) < limit // 3)
        ):
            heading = f"{b.section}\n" if b.section and b.section != prev.section else ""
            prev.text = f"{prev.text}\n\n{heading}{b.text}"
            continue
        packed.append(Block(text=b.text, page=b.page, section=b.section))
    return packed


def build_chunks(parsed: ParsedDocument, doc_id: str, title: str, source: str) -> tuple[list[Document], list[str]]:
    splitter = _splitter()
    added_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    docs: list[Document] = []
    ids: list[str] = []
    for block in _pack_blocks(parsed.blocks, settings.chunk_size):
        for piece in splitter.split_text(block.text):
            piece = piece.strip()
            if len(piece) < 15:
                continue
            index = len(docs)
            header_parts = [f"Document: {title}"]
            if block.section:
                header_parts.append(f"Section: {block.section}")
            if block.page:
                header_parts.append(f"Page: {block.page}")
            content = " | ".join(header_parts) + "\n" + piece
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "doc_id": doc_id,
                        "title": title,
                        "source": source,
                        "file_type": parsed.file_type,
                        "page": block.page,
                        "section": block.section,
                        "chunk_index": index,
                        "ocr": bool(block.page in parsed.ocr_pages or parsed.file_type == "image"),
                        "added_at": added_at,
                    },
                )
            )
            ids.append(str(uuid.uuid5(CHUNK_NAMESPACE, f"{doc_id}:{index}")))
    return docs, ids


# --------------------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------------------
def ingest_bytes(kb, data: bytes, filename: str, content_type: str | None = None,
                 title: str | None = None, source: str | None = None) -> IngestResult:
    source = source or filename
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        return IngestResult(status="failed", source=source, error=f"File larger than {settings.max_upload_mb} MB")

    doc_id = hashlib.sha256(data).hexdigest()[:20]
    existing = kb.get_document(doc_id)
    if existing:
        return IngestResult(status="duplicate", source=source, doc_id=doc_id, title=existing.get("title"),
                            file_type=existing.get("file_type"), chunks=existing.get("chunks", 0))
    try:
        parsed = parse_document(data, filename, content_type)
    except ValueError as exc:  # UnsupportedFormat / ParseError
        return IngestResult(status="failed", source=source, error=str(exc))
    except Exception as exc:  # noqa: BLE001 - never let one bad file kill a batch
        log.exception("parse failed for %s", source)
        return IngestResult(status="failed", source=source, error=f"{type(exc).__name__}: {exc}")

    doc_title = title or parsed.title or Path(filename).stem.replace("_", " ") or source
    chunks, ids = build_chunks(parsed, doc_id, doc_title, source)
    if not chunks:
        return IngestResult(status="failed", source=source, error="No text chunks produced")
    kb.add_chunks(chunks, ids)
    log.info("ingested %s: %s chunks (%s)", source, len(chunks), parsed.file_type)
    return IngestResult(status="added", source=source, doc_id=doc_id, title=doc_title,
                        file_type=parsed.file_type, chunks=len(chunks), ocr_pages=parsed.ocr_pages)


def is_blocked_ip(address: str) -> bool:
    """Private, loopback, link-local, reserved, multicast or unspecified addresses (incl. IPv4-mapped IPv6)."""
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return True
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
            or ip.is_multicast or ip.is_unspecified)


def _assert_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise IngestionError("Only http(s) URLs are supported")
    if settings.allow_private_urls:
        return
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as exc:
        raise IngestionError(f"Cannot resolve host {parsed.hostname}") from exc
    for info in infos:
        if is_blocked_ip(str(info[4][0])):
            raise IngestionError("URLs pointing to private or local network addresses are blocked")


def ingest_url(kb, url: str) -> IngestResult:
    try:
        _assert_public_url(url)
        with httpx.Client(follow_redirects=True, timeout=30,
                          headers={"User-Agent": "Mozilla/5.0 (research-assistant ingestion)"}) as client:
            resp = client.get(url)
            _assert_public_url(str(resp.url))  # redirects must stay public too
            resp.raise_for_status()
    except IngestionError as exc:
        return IngestResult(status="failed", source=url, error=str(exc))
    except httpx.HTTPError as exc:
        return IngestResult(status="failed", source=url, error=f"Download failed: {exc}")
    name = Path(urlparse(str(resp.url)).path).name or "page.html"
    return ingest_bytes(kb, resp.content, filename=name, content_type=resp.headers.get("content-type"), source=url)


def ingest_text(kb, text: str, title: str | None = None) -> IngestResult:
    name = (title or "pasted text").strip()
    return ingest_bytes(kb, text.encode("utf-8"), filename=f"{name}.txt", content_type="text/plain",
                        title=name, source=f"text:{name}")


def seed_directory(kb, directory: str | Path) -> list[IngestResult]:
    results = []
    for path in sorted(Path(directory).glob("*")):
        if path.is_file() and not path.name.startswith("."):
            results.append(ingest_bytes(kb, path.read_bytes(), filename=path.name))
    return results
