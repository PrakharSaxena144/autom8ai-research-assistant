"""Format detection and parsing. Needs only the parsing libraries (and tesseract for the OCR tests)."""

import json
import shutil

import pytest

from app.ingestion.parsers import (
    ParseError,
    UnsupportedFormat,
    _looks_like_heading,
    detect_type,
    parse_document,
    rows_to_lines,
    split_sections,
)
from tests.conftest import CORPUS

needs_ocr = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")


def _all_text(parsed) -> str:
    return "\n".join(b.text for b in parsed.blocks)


# ---- detection ------------------------------------------------------------------------
@pytest.mark.parametrize(
    "data,filename,content_type,expected",
    [
        (b"%PDF-1.7 ...", "whatever.bin", None, "pdf"),           # magic bytes beat the extension
        (b"\x89PNG\r\n\x1a\n....", "scan", None, "image"),
        (b"\xff\xd8\xff\xe0....", "", None, "image"),
        (b"a,b\n1,2\n", "table.csv", None, "csv"),
        (b"a\tb\n1\t2\n", "table.tsv", None, "tsv"),
        (b"# Title\ntext", "notes.md", None, "markdown"),
        (b'{"a": 1}', "", "application/json", "json"),
        (b"<html><body><p>hi</p></body></html>", "", None, "html"),  # sniffed from content
        (b"plain words only", "", None, "text"),
    ],
)
def test_detect_type(data, filename, content_type, expected):
    assert detect_type(data, filename, content_type) == expected


def test_detect_rejects_binary_and_non_docx_zip():
    with pytest.raises(UnsupportedFormat):
        detect_type(bytes(range(256)) * 10, "blob")
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("xl/workbook.xml", "<x/>")
    with pytest.raises(UnsupportedFormat):
        detect_type(buf.getvalue(), "book.xlsx")


def test_empty_and_textless_inputs_raise_parse_error():
    with pytest.raises(ParseError):
        parse_document(b"", "empty.txt")
    with pytest.raises(ParseError):
        parse_document(b"   \n  ", "blank.txt")


# ---- helpers --------------------------------------------------------------------------
def test_heading_heuristics():
    assert _looks_like_heading("## Parental leave") == "Parental leave"
    assert _looks_like_heading("3. SUPPLY CHAIN: LX-9 LIDAR SECOND SOURCE") is not None
    assert _looks_like_heading("5.1 Supply concentration") == "5.1 Supply concentration"
    assert _looks_like_heading("This is an ordinary sentence.") is None
    assert _looks_like_heading("Hardware revenue | 151.2 | 184.9") is None
    assert _looks_like_heading("Root cause", prev_blank=True, next_blank=True) == "Root cause"
    assert _looks_like_heading("Root cause", prev_blank=False, next_blank=False) is None


def test_split_sections_joins_consecutive_headings():
    text = "# Report\n## Risks\nSupply is concentrated.\n## Outlook\nGrowth expected."
    sections = split_sections(text)
    assert sections == [("Report > Risks", "Supply is concentrated."), ("Outlook", "Growth expected.")]


def test_rows_to_lines_keeps_column_context():
    rows = [["Specification", "Porter", "Atlas"], ["Rated payload", "400 kg", "700 kg"]]
    assert rows_to_lines(rows, caption="Specs") == ["Table: Specs", "Rated payload: Porter = 400 kg; Atlas = 700 kg"]
    records = rows_to_lines([["id", "country"], ["SUP-117", "Austria"]], row_labels=False)
    assert records == ["id = SUP-117; country = Austria"]


def test_csv_blocks_repeat_header_every_group():
    rows = ["id,value"] + [f"R{i},{i}" for i in range(60)]
    parsed = parse_document("\n".join(rows).encode(), "big.csv")
    assert len(parsed.blocks) == 3
    assert all(b.text.startswith("Columns: id, value") for b in parsed.blocks)


def test_json_record_lists_become_one_block_per_record():
    data = {"source": "help centre", "articles": [{"id": "KB-1", "answer": "a"}, {"id": "KB-2", "answer": "b"}]}
    parsed = parse_document(json.dumps(data).encode(), "faq.json")
    sections = [b.section for b in parsed.blocks]
    assert sections == ["metadata", "articles #1", "articles #2"]


def test_invalid_json_falls_back_to_text():
    parsed = parse_document(b"{not json at all, but readable text}", "broken.json")
    assert parsed.file_type == "text"


# ---- sample corpus --------------------------------------------------------------------
def _corpus(name: str):
    path = CORPUS / name
    if not path.exists():
        pytest.skip(f"{name} missing (run scripts/generate_corpus.py)")
    return parse_document(path.read_bytes(), path.name)


def test_docx_handbook_sections_and_table():
    parsed = _corpus("Employee_Handbook_2024.docx")
    assert parsed.file_type == "docx"
    text = _all_text(parsed)
    assert "16 weeks" in text
    assert any("Remote work" in (b.section or "") for b in parsed.blocks)
    assert "Austin" in text and "USD 220" in text  # hotel cap table survived


def test_markdown_memo():
    parsed = _corpus("Policy_Update_Memo_2025.md")
    assert parsed.title == "Memo: People policy update 2025"
    assert any(b.section == "Parental leave" for b in parsed.blocks)


def test_html_specs_drop_boilerplate_and_keep_tables():
    parsed = _corpus("product_specifications.html")
    text = _all_text(parsed)
    assert "analytics placeholder" not in text
    assert "Rated payload" in text and "700 kg" in text


def test_csv_register_rows_are_self_describing():
    parsed = _corpus("supplier_register_2024-10.csv")
    text = _all_text(parsed)
    assert "supplier_id = SUP-117" in text and "contract_end = 2026-11-30" in text


def test_json_faq():
    parsed = _corpus("support_faq_export.json")
    assert any("36-month" in b.text for b in parsed.blocks)


def test_text_board_minutes_sections():
    parsed = _corpus("board_meeting_minutes_2025-06-26.txt")
    assert any("KESTREL" in (b.section or "") for b in parsed.blocks)


def test_pdf_table_rows_keep_columns():
    parsed = _corpus("Nimbus_Annual_Report_FY2025.pdf")
    text = _all_text(parsed)
    assert "SUP-117" in text
    assert "151.2" in text and "184.9" in text


@needs_ocr
def test_pdf_scanned_page_is_ocrd():
    parsed = _corpus("Nimbus_Annual_Report_FY2025.pdf")
    assert parsed.ocr_pages == [4]
    ocr_text = " ".join(b.text for b in parsed.blocks if b.page == 4)
    assert "Keller" in ocr_text and "unqualified" in ocr_text


@needs_ocr
def test_image_incident_report_is_ocrd():
    parsed = _corpus("Field_Incident_Report_IR-2025-031_scan.png")
    assert parsed.file_type == "image"
    text = _all_text(parsed)
    assert any("IR-2025-031" in (b.section or "") for b in parsed.blocks)  # heading becomes the section
    assert [b.section for b in parsed.blocks][1:] == ["Description", "Root cause", "Corrective action"]
    assert "4.2.3" in text
    assert "frost" in text.lower()
