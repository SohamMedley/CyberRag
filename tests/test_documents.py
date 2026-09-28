"""Text extraction, cleaning and chunking."""

from __future__ import annotations

import pytest

from rag.documents import build_chunks, chunk_text_sliding_window, clean_text, extract_text_from_file
from tests.conftest import build_pdf


def test_clean_text_normalises_whitespace():
    assert clean_text("a\r\n\r\n\r\nb\xa0 c\t\td") == "a\n\nb c d"
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text("   padded   ") == "padded"


def test_extract_txt(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Hello\n\n\n\nWorld", encoding="utf-8")
    pages = extract_text_from_file(path, "notes.txt")
    assert pages == [{"page_number": None, "text": "Hello\n\nWorld"}]


def test_extract_empty_txt_returns_nothing(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("   \n\n  ", encoding="utf-8")
    assert extract_text_from_file(path, "empty.txt") == []


def test_extract_pdf_pages(tmp_path, pdf_bytes):
    path = tmp_path / "policy.pdf"
    path.write_bytes(pdf_bytes)
    pages = extract_text_from_file(path, "policy.pdf")
    assert [page["page_number"] for page in pages] == [1, 2]
    assert "Attendance policy" in pages[0]["text"]
    assert "medical certificate" in pages[0]["text"]
    assert "HPC laboratory rules" in pages[1]["text"]
    assert "Attendance policy" not in pages[1]["text"]


def test_extract_pdf_single_page(tmp_path):
    path = tmp_path / "single.pdf"
    path.write_bytes(build_pdf(["Only one page here."]))
    pages = extract_text_from_file(path, "single.pdf")
    assert len(pages) == 1
    assert pages[0]["page_number"] == 1
    assert "Only one page here." in pages[0]["text"]


def test_extract_rejects_unknown_extension(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,b", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file format"):
        extract_text_from_file(path, "data.csv")


def test_extract_rejects_corrupt_pdf(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.4 not really a pdf")
    with pytest.raises(ValueError):
        extract_text_from_file(path, "broken.pdf")


@pytest.mark.parametrize("size,overlap", [(10, 2), (5, 0), (50, 49)])
def test_chunking_uses_sliding_window(size, overlap):
    text = " ".join(f"word{i}" for i in range(60))
    chunks = chunk_text_sliding_window(
        [{"page_number": 1, "text": text}], "doc", "doc.txt", chunk_size=size, overlap=overlap
    )
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk["text"].split()) <= size
    assert [c["chunk_id"] for c in chunks] == [f"doc-c{i}" for i in range(1, len(chunks) + 1)]

    # Consecutive chunks share exactly the configured overlap.
    first = chunks[0]["text"].split()
    second = chunks[1]["text"].split()
    if overlap:
        assert first[-overlap:] == second[:overlap]
    else:
        assert first[-1] != second[0]


def test_short_document_becomes_one_chunk():
    chunks = chunk_text_sliding_window(
        [{"page_number": None, "text": "one two three"}], "doc", "doc.txt", chunk_size=100, overlap=10
    )
    assert len(chunks) == 1
    assert chunks[0]["page_number"] is None
    assert chunks[0]["doc_id"] == "doc"


def test_blank_pages_are_skipped():
    chunks = chunk_text_sliding_window(
        [{"page_number": 1, "text": "   "}, {"page_number": 2, "text": "real content"}],
        "doc",
        "doc.txt",
    )
    assert len(chunks) == 1
    assert chunks[0]["page_number"] == 2


def test_tiny_trailing_window_is_dropped():
    text = " ".join(f"w{i}" for i in range(105))
    chunks = chunk_text_sliding_window(
        [{"page_number": 1, "text": text}], "doc", "doc.txt", chunk_size=100, overlap=0
    )
    # The 5-word tail is discarded instead of creating a useless chunk.
    assert len(chunks) == 1
    assert len(chunks[0]["text"].split()) == 100


def test_invalid_chunk_settings():
    pages = [{"page_number": 1, "text": "some words here"}]
    with pytest.raises(ValueError):
        chunk_text_sliding_window(pages, "doc", "doc.txt", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_text_sliding_window(pages, "doc", "doc.txt", chunk_size=10, overlap=-1)
    # overlap >= chunk_size is clamped rather than crashing
    chunks = chunk_text_sliding_window(pages, "doc", "doc.txt", chunk_size=4, overlap=99)
    assert chunks


def test_build_chunks_derives_doc_id():
    chunks = build_chunks([{"page_number": None, "text": "alpha beta gamma"}], "My Doc!.txt", 10, 2)
    assert chunks[0]["doc_id"] == "My_Doc__txt"
    assert chunks[0]["doc_name"] == "My Doc!.txt"
