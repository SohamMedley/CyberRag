"""Text extraction, cleaning and chunking."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from pypdf import PdfReader

from .config import slugify_doc_id

# Windows smaller than this (in words) at the tail of a page are merged away
# because they are almost always page footers / stray numbers.
MIN_TAIL_WORDS = 30


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\xa0", " ").replace("\t", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()


def extract_text_from_file(file_path: str | Path, filename: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return ``[{"page_number": int | None, "text": str}, ...]`` for a PDF/TXT."""
    path = Path(file_path)
    name = filename or path.name
    if "." not in name:
        raise ValueError(f"Unsupported file format (no extension): {name}")
    ext = name.rsplit(".", 1)[1].lower()

    if ext == "pdf":
        pages_data: List[Dict[str, Any]] = []
        try:
            reader = PdfReader(str(path))
            if getattr(reader, "is_encrypted", False):
                try:
                    reader.decrypt("")
                except Exception as exc:  # pragma: no cover - depends on the file
                    raise ValueError("PDF is password protected and cannot be read.") from exc
            for idx, page in enumerate(reader.pages):
                cleaned = clean_text(page.extract_text() or "")
                if cleaned:
                    pages_data.append({"page_number": idx + 1, "text": cleaned})
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Corrupt or unreadable PDF: {exc}") from exc
        return pages_data

    if ext == "txt":
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            raise ValueError(f"Unable to read TXT file: {exc}") from exc
        cleaned = clean_text(content)
        return [{"page_number": None, "text": cleaned}] if cleaned else []

    raise ValueError(f"Unsupported file format: .{ext}")


def chunk_text_sliding_window(
    pages_data: List[Dict[str, Any]],
    doc_id: str,
    doc_name: str,
    chunk_size: int = 350,
    overlap: int = 50,
    min_tail_words: int = MIN_TAIL_WORDS,
) -> List[Dict[str, Any]]:
    """Sliding window chunking over each page, preserving page numbers.

    Chunk ids are deterministic: ``<doc_id>-c<index>``.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive number of words")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)

    chunks: List[Dict[str, Any]] = []
    counter = 0
    step = max(1, chunk_size - overlap)

    for page_entry in pages_data:
        page_num = page_entry.get("page_number")
        words = str(page_entry.get("text") or "").split()
        if not words:
            continue

        if len(words) <= chunk_size:
            counter += 1
            chunks.append(
                {
                    "chunk_id": f"{doc_id}-c{counter}",
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "page_number": page_num,
                    "text": " ".join(words),
                }
            )
            continue

        for start in range(0, len(words), step):
            window = words[start : start + chunk_size]
            if not window:
                break
            # Drop tiny trailing windows unless nothing has been emitted yet.
            if len(window) < min_tail_words and chunks and start + chunk_size >= len(words):
                continue
            counter += 1
            chunks.append(
                {
                    "chunk_id": f"{doc_id}-c{counter}",
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "page_number": page_num,
                    "text": " ".join(window),
                }
            )
            if start + chunk_size >= len(words):
                break

    return chunks


def build_chunks(
    pages_data: List[Dict[str, Any]],
    filename: str,
    chunk_size: int,
    overlap: int,
) -> List[Dict[str, Any]]:
    """Convenience wrapper used by the indexing endpoint."""
    return chunk_text_sliding_window(
        pages_data=pages_data,
        doc_id=slugify_doc_id(filename),
        doc_name=filename,
        chunk_size=chunk_size,
        overlap=overlap,
    )
