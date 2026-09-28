"""Shared test fixtures.

The suite never touches the real ``data/`` folder, never downloads the
sentence-transformer model and never calls Groq: a deterministic stub embedder
and a stub LLM keep every test fast and offline.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep the module level ``app = create_app()`` (app.py) inside a throwaway
# sandbox and make sure it never tries to reach Groq.
_SANDBOX = Path(tempfile.mkdtemp(prefix="cyberrag-import-"))
os.environ.setdefault("DATA_DIR", str(_SANDBOX / "data"))
os.environ.setdefault("UPLOAD_DIR", str(_SANDBOX / "uploads"))
os.environ["GROQ_AUTO_RESOLVE_MODEL"] = "false"
os.environ["GROQ_RESOLVE_ON_BOOT"] = "false"
os.environ["GROQ_API_KEY"] = ""
os.environ["WARMUP_EMBEDDINGS"] = "false"

from rag.config import Settings  # noqa: E402

TOKEN_RE = re.compile(r"[a-z0-9]+")


class DeterministicEmbedder:
    """Hashing bag-of-words embedder with unit-norm vectors.

    Texts that share words end up close in cosine space, which is enough to
    exercise the whole retrieval stack deterministically.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    # API compatible with rag.embedder.Embedder
    def load(self):  # pragma: no cover - trivial
        return self

    @property
    def is_loaded(self) -> bool:
        return True

    @property
    def model_name(self) -> str:
        return "stub-embedder"

    @property
    def dimension(self) -> int:
        return self.dim

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in TOKEN_RE.findall(str(text).lower()):
            vector[hash(token) % self.dim] += 1.0
        norm = float(np.linalg.norm(vector)) or 1.0
        return (vector / norm).astype(np.float32)

    def encode(self, texts, batch_size: int = 32) -> np.ndarray:
        items = [str(t) for t in texts]
        self.calls.append(items)
        if not items:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._vector(text) for text in items]).astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])


class StubLLM:
    """Stand-in for :class:`rag.llm.GroqLLM` with no network access."""

    def __init__(self, answer: str = "Answer grounded in [Passage 1] (page 1).", error: Exception | None = None):
        self.answer_text = answer
        self.error = error
        self.calls: list[dict] = []

    @property
    def api_key_set(self) -> bool:
        return True

    @property
    def current_model(self) -> str:
        return "test/model"

    def resolve_model(self, force: bool = False) -> str:
        return "test/model"

    def answer(self, question, chunks, history=None):
        self.calls.append({"question": question, "chunks": list(chunks or []), "history": history})
        if self.error:
            raise self.error
        return {
            "answer": self.answer_text,
            "model": "test/model",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "passages": list(chunks or []),
            "context_passages": len(chunks or []),
            "context_chars": sum(len(c.get("text", "")) for c in chunks or []),
            "truncated": False,
            "latency_ms": 1,
        }


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    config = Settings(
        base_dir=ROOT,
        data_dir=tmp_path / "data",
        upload_dir=tmp_path / "uploads",
        groq_api_key="test-key",
        groq_model="test/model",
        auto_resolve_groq_model=False,
        resolve_model_on_boot=False,
        warmup_embeddings=False,
        chunk_size_words=120,
        chunk_overlap_words=20,
        top_k_retrieval=3,
        min_similarity_score=0.05,
    )
    config.ensure_directories()
    return config


@pytest.fixture()
def embedder() -> DeterministicEmbedder:
    return DeterministicEmbedder()


@pytest.fixture()
def store(settings: Settings):
    from rag.store import VectorStore

    return VectorStore(settings).load()


@pytest.fixture()
def app(settings: Settings, embedder: DeterministicEmbedder, monkeypatch):
    """Flask app with stubbed embedder + LLM and a throwaway data directory."""
    from app import create_app

    application = create_app(settings)
    services = application.extensions["cyberrag"]
    services.embedder = embedder
    services.llm = StubLLM()
    application.config.update(TESTING=True, PROPAGATE_EXCEPTIONS=False)
    return application


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def services(app):
    return app.extensions["cyberrag"]


# --------------------------------------------------------------------------- pdf
def build_pdf(pages: list[str]) -> bytes:
    """Build a tiny, valid, text-based PDF (no external dependencies)."""
    objects: list[bytes] = []
    page_ids: list[int] = []
    font_id = 3 + 2 * len(pages)

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kid_refs = " ".join(f"{3 + 2 * i} 0 R" for i in range(len(pages)))
    objects.append(f"<< /Type /Pages /Kids [{kid_refs}] /Count {len(pages)} >>".encode())

    for index, text in enumerate(pages):
        page_id = 3 + 2 * index
        content_id = page_id + 1
        page_ids.append(page_id)
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
            ).encode()
        )
        escaped = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode()
        objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")

    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_offset = len(out)
    size = len(objects) + 1
    out += f"xref\n0 {size}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode()
    return bytes(out)


@pytest.fixture()
def pdf_bytes() -> bytes:
    return build_pdf(
        [
            "Attendance policy: a student whose attendance falls below sixty five percent "
            "must submit a medical certificate within five working days.",
            "HPC laboratory rules: training jobs are limited to one hour per user per day.",
        ]
    )
