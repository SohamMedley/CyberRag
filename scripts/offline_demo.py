"""Run CYBER-RAG fully offline for UI work and demos.

    python scripts/offline_demo.py            # http://127.0.0.1:5055
    PORT=8080 python scripts/offline_demo.py

This swaps the real embedding model for a deterministic hashing embedder and
Groq for a canned answer generator, so the whole upload -> index -> ask ->
delete flow works without downloading a model or spending API tokens.
It never touches the real data/ library: everything lives in a temp folder.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import Settings  # noqa: E402
from app import create_app  # noqa: E402

TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Tiny deterministic stand-in for sentence-transformers."""

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def load(self):
        return self

    @property
    def is_loaded(self) -> bool:
        return True

    @property
    def model_name(self) -> str:
        return "offline-hashing-embedder"

    @property
    def dimension(self) -> int:
        return self.dim

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in TOKEN_RE.findall(str(text).lower()):
            vector[hash(token) % self.dim] += 1.0
        return (vector / (float(np.linalg.norm(vector)) or 1.0)).astype(np.float32)

    def encode(self, texts, batch_size: int = 32) -> np.ndarray:
        items = [str(text) for text in texts]
        if not items:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._vector(text) for text in items]).astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])


class OfflineLLM:
    """Echoes the best matching passage instead of calling Groq."""

    api_key_set = True
    current_model = "offline/canned-answer"

    def resolve_model(self, force: bool = False) -> str:
        return self.current_model

    def answer(self, question, chunks, history=None):
        passage = chunks[0] if chunks else {}
        text = (passage.get("text") or "")[:400]
        source = passage.get("doc_name", "unknown")
        page = passage.get("page_number")
        where = f"page {page}" if page else "full text"
        answer = (
            f"[offline demo] The most relevant passage comes from **{source}** ({where}).\n\n"
            f"> {text}...\n\n"
            "Run the app with a real GROQ_API_KEY and the full embedding model to get a "
            "written answer grounded in this passage."
        )
        return {
            "answer": answer,
            "model": self.current_model,
            "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
            "passages": list(chunks),
            "context_passages": len(chunks),
            "context_chars": sum(len(c.get("text", "")) for c in chunks),
            "truncated": False,
            "latency_ms": 0,
        }


def build_app(workdir: Optional[Path] = None):
    """Create the offline demo app (importable, so gunicorn can serve it)."""
    sandbox = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="cyberrag-offline-"))

    settings = Settings(
        data_dir=sandbox / "data",
        upload_dir=sandbox / "uploads",
        groq_api_key="offline",
        groq_model="offline/canned-answer",
        auto_resolve_groq_model=False,
        resolve_model_on_boot=False,
        min_similarity_score=0.05,
    )

    application = create_app(settings)
    application.extensions["cyberrag"].embedder = HashingEmbedder()
    application.extensions["cyberrag"].llm = OfflineLLM()
    return application, sandbox


#: WSGI entry point: ``gunicorn --bind 0.0.0.0:5055 scripts.offline_demo:application``
application, _SANDBOX = build_app()


def main() -> None:
    logging.basicConfig(level="INFO", format="%(asctime)s [%(levelname)s] %(message)s")
    port = int(os.getenv("PORT", "5055"))
    print("\n" + "=" * 65)
    print("  CYBER-RAG // OFFLINE DEMO (no model download, no Groq calls)")
    print(f"  * http://127.0.0.1:{port}")
    print(f"  * library lives in: {_SANDBOX}")
    print("=" * 65 + "\n")
    application.run(host="0.0.0.0", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
