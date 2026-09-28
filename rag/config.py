"""Environment driven configuration.

Every path and tunable lives here so the same code can run:

* locally, with the classic ``uploads/`` + ``data/`` folders in the repo, and
* on a host such as Render, where ``DATA_DIR`` / ``UPLOAD_DIR`` point at a
  mounted persistent disk so the FAISS index survives restarts and deploys.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent


def _env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None else value.strip()


def _env_int(name: str, default: int) -> int:
    raw = _env_str(name)
    if not raw:
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    raw = _env_str(name)
    if not raw:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_path(name: str, default: Path) -> Path:
    """Read a path from the environment; empty/blank values mean 'use default'.

    Without this, ``DATA_DIR=`` (as in ``.env.example``) would resolve to the
    current working directory and scatter index files across the project root.
    """
    raw = _env_str(name)
    return Path(raw).expanduser() if raw else default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env_str(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "t", "yes", "y", "on"}


def slugify_doc_id(filename: str) -> str:
    """Turn a filename into a filesystem/URL friendly document id."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", filename)


@dataclass
class Settings:
    """Runtime configuration. Build one with :meth:`from_env`."""

    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    upload_dir: Path = BASE_DIR / "uploads"

    allowed_extensions: set = field(default_factory=lambda: {"pdf", "txt"})
    max_content_length: int = 16 * 1024 * 1024  # 16 MB per request

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_threads: Optional[int] = None
    embedding_device: str = "cpu"
    warmup_embeddings: bool = False

    chunk_size_words: int = 350
    chunk_overlap_words: int = 50
    top_k_retrieval: int = 4
    min_similarity_score: float = 0.15

    groq_api_key: str = ""
    groq_model: str = ""
    groq_fallback_models: List[str] = field(default_factory=list)
    groq_reasoning_effort: str = "low"  # gpt-oss style models only
    groq_timeout_seconds: float = 60.0
    groq_max_retries: int = 2
    max_answer_tokens: int = 700
    max_context_tokens: int = 3000
    max_history_turns: int = 3
    conversation_history_enabled: bool = True

    flask_port: int = 5000
    flask_debug: bool = False
    auto_resolve_groq_model: bool = True
    resolve_model_on_boot: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = _env_path("DATA_DIR", BASE_DIR / "data")
        upload_dir = _env_path("UPLOAD_DIR", BASE_DIR / "uploads")

        fallback_models = [
            m.strip()
            for m in _env_str(
                "GROQ_FALLBACK_MODELS",
                "openai/gpt-oss-20b,llama-3.1-8b-instant",
            ).split(",")
            if m.strip()
        ]

        return cls(
            base_dir=BASE_DIR,
            data_dir=data_dir,
            upload_dir=upload_dir,
            max_content_length=_env_int("MAX_CONTENT_LENGTH_MB", 16) * 1024 * 1024,
            embedding_model_name=_env_str(
                "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            embedding_threads=_env_int("EMBEDDING_THREADS", 0) or None,
            embedding_device=_env_str("EMBEDDING_DEVICE", "cpu") or "cpu",
            warmup_embeddings=_env_bool("WARMUP_EMBEDDINGS", False),
            chunk_size_words=_env_int("CHUNK_SIZE_WORDS", 350),
            chunk_overlap_words=_env_int("CHUNK_OVERLAP_WORDS", 50),
            top_k_retrieval=_env_int("TOP_K_RETRIEVAL", 4),
            min_similarity_score=_env_float("MIN_SIMILARITY_SCORE", 0.15),
            groq_api_key=_env_str("GROQ_API_KEY"),
            groq_model=_env_str("GROQ_MODEL"),
            groq_fallback_models=fallback_models,
            groq_reasoning_effort=_env_str("GROQ_REASONING_EFFORT", "low").lower(),
            groq_timeout_seconds=_env_float("GROQ_TIMEOUT_SECONDS", 60.0),
            groq_max_retries=_env_int("GROQ_MAX_RETRIES", 2),
            max_answer_tokens=_env_int("MAX_ANSWER_TOKENS", 700),
            max_context_tokens=_env_int("MAX_CONTEXT_TOKENS", 3000),
            max_history_turns=_env_int("MAX_HISTORY_TURNS", 3),
            conversation_history_enabled=_env_bool("CONVERSATION_HISTORY", True),
            flask_port=_env_int("PORT", _env_int("FLASK_PORT", 5000)),
            flask_debug=_env_bool("FLASK_DEBUG", False),
            auto_resolve_groq_model=_env_bool("GROQ_AUTO_RESOLVE_MODEL", True),
            resolve_model_on_boot=_env_bool("GROQ_RESOLVE_ON_BOOT", False),
        )

    # ------------------------------------------------------------------ paths
    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def documents_dir(self) -> Path:
        return self.data_dir / "documents"

    @property
    def faiss_index_path(self) -> Path:
        return self.index_dir / "faiss_index.bin"

    @property
    def metadata_path(self) -> Path:
        return self.index_dir / "chunks_metadata.json"

    @property
    def document_summary_path(self) -> Path:
        return self.index_dir / "document_summary.json"

    def ensure_directories(self) -> None:
        for directory in (self.data_dir, self.index_dir, self.documents_dir, self.upload_dir):
            directory.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------- helpers
    def is_allowed_file(self, filename: str) -> bool:
        return "." in filename and filename.rsplit(".", 1)[1].lower() in self.allowed_extensions

    def uses_persistent_disk(self) -> bool:
        """True when the data/upload folders live outside the git checkout."""
        try:
            return not self.data_dir.resolve().is_relative_to(self.base_dir.resolve())
        except AttributeError:  # pragma: no cover - Python < 3.9 fallback
            return not str(self.data_dir.resolve()).startswith(str(self.base_dir.resolve()))

    def public_dict(self) -> dict:
        return {
            "embedding_model": self.embedding_model_name,
            "chunk_size_words": self.chunk_size_words,
            "chunk_overlap_words": self.chunk_overlap_words,
            "top_k_retrieval": self.top_k_retrieval,
            "min_similarity_score": self.min_similarity_score,
            "max_context_tokens": self.max_context_tokens,
            "max_answer_tokens": self.max_answer_tokens,
            "history_enabled": self.conversation_history_enabled,
            "data_dir": str(self.data_dir),
            "upload_dir": str(self.upload_dir),
            "persistent_storage": self.uses_persistent_disk(),
        }
