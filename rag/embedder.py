"""Lazily loaded sentence-transformers embedder.

The model is heavy (~90 MB download, a few hundred MB of RAM), so it is only
built on first use. On a host such as Render this keeps the web process booting
in under a second instead of blocking the port while torch imports.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import List, Optional, Sequence

import numpy as np

from .config import Settings

logger = logging.getLogger("CYBER-RAG.embedder")


class Embedder:
    """Thin, thread-safe wrapper around ``SentenceTransformer``."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()

    # --------------------------------------------------------------- loading
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_name(self) -> str:
        return self.settings.embedding_model_name

    def load(self):
        """Import torch / sentence-transformers and build the model once."""
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model

            # Cache the downloaded weights inside the mounted disk when present,
            # so a redeploy does not re-download the model.
            cache_dir = os.getenv("HF_HOME") or os.getenv("SENTENCE_TRANSFORMERS_HOME")
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                logger.info("Using model cache directory: %s", cache_dir)

            if self.settings.embedding_threads:
                try:
                    import torch

                    torch.set_num_threads(int(self.settings.embedding_threads))
                    logger.info("torch intra-op threads set to %s", self.settings.embedding_threads)
                except Exception as exc:  # pragma: no cover - env dependent
                    logger.warning("Could not set torch threads: %s", exc)

            from sentence_transformers import SentenceTransformer

            logger.info("Loading Sentence Transformer: %s", self.model_name)
            self._model = SentenceTransformer(
                self.model_name, device=self.settings.embedding_device or "cpu"
            )
            logger.info(
                "Embedding model ready (%s dims).",
                self._model.get_sentence_embedding_dimension(),
            )
            return self._model

    def set_model_for_testing(self, model) -> None:
        """Inject a stub model (used by the test-suite)."""
        self._model = model

    # ------------------------------------------------------------- dimension
    @property
    def dimension(self) -> Optional[int]:
        if self._model is None:
            return None
        try:
            return int(self._model.get_sentence_embedding_dimension())
        except AttributeError:  # custom test doubles
            return int(len(self.encode(["dimension probe"])[0]))

    # -------------------------------------------------------------- encoding
    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """Embed texts and return **L2 normalised** float32 vectors."""
        items = ["" if t is None else str(t) for t in texts]
        if not items:
            dim = self.dimension or 384
            return np.zeros((0, dim), dtype=np.float32)

        model = self.load()
        with self._lock:  # one encode at a time keeps RAM/CPU predictable
            vectors = model.encode(
                items,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        # Defensive: guarantee unit length even for custom models.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vectors / norms).astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])
