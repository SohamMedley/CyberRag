"""Service container: one place that builds and holds the app's singletons."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

from .config import Settings
from .embedder import Embedder
from .llm import GroqLLM
from .store import VectorStore

logger = logging.getLogger("CYBER-RAG.services")


@dataclass
class Services:
    settings: Settings
    store: VectorStore
    embedder: Embedder
    llm: GroqLLM
    #: Serialises index/delete/reset so two requests cannot corrupt the store.
    mutation_lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def build(cls, settings: Settings) -> "Services":
        settings.ensure_directories()
        embedder = Embedder(settings)
        store = VectorStore(settings).load()
        llm = GroqLLM(settings)
        services = cls(settings=settings, store=store, embedder=embedder, llm=llm)

        if not settings.groq_api_key:
            logger.warning("GROQ_API_KEY is not set - /ask will return a configuration error.")
        elif settings.resolve_model_on_boot:
            # Optional: only for deployments that prefer failing fast at boot.
            try:
                logger.info("Groq model resolved to: %s", llm.resolve_model())
            except Exception as exc:  # pragma: no cover - network dependent
                logger.warning("Could not resolve the Groq model at startup: %s", exc)
        else:
            logger.info(
                "Groq model '%s' will be validated on the first question (lazy).",
                llm.current_model,
            )
        return services
