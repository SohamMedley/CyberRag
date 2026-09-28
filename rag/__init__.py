"""CYBER-RAG core package.

Small, testable building blocks used by the Flask entrypoint (``app.py``):

* :mod:`rag.config`     - environment driven settings + paths
* :mod:`rag.documents`  - text extraction, cleaning and chunking
* :mod:`rag.embedder`   - lazily loaded sentence-transformer embedder
* :mod:`rag.store`      - persisted FAISS vector store (add / delete / reset)
* :mod:`rag.llm`        - Groq chat client with model auto-detection
* :mod:`rag.services`   - container that wires the pieces together per app
"""

__all__ = ["config", "documents", "embedder", "store", "llm", "services"]

__version__ = "2.0.0"
