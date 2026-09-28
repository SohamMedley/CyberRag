"""Persisted FAISS vector store with per-document add / delete support.

Design notes
------------
* The FAISS index is **lazy**: it is created on the first ``add_document`` call
  using the dimensionality of the embeddings that are passed in. Nothing needs
  the embedding model during boot, which keeps cold starts fast.
* Every mutation is written to disk atomically (temp file + ``os.replace``) so a
  container restart (Render redeploys, instance recycling) can never leave a
  truncated ``faiss_index.bin`` behind.
* Deleting one document rebuilds the flat index from the surviving vectors with
  ``IndexFlat.reconstruct_n`` -- no re-embedding required.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import faiss
import numpy as np

from .config import Settings, slugify_doc_id

logger = logging.getLogger("CYBER-RAG.store")


class DocumentNotFoundError(KeyError):
    """Raised when a document is not part of the index."""


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _atomic_write_index(path: Path, index: "faiss.Index") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    faiss.write_index(index, str(tmp))
    os.replace(tmp, path)


class VectorStore:
    """In-memory FAISS index + chunk metadata, persisted to ``data/index``."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.RLock()
        self.index: Optional["faiss.Index"] = None
        self.chunks: List[Dict[str, Any]] = []
        self.documents: List[Dict[str, Any]] = []
        self.load_error: Optional[str] = None

    # ------------------------------------------------------------------ load
    def load(self) -> "VectorStore":
        """Load the persisted index if it exists; otherwise start empty."""
        with self._lock:
            self.index = None
            self.chunks = []
            self.documents = []
            self.load_error = None

            index_path = self.settings.faiss_index_path
            meta_path = self.settings.metadata_path
            summary_path = self.settings.document_summary_path

            if not index_path.exists() or not meta_path.exists():
                logger.info("No persisted index found - starting with an empty library.")
                return self

            try:
                self.index = faiss.read_index(str(index_path))
                with open(meta_path, "r", encoding="utf-8") as handle:
                    raw_chunks = json.load(handle)
                if not isinstance(raw_chunks, list):
                    raise ValueError("chunks_metadata.json must contain a list")
                self.chunks = [self._normalise_chunk(c) for c in raw_chunks]

                if summary_path.exists():
                    with open(summary_path, "r", encoding="utf-8") as handle:
                        raw_docs = json.load(handle)
                    if isinstance(raw_docs, list):
                        self.documents = [self._normalise_document(d) for d in raw_docs]

                self._reconcile()
                logger.info(
                    "Index loaded: %s vectors, %s chunks, %s documents.",
                    getattr(self.index, "ntotal", 0),
                    len(self.chunks),
                    len(self.documents),
                )
            except Exception as exc:  # pragma: no cover - defensive
                self.load_error = str(exc)
                self.index = None
                self.chunks = []
                self.documents = []
                logger.error(
                    "Could not load the persisted index (%s). Starting empty; "
                    "re-index your documents to rebuild it.",
                    exc,
                )
            return self

    # ----------------------------------------------------------- normalising
    @staticmethod
    def _normalise_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
        chunk = dict(chunk or {})
        doc_name = str(chunk.get("doc_name") or "")
        chunk.setdefault("doc_name", doc_name)
        chunk.setdefault("doc_id", slugify_doc_id(doc_name))
        chunk.setdefault("chunk_id", f"{chunk['doc_id']}-c0")
        chunk.setdefault("page_number", None)
        chunk.setdefault("text", "")
        return chunk

    @staticmethod
    def _normalise_document(document: Dict[str, Any]) -> Dict[str, Any]:
        document = dict(document or {})
        filename = str(document.get("filename") or document.get("doc_name") or "")
        document["filename"] = filename
        document.setdefault("doc_id", slugify_doc_id(filename))
        document.setdefault("pages", 0)
        document.setdefault("chunks", 0)
        document.setdefault("indexed_at", None)
        document.setdefault("size_bytes", None)
        return document

    def _reconcile(self) -> None:
        """Make the document summary agree with the chunk metadata."""
        counts: Dict[str, int] = {}
        max_page: Dict[str, int] = {}
        for chunk in self.chunks:
            name = chunk["doc_name"]
            counts[name] = counts.get(name, 0) + 1
            page = chunk.get("page_number")
            if isinstance(page, int):
                max_page[name] = max(max_page.get(name, 0), page)

        # Drop summary rows whose chunks disappeared (stale state).
        cleaned = [d for d in self.documents if counts.get(d["filename"])]
        known = {d["filename"] for d in cleaned}

        # Rebuild counts for surviving rows.
        for document in cleaned:
            document["chunks"] = counts[document["filename"]]
            if not document.get("pages"):
                document["pages"] = max_page.get(document["filename"], 0) or 1

        # Adopt chunks that have no summary row (older index format).
        for name, count in counts.items():
            if name in known:
                continue
            logger.warning("Adopting orphaned chunks for %s into the document summary.", name)
            cleaned.append(
                {
                    "filename": name,
                    "doc_id": slugify_doc_id(name),
                    "pages": max_page.get(name, 0) or 1,
                    "chunks": count,
                    "indexed_at": None,
                    "size_bytes": None,
                }
            )

        if len(cleaned) != len(self.documents):
            logger.info(
                "Reconciled document summary: %s -> %s entries.",
                len(self.documents),
                len(cleaned),
            )
        self.documents = cleaned

    # ---------------------------------------------------------------- persist
    def persist(self) -> None:
        with self._lock:
            if self.index is not None:
                _atomic_write_index(self.settings.faiss_index_path, self.index)
            _atomic_write_json(self.settings.metadata_path, self.chunks)
            _atomic_write_json(self.settings.document_summary_path, self.documents)

    # ------------------------------------------------------------------- info
    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    @property
    def total_documents(self) -> int:
        return len(self.documents)

    @property
    def total_vectors(self) -> int:
        return int(getattr(self.index, "ntotal", 0) or 0)

    @property
    def is_empty(self) -> bool:
        return self.total_vectors == 0

    def dimension(self) -> Optional[int]:
        return int(self.index.d) if self.index is not None else None

    def has_document(self, filename: str) -> bool:
        return any(d["filename"] == filename for d in self.documents)

    def get_document(self, filename: str) -> Optional[Dict[str, Any]]:
        for document in self.documents:
            if document["filename"] == filename:
                return dict(document)
        return None

    def documents_public(self) -> List[Dict[str, Any]]:
        return [
            {
                "filename": d["filename"],
                "doc_id": d.get("doc_id"),
                "pages": d.get("pages", 0),
                "chunks": d.get("chunks", 0),
                "indexed_at": d.get("indexed_at"),
                "size_bytes": d.get("size_bytes"),
            }
            for d in self.documents
        ]

    def stats(self) -> Dict[str, Any]:
        return {
            "total_chunks_indexed": self.total_chunks,
            "total_vectors_indexed": self.total_vectors,
            "documents_count": len(self.documents),
            "indexed_documents": self.documents_public(),
            "index_dimension": self.dimension(),
            "index_load_error": self.load_error,
        }

    # -------------------------------------------------------------------- add
    def add_document(
        self,
        filename: str,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray,
        pages: int = 0,
        size_bytes: Optional[int] = None,
        indexed_at: Optional[str] = None,
        replace: bool = False,
        reembed: Optional[Callable[[List[str]], np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Append one document's chunks + vectors and persist everything.

        With ``replace=True`` an existing document of the same name is removed
        first, so re-indexing a changed file never leaves duplicate chunks.
        """
        if not chunks:
            raise ValueError("Cannot index a document without chunks.")
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError("Embedding matrix does not match the number of chunks.")

        with self._lock:
            if replace and self.has_document(filename):
                self._remove_locked(filename, reembed)

            if self.index is None:
                self.index = faiss.IndexFlatIP(vectors.shape[1])
            elif vectors.shape[1] != self.index.d:
                raise ValueError(
                    f"Embedding dimension changed ({vectors.shape[1]} != {self.index.d}). "
                    "Reset the index before switching embedding models."
                )

            self.index.add(vectors)
            self.chunks.extend(chunks)

            entry = {
                "filename": filename,
                "doc_id": slugify_doc_id(filename),
                "pages": pages,
                "chunks": len(chunks),
                "indexed_at": indexed_at,
                "size_bytes": size_bytes,
            }
            self.documents.append(entry)
            self.persist()
            return entry

    # ----------------------------------------------------------------- delete
    def delete_document(
        self,
        filename: str,
        reembed: Optional[Callable[[List[str]], np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Remove a document, its chunks and its vectors. Returns a summary."""
        with self._lock:
            return self._remove_locked(filename, reembed)

    def _remove_locked(
        self,
        filename: str,
        reembed: Optional[Callable[[List[str]], np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Delete a document while the lock is already held."""
        if self.get_document(filename) is None:
            raise DocumentNotFoundError(filename)

        keep_mask = [chunk["doc_name"] != filename for chunk in self.chunks]
        removed = len(self.chunks) - sum(keep_mask)

        if removed == 0:
            # Summary row without chunks: still clean it up.
            self.documents = [d for d in self.documents if d["filename"] != filename]
            self.persist()
            return {
                "filename": filename,
                "chunks_removed": 0,
                "chunks_remaining": len(self.chunks),
                "documents_remaining": len(self.documents),
            }

        # Rebuild the flat index from the vectors we want to keep.
        kept_vectors: Optional[np.ndarray] = None
        if self.index is not None and self.total_vectors == len(self.chunks):
            try:
                all_vectors = self.index.reconstruct_n(0, self.index.ntotal)
                kept_vectors = all_vectors[np.array(keep_mask, dtype=bool)]
            except Exception as exc:  # pragma: no cover - unusual index types
                logger.warning("Could not reconstruct vectors (%s); re-embedding instead.", exc)

        self.chunks = [chunk for chunk, keep in zip(self.chunks, keep_mask) if keep]
        self.documents = [d for d in self.documents if d["filename"] != filename]

        if not self.chunks:
            self.index = None
        elif kept_vectors is None:
            if reembed is None:
                raise RuntimeError(
                    "The vector index could not be rebuilt automatically. "
                    "Reset the index and re-upload your documents."
                )
            rebuilt = np.asarray(reembed([chunk["text"] for chunk in self.chunks]), dtype=np.float32)
            self.index = faiss.IndexFlatIP(rebuilt.shape[1])
            self.index.add(rebuilt)
        else:
            dimension = int(self.index.d)
            self.index = faiss.IndexFlatIP(dimension)
            if len(kept_vectors):
                self.index.add(np.ascontiguousarray(kept_vectors, dtype=np.float32))

        self.persist()
        return {
            "filename": filename,
            "chunks_removed": removed,
            "chunks_remaining": len(self.chunks),
            "documents_remaining": len(self.documents),
        }

    # ------------------------------------------------------------------ reset
    def reset(self) -> Dict[str, int]:
        """Drop every vector, chunk and document summary; delete index files."""
        with self._lock:
            removed_chunks = len(self.chunks)
            removed_documents = len(self.documents)

            self.index = None
            self.chunks = []
            self.documents = []
            self.load_error = None

            for path in (
                self.settings.faiss_index_path,
                self.settings.metadata_path,
                self.settings.document_summary_path,
            ):
                try:
                    if path.exists():
                        path.unlink()
                except OSError as exc:  # pragma: no cover - defensive
                    logger.warning("Could not delete %s: %s", path, exc)

            return {"chunks_removed": removed_chunks, "documents_removed": removed_documents}

    # ----------------------------------------------------------------- search
    def search(self, query_vector: np.ndarray, k: int) -> List[Dict[str, Any]]:
        with self._lock:
            if self.index is None or self.total_vectors == 0:
                return []
            vectors = np.asarray(query_vector, dtype=np.float32)
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
            k_to_retrieve = min(max(1, k), self.total_vectors)
            similarities, indices = self.index.search(vectors, k_to_retrieve)

            results: List[Dict[str, Any]] = []
            seen_texts: set = set()
            for score, idx in zip(similarities[0], indices[0]):
                if idx == -1 or idx >= len(self.chunks):
                    continue
                chunk = dict(self.chunks[idx])
                text = chunk.get("text", "")
                if text in seen_texts:  # de-duplicate identical passages
                    continue
                seen_texts.add(text)
                chunk["similarity_score"] = float(round(float(score), 4))
                results.append(chunk)
            return results
