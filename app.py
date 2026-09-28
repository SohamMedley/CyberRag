"""CYBER-RAG - Flask entrypoint.

Run locally::

    python app.py                 # http://127.0.0.1:5000
    flask --app app run --debug   # development reloader

Run in production (Render, Railway, Fly, ...)::

    gunicorn -c gunicorn.conf.py app:app

Endpoints
---------
``GET    /``                      single page app
``GET    /status``                index + configuration summary
``POST   /upload``                stage one or more PDF/TXT files
``POST   /index``                 extract, embed and index staged files
``POST   /ask``                   retrieve + answer a question with Groq
``DELETE /documents/<filename>``  delete a single indexed document
``POST   /documents/delete``      same, for clients that cannot send DELETE
``POST   /reset``                 clear the whole index and upload staging
``GET    /healthz``               health probe (used by Render)
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from flask import Flask, current_app, jsonify, render_template, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from werkzeug.utils import secure_filename

from rag import __version__
from rag.config import Settings
from rag.documents import build_chunks, extract_text_from_file
from rag.llm import INSUFFICIENT_CONTEXT_ANSWER, LLMConfigurationError, LLMUpstreamError
from rag.services import Services
from rag.store import DocumentNotFoundError

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("CYBER-RAG")

load_dotenv()  # .env (safe to commit: placeholders only)
load_dotenv(Path(__file__).resolve().parent / ".env.local", override=True)  # personal secrets, git-ignored


# ------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------
def _safe_child(directory: Path, filename: str) -> Optional[Path]:
    """Return ``directory/filename`` when it is a plain file name inside it."""
    if not filename or filename != secure_filename(filename):
        return None
    candidate = (directory / filename).resolve()
    try:
        if candidate.parent != Path(directory).resolve():
            return None
    except OSError:  # pragma: no cover - defensive
        return None
    return candidate


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _remove_document_files(settings: Settings, filename: str) -> List[str]:
    """Delete every on-disk copy of a document; returns the paths that were removed."""
    removed: List[str] = []
    for directory in (settings.upload_dir, settings.documents_dir):
        path = _safe_child(directory, filename)
        if path and path.exists():
            try:
                path.unlink()
                removed.append(str(path))
            except OSError as exc:  # pragma: no cover - disk issues
                logger.warning("Could not delete %s: %s", path, exc)
    return removed


def _services() -> Services:
    return current_app.extensions["cyberrag"]


def _embedding_failure(action: str, error: Exception):
    """Friendly response when the sentence-transformer model cannot be used."""
    logger.error("Embedding model unavailable while trying to %s: %s", action, error, exc_info=True)
    return (
        jsonify(
            {
                "error": (
                    "The embedding model is not available yet. The first request downloads "
                    f"'{_services().settings.embedding_model_name}' (~90 MB) from Hugging Face, "
                    "so the host needs outbound internet access once (or a pre-populated "
                    "HF_HOME cache). Details: " + str(error)
                )
            }
        ),
        503,
    )


def create_app(settings: Optional[Settings] = None) -> Flask:
    """Application factory. ``settings`` defaults to ``Settings.from_env()``."""
    settings = settings or Settings.from_env()

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = settings.max_content_length
    app.config["UPLOAD_FOLDER"] = str(settings.upload_dir)
    app.config["JSON_SORT_KEYS"] = False
    app.config["APP_VERSION"] = __version__

    services = Services.build(settings)
    app.extensions["cyberrag"] = services

    # ------------------------------------------------------------------ pages
    @app.route("/")
    def index():
        return render_template("index.html", version=__version__)

    @app.route("/healthz", methods=["GET"])
    def healthz():
        store = _services().store
        return jsonify({"status": "ok", "version": __version__, "chunks": store.total_chunks}), 200

    @app.route("/status", methods=["GET"])
    def get_status():
        svc = _services()
        store = svc.store
        payload: Dict[str, Any] = {
            "status": "success",
            "version": __version__,
            "embedding_model": svc.settings.embedding_model_name,
            "embedding_loaded": svc.embedder.is_loaded,
            "llm_model": svc.llm.current_model,
            "is_groq_key_set": svc.llm.api_key_set,
            "limits": svc.settings.public_dict(),
        }
        payload.update(store.stats())
        return jsonify(payload)

    # ----------------------------------------------------------------- upload
    @app.route("/upload", methods=["POST"])
    def upload_files():
        if "files" not in request.files:
            return jsonify({"error": "No file field found in the upload request."}), 400

        uploaded_files = [f for f in request.files.getlist("files") if f and f.filename]
        if not uploaded_files:
            return jsonify({"error": "No files were selected for upload."}), 400

        settings = _services().settings
        saved_files: List[str] = []
        warnings: List[str] = []

        for file in uploaded_files:
            filename = secure_filename(file.filename)
            if not filename:
                warnings.append(f"'{file.filename}' was rejected: unusable file name.")
                continue
            if not settings.is_allowed_file(filename):
                warnings.append(f"File '{filename}' rejected: only .pdf and .txt are supported.")
                continue

            save_path = _safe_child(settings.upload_dir, filename)
            if save_path is None:  # pragma: no cover - secure_filename already guards
                warnings.append(f"File '{filename}' rejected: unsafe file name.")
                continue

            try:
                file.save(str(save_path))
            except OSError as exc:
                logger.error("Could not save %s: %s", filename, exc)
                warnings.append(f"File '{filename}' could not be stored: {exc}")
                continue
            saved_files.append(filename)

        if not saved_files:
            return jsonify({"error": "No valid files uploaded.", "details": warnings}), 400

        logger.info("Staged %s file(s): %s", len(saved_files), ", ".join(saved_files))
        return (
            jsonify(
                {
                    "message": f"Successfully uploaded {len(saved_files)} file(s).",
                    "saved_files": saved_files,
                    "warnings": warnings,
                }
            ),
            200,
        )

    # ------------------------------------------------------------------ index
    @app.route("/index", methods=["POST"])
    def process_and_index():
        svc = _services()
        settings, store = svc.settings, svc.store

        payload = request.get_json(silent=True) or {}
        filenames: List[str] = payload.get("filenames") or [
            f.name
            for f in sorted(settings.upload_dir.iterdir())
            if f.is_file() and settings.is_allowed_file(f.name)
        ]
        replace = bool(payload.get("replace", False))

        if not filenames:
            return (
                jsonify(
                    {
                        "error": "No files are waiting to be indexed. Upload documents first - "
                        "an indexed document only needs re-indexing if you deleted it."
                    }
                ),
                400,
            )

        report: List[Dict[str, Any]] = []
        pending: List[Tuple[str, Path]] = []

        for raw_name in filenames:
            filename = secure_filename(str(raw_name))
            if not filename:
                report.append(
                    {"filename": str(raw_name), "status": "failed", "reason": "Unsafe or invalid file name."}
                )
                continue

            # An indexed document has been archived out of uploads/, so when a
            # re-index is requested (replace=true) fall back to that copy.
            candidates = [
                path
                for path in (
                    _safe_child(settings.upload_dir, filename),
                    _safe_child(settings.documents_dir, filename),
                )
                if path is not None
            ]
            file_path = next((path for path in candidates if path.exists()), None)
            if file_path is None:
                report.append({"filename": filename, "status": "failed", "reason": "File not found."})
                continue
            if store.has_document(filename) and not replace:
                report.append(
                    {
                        "filename": filename,
                        "status": "skipped",
                        "reason": "Already indexed. Delete it from the library to index it again.",
                    }
                )
                continue
            pending.append((filename, file_path))

        ready: List[Dict[str, Any]] = []

        for filename, file_path in pending:
            try:
                pages_data = extract_text_from_file(file_path, filename)
                if not pages_data:
                    report.append(
                        {
                            "filename": filename,
                            "status": "failed",
                            "reason": "No readable text found (empty or image-only PDF).",
                        }
                    )
                    continue

                chunks = build_chunks(
                    pages_data=pages_data,
                    filename=filename,
                    chunk_size=settings.chunk_size_words,
                    overlap=settings.chunk_overlap_words,
                )
                if not chunks:
                    report.append(
                        {
                            "filename": filename,
                            "status": "failed",
                            "reason": "Text extracted but yielded 0 chunks.",
                        }
                    )
                    continue

                ready.append(
                    {
                        "filename": filename,
                        "path": file_path,
                        "pages": len(pages_data),
                        "chunks": chunks,
                        "size_bytes": file_path.stat().st_size,
                    }
                )
            except Exception as exc:
                logger.error("Error processing %s: %s", filename, exc)
                report.append({"filename": filename, "status": "failed", "reason": str(exc)})

        if not ready:
            if any(item["status"] == "failed" for item in report):
                return (
                    jsonify(
                        {
                            "error": "Failed to create chunks from the uploaded documents.",
                            "report": report,
                        }
                    ),
                    400,
                )
            return (
                jsonify(
                    {
                        "message": "Nothing to index - every file is already in the library.",
                        "total_documents_indexed": store.total_documents,
                        "new_chunks_added": 0,
                        "total_chunks_in_index": store.total_chunks,
                        "report": report,
                    }
                ),
                200,
            )

        try:
            all_chunks = [chunk for source in ready for chunk in source["chunks"]]
            logger.info("Generating embeddings for %s chunk(s)...", len(all_chunks))
            try:
                embeddings = svc.embedder.encode([c["text"] for c in all_chunks])
            except Exception as exc:
                return _embedding_failure("index documents", exc)

            added_chunks = 0
            with svc.mutation_lock:
                offset = 0
                for source in ready:
                    count = len(source["chunks"])
                    document_vectors = embeddings[offset : offset + count]
                    offset += count

                    existed = store.has_document(source["filename"])
                    store.add_document(
                        filename=source["filename"],
                        chunks=source["chunks"],
                        embeddings=document_vectors,
                        pages=source["pages"],
                        size_bytes=source["size_bytes"],
                        indexed_at=_utc_now(),
                        replace=replace and existed,
                        reembed=svc.embedder.encode,
                    )
                    added_chunks += count
                    report.append(
                        {
                            "filename": source["filename"],
                            "status": "success",
                            "pages": source["pages"],
                            "chunks_generated": count,
                            "replaced": existed,
                        }
                    )

                # Archive the original and clear the staging copy so `uploads/`
                # only ever holds files that are waiting to be indexed.
                for source in ready:
                    staged = source["path"]
                    archive = _safe_child(settings.documents_dir, source["filename"])
                    if not staged.exists() or archive is None:
                        continue
                    if archive.resolve() == staged.resolve():
                        continue  # re-indexed straight from the archive
                    try:
                        archive.write_bytes(staged.read_bytes())
                        staged.unlink()
                    except OSError as exc:  # pragma: no cover - disk issues
                        logger.warning("Could not archive %s: %s", source["filename"], exc)

            return (
                jsonify(
                    {
                        "message": "Indexing completed successfully.",
                        "total_documents_indexed": store.total_documents,
                        "new_chunks_added": added_chunks,
                        "total_chunks_in_index": store.total_chunks,
                        "report": report,
                    }
                ),
                200,
            )
        except Exception as exc:
            logger.error("Vector indexing pipeline failed: %s", exc, exc_info=True)
            return jsonify({"error": f"Vector indexing failure: {exc}"}), 500

    # -------------------------------------------------------------------- ask
    @app.route("/ask", methods=["POST"])
    def ask_question():
        svc = _services()
        store = svc.store

        data = request.get_json(silent=True)
        if not data or "question" not in data:
            return jsonify({"error": "Invalid request. 'question' string field is required."}), 400

        question = str(data.get("question", "")).strip()
        if not question:
            return jsonify({"error": "Question field cannot be empty."}), 400
        if len(question) > 6000:
            return jsonify({"error": "Question is too long (6000 characters max)."}), 400

        if store.is_empty:
            return (
                jsonify(
                    {
                        "error": "Your library is empty. Upload and index documents before asking questions."
                    }
                ),
                400,
            )

        try:
            query_vector = svc.embedder.encode_one(question)
        except Exception as exc:
            return _embedding_failure("answer a question", exc)

        try:
            candidates = store.search(query_vector, k=svc.settings.top_k_retrieval)
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc, exc_info=True)
            return jsonify({"error": f"Could not search the index: {exc}"}), 500

        relevant = [c for c in candidates if c.get("similarity_score", 0) >= svc.settings.min_similarity_score]
        if not relevant:
            return (
                jsonify(
                    {
                        "question": question,
                        "answer": INSUFFICIENT_CONTEXT_ANSWER,
                        "sources": [],
                        "retrieved_context": candidates,
                        "grounded": False,
                        "model": None,
                        "usage": {},
                    }
                ),
                200,
            )

        try:
            result = svc.llm.answer(question, relevant, history=data.get("history"))
        except LLMConfigurationError as exc:
            return jsonify({"error": str(exc)}), 400
        except LLMUpstreamError as exc:
            return jsonify({"error": str(exc)}), 502
        except Exception as exc:  # pragma: no cover - unexpected
            logger.error("Unhandled /ask failure: %s", exc, exc_info=True)
            return jsonify({"error": "An internal error occurred while processing your query."}), 500

        sources_map: Dict[str, Dict[str, Any]] = {}
        for chunk in result.get("passages") or relevant:
            doc_name = chunk.get("doc_name", "unknown")
            page = chunk.get("page_number")
            reference = f"{doc_name} - Page {page}" if page else f"{doc_name} (full text)"
            sources_map.setdefault(
                reference,
                {"doc_name": doc_name, "page_number": page, "citation": reference},
            )

        return (
            jsonify(
                {
                    "question": question,
                    "answer": result["answer"],
                    "sources": list(sources_map.values()),
                    "retrieved_context": [
                        {
                            "chunk_id": c.get("chunk_id"),
                            "doc_name": c.get("doc_name"),
                            "page_number": c.get("page_number"),
                            "similarity_score": c.get("similarity_score"),
                            "text": c.get("text"),
                        }
                        for c in relevant
                    ],
                    "grounded": result["answer"].strip() != INSUFFICIENT_CONTEXT_ANSWER,
                    "model": result.get("model"),
                    "usage": result.get("usage", {}),
                    "latency_ms": result.get("latency_ms"),
                }
            ),
            200,
        )

    # --------------------------------------------------------------- deletion
    def _delete_document(filename: str):
        svc = _services()
        settings, store = svc.settings, svc.store

        safe_name = secure_filename(filename or "")
        if not safe_name or safe_name != filename:
            return jsonify({"error": "Invalid document name."}), 400
        if not settings.is_allowed_file(safe_name):
            return jsonify({"error": "Only .pdf and .txt documents can be deleted."}), 400

        if not store.has_document(safe_name):
            # Uploaded but never indexed (for example because indexing failed):
            # there is nothing in the vector store, so only clean up the disk.
            staged = _remove_document_files(settings, safe_name)
            if not staged:
                return jsonify({"error": f"'{safe_name}' is not in the index."}), 404
            logger.info("Removed un-indexed upload '%s' (%s file(s)).", safe_name, len(staged))
            return (
                jsonify(
                    {
                        "message": f"Removed the un-indexed upload '{safe_name}'.",
                        "deleted_document": safe_name,
                        "chunks_removed": 0,
                        "total_chunks_in_index": store.total_chunks,
                        "total_documents_indexed": store.total_documents,
                        "deleted_files": staged,
                    }
                ),
                200,
            )

        try:
            with svc.mutation_lock:
                result = store.delete_document(safe_name, reembed=svc.embedder.encode)

            removed_files = _remove_document_files(settings, safe_name)

            logger.info(
                "Deleted '%s' (%s chunks, %s files).",
                safe_name,
                result["chunks_removed"],
                len(removed_files),
            )
            return (
                jsonify(
                    {
                        "message": f"Removed '{safe_name}' from the library.",
                        "deleted_document": safe_name,
                        "chunks_removed": result["chunks_removed"],
                        "total_chunks_in_index": result["chunks_remaining"],
                        "total_documents_indexed": result["documents_remaining"],
                        "deleted_files": removed_files,
                    }
                ),
                200,
            )
        except DocumentNotFoundError:
            return jsonify({"error": f"'{safe_name}' is not in the index."}), 404
        except Exception as exc:
            logger.error("Failed to delete %s: %s", safe_name, exc, exc_info=True)
            return jsonify({"error": f"Failed to delete the document: {exc}"}), 500

    @app.route("/documents/<path:filename>", methods=["DELETE"])
    def delete_document(filename: str):
        return _delete_document(filename)

    @app.route("/documents/delete", methods=["POST"])
    def delete_document_post():
        payload = request.get_json(silent=True) or {}
        filename = payload.get("filename") or request.form.get("filename", "")
        if not filename:
            return jsonify({"error": "A 'filename' field is required."}), 400
        return _delete_document(str(filename))

    # ------------------------------------------------------------------ reset
    @app.route("/reset", methods=["POST"])
    def reset_index():
        svc = _services()
        try:
            with svc.mutation_lock:
                result = svc.store.reset()
                deleted_files = 0
                for directory in (svc.settings.upload_dir, svc.settings.documents_dir):
                    if not directory.exists():
                        continue
                    for entry in directory.iterdir():
                        if entry.is_file() and svc.settings.is_allowed_file(entry.name):
                            try:
                                entry.unlink()
                                deleted_files += 1
                            except OSError as exc:  # pragma: no cover
                                logger.warning("Could not delete %s: %s", entry, exc)
            logger.info("Reset: %s chunks, %s documents, %s files.", result["chunks_removed"], result["documents_removed"], deleted_files)
            return (
                jsonify(
                    {
                        "message": "Vector store and upload staging reset successfully.",
                        "chunks_removed": result["chunks_removed"],
                        "documents_removed": result["documents_removed"],
                        "files_deleted": deleted_files,
                    }
                ),
                200,
            )
        except Exception as exc:
            logger.error("Failed to reset: %s", exc, exc_info=True)
            return jsonify({"error": f"Failed to reset index: {exc}"}), 500

    # ------------------------------------------------------------- error JSON
    @app.errorhandler(RequestEntityTooLarge)
    def handle_large_payload(error):  # pragma: no cover - exercised via tests
        limit_mb = settings.max_content_length // (1024 * 1024)
        return (
            jsonify({"error": f"The request is larger than the {limit_mb} MB limit. Upload a smaller batch."}),
            413,
        )

    @app.errorhandler(HTTPException)
    def handle_http_error(error: HTTPException):
        if request.path.startswith(("/static/", "/documents/")) or request.path == "/":
            return error
        return jsonify({"error": error.description}), error.code

    @app.errorhandler(Exception)
    def handle_unexpected(error: Exception):  # pragma: no cover - safety net
        logger.error("Unhandled error on %s: %s", request.path, error, exc_info=True)
        if isinstance(error, HTTPException):
            return error
        return jsonify({"error": "An unexpected server error occurred."}), 500

    # ----------------------------------------------------------- warm up (opt)
    if settings.warmup_embeddings:
        def _warmup() -> None:
            try:
                services.embedder.load()
                logger.info("Embedding model warmed up in the background.")
            except Exception as exc:  # pragma: no cover - env dependent
                logger.warning("Embedding warmup failed: %s", exc)

        threading.Thread(target=_warmup, name="embedder-warmup", daemon=True).start()

    logger.info(
        "CYBER-RAG %s ready | data=%s uploads=%s | groq key %s",
        __version__,
        settings.data_dir,
        settings.upload_dir,
        "set" if settings.groq_api_key else "missing",
    )
    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("PORT", os.getenv("FLASK_PORT", "5000")))
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1", "t")
    print("\n" + "=" * 65)
    print("  CYBER-RAG // DOCUMENT INTELLIGENCE SYSTEM")
    print(f"  * Local URL: http://127.0.0.1:{port}")
    llm = app.extensions["cyberrag"].llm
    print(f"  * Embedding Model: {app.extensions['cyberrag'].settings.embedding_model_name}")
    print(f"  * Groq Model: {llm.current_model or 'not configured (set GROQ_API_KEY)'}")
    print("=" * 65 + "\n")
    app.run(host="0.0.0.0", port=port, debug=debug_mode, threaded=True)
