"""Vector store behaviour: persistence, replacement, deletion, reconciliation."""

from __future__ import annotations

import json

import numpy as np
import pytest

from rag.store import DocumentNotFoundError, VectorStore
from tests.conftest import DeterministicEmbedder


def make_chunks(text: str, doc_name: str, split: bool = False):
    """Build one or two chunks for a document."""
    parts = [text, text + " second part"] if split else [text]
    return [
        {
            "chunk_id": f"{doc_name}-c{index}",
            "doc_id": doc_name,
            "doc_name": doc_name,
            "page_number": index + 1,
            "text": part,
        }
        for index, part in enumerate(parts)
    ]


@pytest.fixture()
def embedder():
    return DeterministicEmbedder()


def add(store: VectorStore, embedder: DeterministicEmbedder, name: str, text: str, split: bool = False, **kwargs):
    chunks = make_chunks(text, name, split=split)
    vectors = embedder.encode([chunk["text"] for chunk in chunks])
    return store.add_document(filename=name, chunks=chunks, embeddings=vectors, pages=1, **kwargs)


def test_add_and_persist_round_trip(store, embedder, settings):
    add(store, embedder, "alpha.txt", "attendance policy medical certificate", split=True)
    assert store.total_chunks == 2
    assert store.total_vectors == 2
    assert store.dimension()

    reloaded = VectorStore(settings).load()
    assert reloaded.total_chunks == 2
    assert [doc["filename"] for doc in reloaded.documents_public()] == ["alpha.txt"]
    assert reloaded.get_document("alpha.txt")["chunks"] == 2
    assert reloaded.load_error is None


def test_add_rejects_mismatched_matrix(store, embedder):
    chunks = make_chunks("hello world", "beta.txt", split=True)
    with pytest.raises(ValueError, match="does not match"):
        store.add_document("beta.txt", chunks, np.zeros((1, 8), dtype=np.float32))


def test_replace_does_not_duplicate_chunks(store, embedder):
    add(store, embedder, "alpha.txt", "first version of the policy")
    add(store, embedder, "alpha.txt", "second version of the policy", replace=True)
    assert store.total_chunks == 1
    assert store.total_vectors == 1
    assert len(store.documents) == 1


def test_delete_removes_only_that_document(store, embedder):
    add(store, embedder, "alpha.txt", "attendance policy medical certificate")
    add(store, embedder, "beta.txt", "hpc laboratory training job limits")

    result = store.delete_document("alpha.txt", reembed=embedder.encode)
    assert result["chunks_removed"] == 1
    assert result["chunks_remaining"] == 1
    assert result["documents_remaining"] == 1
    assert store.total_vectors == 1
    assert [doc["filename"] for doc in store.documents_public()] == ["beta.txt"]

    hits = store.search(embedder.encode_one("hpc laboratory training job limits"), k=3)
    assert hits and all(hit["doc_name"] == "beta.txt" for hit in hits)


def test_delete_keeps_index_consistent_for_multiple_docs(store, embedder):
    add(store, embedder, "a.txt", "alpha content about attendance", split=True)
    add(store, embedder, "b.txt", "beta content about hpc jobs", split=True)
    add(store, embedder, "c.txt", "gamma content about library rules", split=True)
    assert store.total_vectors == 6

    store.delete_document("b.txt", reembed=embedder.encode)
    assert store.total_chunks == 4
    assert store.total_vectors == 4  # vectors and chunks stay in sync
    assert store.dimension()

    hits = store.search(embedder.encode_one("beta content about hpc jobs"), k=5)
    assert hits and all(hit["doc_name"] != "b.txt" for hit in hits)


def test_delete_last_document_empties_index(store, embedder):
    add(store, embedder, "only.txt", "single document content")
    store.delete_document("only.txt", reembed=embedder.encode)
    assert store.is_empty
    assert store.total_chunks == 0
    assert store.index is None
    assert store.search(np.zeros((1, 8), dtype=np.float32), k=3) == []


def test_delete_unknown_document_raises(store, embedder):
    with pytest.raises(DocumentNotFoundError):
        store.delete_document("missing.txt")


def test_delete_without_reembed_uses_reconstruction(store, embedder):
    """The normal path rebuilds from stored vectors - no re-embedding needed."""
    add(store, embedder, "a.txt", "alpha content", split=True)
    add(store, embedder, "b.txt", "beta content", split=True)
    calls_before = len(embedder.calls)
    store.delete_document("a.txt")  # no reembed callable
    assert len(embedder.calls) == calls_before
    assert store.total_vectors == 2


def test_search_deduplicates_identical_text(store, embedder):
    chunks = make_chunks("identical passage text", "dup.txt", split=True)
    chunks[1]["text"] = chunks[0]["text"]
    store.add_document("dup.txt", chunks, embedder.encode([c["text"] for c in chunks]))
    hits = store.search(embedder.encode_one("identical passage text"), k=5)
    assert len(hits) == 1
    assert "similarity_score" in hits[0]


def test_search_returns_empty_for_empty_store(store, embedder):
    assert store.search(embedder.encode_one("anything"), k=4) == []


def test_search_respects_k(store, embedder):
    for index in range(5):
        add(store, embedder, f"doc{index}.txt", f"content number {index} about topic")
    hits = store.search(embedder.encode_one("content about topic"), k=2)
    assert len(hits) == 2


def test_reset_removes_data_and_files(store, embedder, settings):
    add(store, embedder, "a.txt", "alpha content")
    assert settings.faiss_index_path.exists()
    result = store.reset()
    assert result == {"chunks_removed": 1, "documents_removed": 1}
    assert store.is_empty and store.documents == []
    assert not settings.faiss_index_path.exists()
    assert not settings.metadata_path.exists()
    assert not settings.document_summary_path.exists()


def test_persist_is_atomic_and_leaves_no_temp_files(store, embedder, settings):
    add(store, embedder, "a.txt", "alpha content")
    leftovers = list(settings.index_dir.glob("*.tmp"))
    assert leftovers == []
    # JSON payloads stay valid after several mutations
    store.delete_document("a.txt", reembed=embedder.encode)
    json.loads(settings.metadata_path.read_text(encoding="utf-8"))
    json.loads(settings.document_summary_path.read_text(encoding="utf-8"))


def test_load_reconciles_stale_summary_rows(settings, embedder):
    """A summary row without chunks must not survive a reload."""
    settings.ensure_directories()
    settings.metadata_path.write_text(json.dumps([]), encoding="utf-8")
    settings.document_summary_path.write_text(
        json.dumps([{"filename": "ghost.txt", "pages": 1, "chunks": 3}]), encoding="utf-8"
    )
    import faiss

    faiss.write_index(faiss.IndexFlatIP(8), str(settings.faiss_index_path))

    store = VectorStore(settings).load()
    assert store.documents == []


def test_load_adopts_orphaned_chunks(settings):
    """Chunks without a summary row are adopted (older index format)."""
    settings.ensure_directories()
    chunks = [
        {
            "chunk_id": "legacy-c1",
            "doc_id": "legacy",
            "doc_name": "legacy.txt",
            "page_number": 2,
            "text": "legacy chunk",
        }
    ]
    settings.metadata_path.write_text(json.dumps(chunks), encoding="utf-8")
    import faiss

    index = faiss.IndexFlatIP(4)
    index.add(np.ones((1, 4), dtype=np.float32))
    faiss.write_index(index, str(settings.faiss_index_path))

    store = VectorStore(settings).load()
    assert store.total_chunks == 1
    assert store.documents[0]["filename"] == "legacy.txt"
    assert store.documents[0]["chunks"] == 1
    # doc_id backfilled for older metadata
    assert store.chunks[0]["doc_id"] == "legacy"


def test_load_reports_corruption_instead_of_crashing(settings, embedder):
    settings.ensure_directories()
    settings.metadata_path.write_text("{not json", encoding="utf-8")
    settings.faiss_index_path.write_bytes(b"garbage")
    store = VectorStore(settings).load()
    assert store.is_empty
    assert store.load_error
