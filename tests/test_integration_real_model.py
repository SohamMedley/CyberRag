"""Integration test with the real sentence-transformer model.

Skipped automatically when the model weights are not available (offline CI,
blocked network, no cache). Run it explicitly with::

    pytest -m slow tests/test_integration_real_model.py -v
"""

from __future__ import annotations

import pytest

from rag.config import Settings
from rag.documents import build_chunks, extract_text_from_file
from rag.embedder import Embedder
from rag.store import VectorStore

pytestmark = pytest.mark.slow

ATTENDANCE = (
    "Attendance policy: a student whose attendance falls below sixty five percent "
    "must submit a medical certificate to the academic office within five working days."
)
HPC = "HPC laboratory rules: training jobs are limited to one hour per user per day."


@pytest.fixture(scope="module")
def real_embedder(tmp_path_factory):
    settings = Settings(embedding_model_name="sentence-transformers/all-MiniLM-L6-v2")
    embedder = Embedder(settings)
    try:
        embedder.load()
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"sentence-transformers model unavailable: {exc}")
    return embedder


def test_embeddings_are_normalised_and_deterministic(real_embedder):
    vectors = real_embedder.encode(["attendance policy", "attendance policy", "hpc jobs"])
    assert vectors.shape == (3, 384)
    assert vectors.dtype.name == "float32"
    assert pytest.approx(1.0, abs=1e-4) == float((vectors[0] ** 2).sum() ** 0.5)
    assert (vectors[0] == vectors[1]).all()
    # Related texts are closer than unrelated ones.
    assert float(vectors[0] @ vectors[1]) > float(vectors[0] @ vectors[2])


def test_real_retrieval_and_delete(tmp_path, real_embedder):
    settings = Settings(
        data_dir=tmp_path / "data",
        upload_dir=tmp_path / "uploads",
        chunk_size_words=120,
        chunk_overlap_words=20,
        min_similarity_score=0.1,
    )
    settings.ensure_directories()
    store = VectorStore(settings).load()

    (settings.upload_dir / "policy.txt").write_text(ATTENDANCE, encoding="utf-8")
    (settings.upload_dir / "hpc.txt").write_text(HPC, encoding="utf-8")

    for name in ("policy.txt", "hpc.txt"):
        pages = extract_text_from_file(settings.upload_dir / name, name)
        chunks = build_chunks(pages, name, settings.chunk_size_words, settings.chunk_overlap_words)
        vectors = real_embedder.encode([chunk["text"] for chunk in chunks])
        store.add_document(name, chunks, vectors, pages=len(pages))

    hits = store.search(real_embedder.encode_one("How much attendance do I need?"), k=2)
    assert hits and hits[0]["doc_name"] == "policy.txt"
    assert hits[0]["similarity_score"] > 0.3

    store.delete_document("policy.txt", reembed=real_embedder.encode)
    hits = store.search(real_embedder.encode_one("How much attendance do I need?"), k=2)
    assert all(hit["doc_name"] != "policy.txt" for hit in hits)
    assert store.total_vectors == store.total_chunks
