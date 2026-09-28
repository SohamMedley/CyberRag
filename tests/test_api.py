"""End-to-end HTTP tests against the Flask app (stubbed model + LLM)."""

from __future__ import annotations

import io
import json
import threading

import pytest

from rag.config import Settings
from rag.llm import LLMConfigurationError, LLMUpstreamError
from tests.conftest import StubLLM

TXT = (
    "Attendance policy: students below sixty five percent attendance must submit a "
    "medical certificate to the academic office within five working days."
)


def upload(client, name: str, content, content_type: str = "text/plain"):
    data = {"files": (io.BytesIO(content if isinstance(content, bytes) else content.encode()), name)}
    return client.post("/upload", data=data, content_type="multipart/form-data")


def index_files(client, names, replace: bool = False):
    return client.post("/index", json={"filenames": names, "replace": replace})


@pytest.fixture()
def indexed(client):
    """A library containing one indexed text document."""
    assert upload(client, "policy.txt", TXT).status_code == 200
    response = index_files(client, ["policy.txt"])
    assert response.status_code == 200, response.get_json()
    return "policy.txt"


# ---------------------------------------------------------------------- pages
def test_index_page_has_no_activity_view(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "Library" in body and "Ask" in body
    assert "activityView" not in body
    assert 'data-view="activity"' not in body
    assert "resetDialog" not in body
    assert 'id="deleteDialog"' in body


def test_healthz(client):
    payload = client.get("/healthz").get_json()
    assert payload["status"] == "ok"
    assert payload["chunks"] == 0


def test_unknown_api_route_returns_json(client):
    response = client.get("/definitely-not-a-route")
    assert response.status_code == 404
    assert "error" in response.get_json()


# --------------------------------------------------------------------- status
def test_status_on_empty_library(client):
    payload = client.get("/status").get_json()
    assert payload["status"] == "success"
    assert payload["documents_count"] == 0
    assert payload["total_chunks_indexed"] == 0
    assert payload["indexed_documents"] == []
    assert payload["is_groq_key_set"] is True
    assert payload["llm_model"] == "test/model"
    assert payload["limits"]["top_k_retrieval"] == 3


# --------------------------------------------------------------------- upload
def test_upload_requires_files(client):
    assert client.post("/upload", data={}, content_type="multipart/form-data").status_code == 400


def test_upload_rejects_unsupported_types(client):
    response = upload(client, "notes.docx", "hello")
    assert response.status_code == 400
    payload = response.get_json()
    assert "only .pdf and .txt" in payload["details"][0]


def test_upload_stores_files_in_staging(services, client):
    response = upload(client, "report.txt", "some content")
    assert response.status_code == 200
    assert response.get_json()["saved_files"] == ["report.txt"]
    assert (services.settings.upload_dir / "report.txt").exists()


def test_upload_sanitises_path_traversal(services, client):
    response = client.post(
        "/upload",
        data={"files": (io.BytesIO(b"payload"), "../../etc/passwd.txt")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    saved = response.get_json()["saved_files"]
    assert saved == ["etc_passwd.txt"]
    assert (services.settings.upload_dir / "etc_passwd.txt").exists()
    assert not (services.settings.upload_dir.parent / "etc_passwd.txt").exists()


# ---------------------------------------------------------------------- index
def test_index_without_files(client):
    assert client.post("/index", json={}).status_code == 400


def test_index_missing_file_reports_failure(client):
    response = index_files(client, ["nope.txt"])
    assert response.status_code == 400
    report = response.get_json()["report"]
    assert report[0]["status"] == "failed"
    assert "not found" in report[0]["reason"].lower()


def test_index_unsafe_filename_is_rejected(client):
    response = index_files(client, ["../secrets.txt"])
    assert response.status_code == 400
    assert response.get_json()["report"][0]["status"] == "failed"


def test_index_flow_stores_metadata_and_moves_the_file(services, client):
    assert upload(client, "policy.txt", TXT).status_code == 200
    payload = index_files(client, ["policy.txt"]).get_json()
    assert payload["new_chunks_added"] == 1
    assert payload["report"][0]["status"] == "success"
    assert payload["report"][0]["replaced"] is False

    # staging -> archive
    assert not (services.settings.upload_dir / "policy.txt").exists()
    assert (services.settings.documents_dir / "policy.txt").exists()

    status = client.get("/status").get_json()
    assert status["documents_count"] == 1
    assert status["total_chunks_indexed"] == 1
    document = status["indexed_documents"][0]
    assert document["filename"] == "policy.txt"
    assert document["chunks"] == 1
    assert document["pages"] == 1
    assert document["size_bytes"] == len(TXT)
    assert document["indexed_at"]


def test_index_defaults_to_every_staged_file(client):
    upload(client, "one.txt", "first document about attendance")
    upload(client, "two.txt", "second document about hpc jobs")
    payload = client.post("/index", json={}).get_json()
    assert payload["new_chunks_added"] == 2
    assert {row["filename"] for row in payload["report"]} == {"one.txt", "two.txt"}


def test_index_skips_documents_already_in_the_library(client, indexed):
    upload(client, "policy.txt", TXT)  # stage the same name again
    response = index_files(client, ["policy.txt"])
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["new_chunks_added"] == 0
    assert payload["report"][0]["status"] == "skipped"
    assert client.get("/status").get_json()["total_chunks_indexed"] == 1


def test_index_replace_reindexes_without_duplicates(client, indexed):
    upload(client, "policy.txt", TXT + " Updated line about probation.")
    response = index_files(client, ["policy.txt"], replace=True)
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["report"][0]["replaced"] is True
    status = client.get("/status").get_json()
    assert status["documents_count"] == 1
    assert status["total_chunks_indexed"] == 1


def test_index_rejects_documents_without_text(client):
    upload(client, "empty.txt", "    ")
    response = index_files(client, ["empty.txt"])
    assert response.status_code == 400
    assert "empty or image-only" in response.get_json()["report"][0]["reason"]


def test_index_handles_pdf_documents(client, pdf_bytes):
    assert upload(client, "policy.pdf", pdf_bytes, "application/pdf").status_code == 200
    payload = index_files(client, ["policy.pdf"]).get_json()
    assert payload["new_chunks_added"] == 2
    assert payload["report"][0]["pages"] == 2


# ------------------------------------------------------------------------ ask
def test_ask_requires_documents(client):
    response = client.post("/ask", json={"question": "anything?"})
    assert response.status_code == 400
    assert "library is empty" in response.get_json()["error"]


@pytest.mark.parametrize(
    "payload",
    [{}, {"question": ""}, {"question": "   "}, {"question": "x" * 6001}],
)
def test_ask_validates_the_question(client, indexed, payload):
    assert client.post("/ask", json=payload).status_code == 400


def test_ask_returns_answer_sources_and_passages(client, indexed, services):
    response = client.post("/ask", json={"question": "What happens below 65% attendance?"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["grounded"] is True
    assert payload["answer"].startswith("Answer grounded")
    assert payload["sources"][0]["doc_name"] == "policy.txt"
    assert payload["sources"][0]["page_number"] is None
    assert payload["retrieved_context"][0]["doc_name"] == "policy.txt"
    assert payload["model"] == "test/model"
    assert payload["usage"]["total_tokens"] == 15
    # the stub records what the model actually received
    assert services.llm.calls[0]["question"] == "What happens below 65% attendance?"


def test_ask_passes_history_to_the_model(client, indexed, services):
    history = [{"role": "user", "content": "previous question"}, {"role": "assistant", "content": "previous answer"}]
    response = client.post(
        "/ask",
        json={"question": "attendance policy medical certificate", "history": history},
    )
    assert response.status_code == 200
    assert services.llm.calls[-1]["history"] == history


def test_ask_reports_insufficient_context(client, indexed, services):
    services.settings.min_similarity_score = 0.99  # nothing can pass this floor
    response = client.post("/ask", json={"question": "unrelated question"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["grounded"] is False
    assert payload["sources"] == []
    assert payload["retrieved_context"]


def test_ask_maps_configuration_errors_to_400(client, indexed, services):
    services.llm = type(services.llm)(error=LLMConfigurationError("GROQ_API_KEY is not configured."))
    response = client.post("/ask", json={"question": "policy?"})
    assert response.status_code == 400
    assert "GROQ_API_KEY" in response.get_json()["error"]


def test_ask_maps_upstream_errors_to_502(client, indexed, services):
    services.llm = type(services.llm)(error=LLMUpstreamError("Groq is unavailable."))
    response = client.post("/ask", json={"question": "policy?"})
    assert response.status_code == 502
    assert "unavailable" in response.get_json()["error"]


# --------------------------------------------------------------------- delete
def test_delete_document_removes_chunks_and_files(services, client, indexed):
    response = client.delete("/documents/policy.txt")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["deleted_document"] == "policy.txt"
    assert payload["chunks_removed"] == 1
    assert payload["total_chunks_in_index"] == 0
    assert payload["total_documents_indexed"] == 0
    assert payload["deleted_files"]

    assert not (services.settings.documents_dir / "policy.txt").exists()
    status = client.get("/status").get_json()
    assert status["documents_count"] == 0
    assert status["total_chunks_indexed"] == 0


def test_delete_keeps_other_documents_searchable(client, indexed):
    upload(client, "hpc.txt", "HPC laboratory rules limit training jobs to one hour per day.")
    index_files(client, ["hpc.txt"])
    assert client.get("/status").get_json()["documents_count"] == 2

    assert client.delete("/documents/policy.txt").status_code == 200
    status = client.get("/status").get_json()
    assert [doc["filename"] for doc in status["indexed_documents"]] == ["hpc.txt"]
    assert status["total_chunks_indexed"] == 1

    answer = client.post("/ask", json={"question": "How long can training jobs run?"}).get_json()
    assert answer["sources"][0]["doc_name"] == "hpc.txt"


def test_delete_unknown_document_is_404(client):
    assert client.delete("/documents/missing.txt").status_code == 404


def test_delete_rejects_invalid_names(client, indexed):
    assert client.delete("/documents/..%2F..%2Fetc%2Fpasswd.txt").status_code == 400
    assert client.post("/documents/delete", json={}).status_code == 400


def test_delete_post_alias_works(client, indexed):
    response = client.post("/documents/delete", json={"filename": "policy.txt"})
    assert response.status_code == 200
    assert response.get_json()["deleted_document"] == "policy.txt"


def test_delete_survives_a_restart(settings, client, indexed, embedder):
    from app import create_app

    assert client.delete("/documents/policy.txt").status_code == 200

    restarted = create_app(settings)
    restarted.extensions["cyberrag"].embedder = embedder
    payload = restarted.test_client().get("/status").get_json()
    assert payload["total_chunks_indexed"] == 0
    assert payload["indexed_documents"] == []


# ---------------------------------------------------------------------- reset
def test_reset_clears_index_and_staging(services, client, indexed):
    upload(client, "pending.txt", "not indexed yet")
    response = client.post("/reset")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["chunks_removed"] == 1
    assert payload["documents_removed"] == 1
    assert payload["files_deleted"] >= 2

    status = client.get("/status").get_json()
    assert status["total_chunks_indexed"] == 0
    assert not list(services.settings.upload_dir.glob("*.txt"))
    assert not (services.settings.documents_dir / "policy.txt").exists()
    assert not services.settings.faiss_index_path.exists()


# ------------------------------------------------------------- limits, restarts
def test_oversized_request_returns_413(tmp_path):
    from app import create_app

    small = Settings(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        upload_dir=tmp_path / "uploads",
        max_content_length=1024,
        groq_api_key="",
        auto_resolve_groq_model=False,
    )
    application = create_app(small)
    application.config.update(TESTING=True)
    response = application.test_client().post(
        "/upload",
        data={"files": (io.BytesIO(b"x" * 5000), "big.txt")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 413
    assert "limit" in response.get_json()["error"]


def test_index_and_delete_survive_a_restart(settings, client, indexed, embedder, pdf_bytes):
    """Everything needed to answer questions must live on disk, not in memory."""
    from app import create_app

    upload(client, "policy.pdf", pdf_bytes, "application/pdf")
    index_files(client, ["policy.pdf"])

    restarted = create_app(settings)
    restarted.extensions["cyberrag"].embedder = embedder
    restarted.extensions["cyberrag"].llm = StubLLM()
    restarted_client = restarted.test_client()

    status = restarted_client.get("/status").get_json()
    assert status["documents_count"] == 2
    assert status["total_chunks_indexed"] == 3

    answer = restarted_client.post("/ask", json={"question": "attendance policy"}).get_json()
    assert answer["retrieved_context"]

    assert restarted_client.delete("/documents/policy.pdf").status_code == 200
    assert restarted_client.get("/status").get_json()["documents_count"] == 1


def test_status_reports_a_corrupt_index(settings):
    """A damaged index file must be reported, not crash the app."""
    from app import create_app

    settings.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    settings.metadata_path.write_text("{broken", encoding="utf-8")
    settings.faiss_index_path.write_bytes(b"nonsense")

    application = create_app(settings)
    application.config.update(TESTING=True)
    payload = application.test_client().get("/status").get_json()
    assert payload["index_load_error"]
    assert payload["documents_count"] == 0
    assert payload["total_chunks_indexed"] == 0


def test_concurrent_index_requests_stay_consistent(services, app):
    """The mutation lock keeps chunk metadata and vectors in sync."""
    for index in range(2):
        file_path = services.settings.upload_dir / f"doc{index}.txt"
        file_path.write_text(f"document {index} " + "content " * 40, encoding="utf-8")

    errors = []

    def worker(name: str) -> None:
        try:
            response = app.test_client().post("/index", json={"filenames": [name]})
            if response.status_code != 200:
                errors.append((name, response.get_json()))
        except Exception as exc:  # pragma: no cover - reported through `errors`
            errors.append((name, str(exc)))

    threads = [threading.Thread(target=worker, args=(f"doc{index}.txt",)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    status = app.test_client().get("/status").get_json()
    assert status["documents_count"] == 2
    assert status["total_chunks_indexed"] == status["total_vectors_indexed"] == 2

    # the persisted metadata is valid JSON after concurrent writes
    json.loads(services.settings.metadata_path.read_text(encoding="utf-8"))
    json.loads(services.settings.document_summary_path.read_text(encoding="utf-8"))


def test_delete_removes_an_upload_that_was_never_indexed(client, services):
    """A file that was uploaded but never indexed must still be removable."""
    services.settings.upload_dir.mkdir(parents=True, exist_ok=True)
    staged = services.settings.upload_dir / "orphan__rag_0000000000000001.txt"
    staged.write_text("staged but never indexed", encoding="utf-8")

    response = client.delete("/documents/orphan__rag_0000000000000001.txt")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["chunks_removed"] == 0
    assert payload["total_documents_indexed"] == len(services.store.documents)
    assert not staged.exists()


def test_delete_rejects_non_document_extensions(client):
    response = client.delete("/documents/notes.md")
    assert response.status_code == 400
    assert "Only .pdf and .txt" in response.get_json()["error"]


def test_index_replace_reindexes_from_the_archived_copy(client, indexed, services):
    """After the first index the staged file is archived; replace must find it there."""
    staged = services.settings.upload_dir / "policy.txt"
    archived = services.settings.documents_dir / "policy.txt"
    assert not staged.exists() and archived.exists()

    payload = index_files(client, ["policy.txt"], replace=True).get_json()

    assert payload["new_chunks_added"] == 1
    assert payload["total_chunks_in_index"] == 1  # no duplicates
    assert payload["report"][0]["replaced"] is True
    assert (services.settings.upload_dir / "policy.txt").exists() is False


def test_index_without_staged_files_explains_why(client, indexed):
    response = client.post("/index", json={})

    assert response.status_code == 400
    assert "waiting to be indexed" in response.get_json()["error"]


def test_index_skips_are_reported_without_duplicating_chunks(client, indexed):
    payload = index_files(client, ["policy.txt"]).get_json()

    assert payload["new_chunks_added"] == 0
    assert payload["total_chunks_in_index"] == 1
    assert payload["report"][0]["status"] == "skipped"


def test_question_length_cap_is_6000_characters(client, indexed):
    accepted = client.post("/ask", json={"question": "a" * 6000})
    too_long = client.post("/ask", json={"question": "a" * 6001})

    assert accepted.status_code == 200
    assert too_long.status_code == 400
    assert "6000 characters max" in too_long.get_json()["error"]
