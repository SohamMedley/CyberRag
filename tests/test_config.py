"""Settings, path handling and filename helpers."""

from __future__ import annotations

import os
from pathlib import Path

from rag.config import Settings, slugify_doc_id


def test_from_env_uses_defaults(monkeypatch):
    for name in (
        "DATA_DIR",
        "UPLOAD_DIR",
        "GROQ_API_KEY",
        "GROQ_MODEL",
        "TOP_K_RETRIEVAL",
        "PORT",
        "FLASK_PORT",
    ):
        monkeypatch.delenv(name, raising=False)

    config = Settings.from_env()
    assert config.data_dir == Path(config.base_dir) / "data"
    assert config.upload_dir == Path(config.base_dir) / "uploads"
    assert config.top_k_retrieval == 4
    assert config.groq_api_key == ""
    assert config.is_allowed_file("notes.TXT")
    assert not config.is_allowed_file("notes.docx")
    assert not config.is_allowed_file("noextension")


def test_from_env_reads_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "staging"))
    monkeypatch.setenv("GROQ_API_KEY", "  gsk_test  ")
    monkeypatch.setenv("TOP_K_RETRIEVAL", "7")
    monkeypatch.setenv("MAX_CONTENT_LENGTH_MB", "4")
    monkeypatch.setenv("CHUNK_SIZE_WORDS", "not-a-number")

    config = Settings.from_env()
    assert config.data_dir == tmp_path / "store"
    assert config.upload_dir == tmp_path / "staging"
    assert config.groq_api_key == "gsk_test"
    assert config.top_k_retrieval == 7
    assert config.max_content_length == 4 * 1024 * 1024
    assert config.chunk_size_words == 350  # invalid value falls back to the default


def test_paths_and_directory_creation(settings: Settings):
    assert settings.index_dir == settings.data_dir / "index"
    assert settings.faiss_index_path.name == "faiss_index.bin"
    settings.ensure_directories()
    for directory in (settings.data_dir, settings.index_dir, settings.documents_dir, settings.upload_dir):
        assert directory.is_dir()


def test_persistent_disk_detection(settings: Settings, tmp_path):
    assert settings.uses_persistent_disk() is True  # data dir lives in tmp_path
    assert Settings(base_dir=tmp_path, data_dir=tmp_path / "data", upload_dir=tmp_path / "uploads").uses_persistent_disk() is False


def test_filename_helpers():
    assert slugify_doc_id("My File (v2).pdf") == "My_File__v2__pdf"
    assert slugify_doc_id("plain.txt") == "plain_txt"  # dots are not id-safe


def test_public_dict_is_json_friendly(settings: Settings):
    payload = settings.public_dict()
    assert payload["top_k_retrieval"] == settings.top_k_retrieval
    assert isinstance(payload["persistent_storage"], bool)
    assert os.path.isabs(payload["data_dir"])


def test_blank_path_env_values_fall_back_to_defaults(monkeypatch):
    """`.env.example` ships `DATA_DIR=` - an empty value must not mean CWD."""
    monkeypatch.setenv("DATA_DIR", "   ")
    monkeypatch.setenv("UPLOAD_DIR", "")
    config = Settings.from_env()
    assert config.data_dir == Path(config.base_dir) / "data"
    assert config.upload_dir == Path(config.base_dir) / "uploads"
