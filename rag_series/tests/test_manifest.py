from __future__ import annotations

import json
from types import SimpleNamespace

from rag_series import database
from rag_series.text_utils import file_sha256


def test_current_books_manifest_is_sorted_and_contains_file_fingerprints(tmp_path, monkeypatch) -> None:
    books_dir = tmp_path / "books"
    books_dir.mkdir()
    (books_dir / "b.fb2").write_text("second", encoding="utf-8")
    (books_dir / "a.fb2").write_text("first", encoding="utf-8")
    (books_dir / "ignore.txt").write_text("ignored", encoding="utf-8")

    monkeypatch.setattr(
        database,
        "CFG",
        SimpleNamespace(
            books_dir=str(books_dir),
            chunk_tokens=420,
            overlap_tokens=80,
            min_chunk_tokens=20,
            embedding_model_name="test-embedding",
            hnsw_m=16,
            hnsw_ef_construction=100,
        ),
    )

    manifest = database.current_books_manifest()

    assert [item["file"] for item in manifest["items"]] == ["a.fb2", "b.fb2"]
    assert manifest["items"][0]["sha256"] == file_sha256(str(books_dir / "a.fb2"))
    assert manifest["items"][0]["size"] == len("first".encode("utf-8"))
    assert manifest["embedding_model"] == "test-embedding"
    assert manifest["faiss"]["m"] == 16


def test_load_manifest_returns_none_when_manifest_is_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(database, "MANIFEST_PATH", str(tmp_path / "missing.json"))

    assert database.load_manifest() is None


def test_load_manifest_reads_json(tmp_path, monkeypatch) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"items": [{"file": "a.fb2"}]}), encoding="utf-8")
    monkeypatch.setattr(database, "MANIFEST_PATH", str(manifest_path))

    assert database.load_manifest() == {"items": [{"file": "a.fb2"}]}
