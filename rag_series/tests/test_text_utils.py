from __future__ import annotations

import hashlib

from rag_series import text_utils


def test_safe_filename_removes_extension_and_replaces_forbidden_chars() -> None:
    assert text_utils.safe_filename('A<B>C:"D"/E\\F|G?H*.fb2') == "A_B_C__D__E_F_G_H_"


def test_normalize_spaces_collapses_spaces_and_blank_lines() -> None:
    assert text_utils.normalize_spaces("  a\t b\xa0c\n   d\n\n\n\n e  ") == "a b c\nd\n\ne"


def test_file_sha256_reads_file_in_binary_chunks(tmp_path) -> None:
    path = tmp_path / "book.fb2"
    path.write_bytes(b"abc" * 1024)

    assert text_utils.file_sha256(str(path)) == hashlib.sha256(b"abc" * 1024).hexdigest()


def test_tokenize_bm25_query_uses_cache_and_lemmatization(monkeypatch) -> None:
    text_utils.tokenize_bm25_query_cached.cache_clear()
    calls: list[str] = []

    def fake_lemma(token: str) -> str:
        calls.append(token)
        return token[::-1]

    monkeypatch.setattr(text_utils, "lemmatize_token", fake_lemma)

    assert text_utils.tokenize_bm25_query("Коты, cats 42!") == ["ыток", "stac", "24"]
    assert text_utils.tokenize_bm25_query("Коты, cats 42!") == ["ыток", "stac", "24"]
    assert calls == ["коты", "cats", "42"]
