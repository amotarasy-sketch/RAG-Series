from __future__ import annotations

from rag_series import indexes


def test_build_fts_query_quotes_tokens_and_escapes_quotes() -> None:
    assert indexes.build_fts_query(["кот", 'say"hi', "", "  мир  "]) == '"кот" OR "say""hi" OR "мир"'


def test_build_fts_query_returns_empty_string_for_empty_tokens() -> None:
    assert indexes.build_fts_query(["", "   "]) == ""


def test_lexical_search_fts5_returns_ranked_chunk_indexes(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "lexical.sqlite3"
    monkeypatch.setattr(indexes, "tokenize_bm25_uncached", lambda text: text.lower().split())
    monkeypatch.setattr(indexes, "tokenize_bm25_query", lambda text: text.lower().split())

    indexes.build_lexical_index_from_chunks(
        ["red apple", "green apple", "blue berry"],
        db_path=str(db_path),
    )

    results = indexes.lexical_search_fts5("apple", db_path=str(db_path), top_k=10)

    assert [item["index"] for item in results] == [0, 1]
    assert [item["bm25_rank"] for item in results] == [1, 2]
    assert all(item["bm25_score"] > 0 for item in results)
