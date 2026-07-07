from __future__ import annotations

from rag_series.service import RagService


class FakeFaissIndex:
    ntotal = 1


def test_search_include_text_uses_global_chunk_id(monkeypatch) -> None:
    service = RagService()
    service._runtime = ([{"global_chunk_id": 42, "book": "demo.fb2"}], FakeFaissIndex(), "fts.sqlite3", {})

    def fake_hybrid_search(query, metadata, faiss_index, lexical_db_path):
        return [dict(metadata[0], index=0, hybrid_score=1.0)]

    def fake_load_needed_chunk_texts(results):
        return {42: "correct chunk text"}

    monkeypatch.setattr("rag_series.service.hybrid_search", fake_hybrid_search)
    monkeypatch.setattr("rag_series.service.load_needed_chunk_texts", fake_load_needed_chunk_texts)

    payload = service.search("dragon", include_text=True)

    assert payload["results"][0]["text"] == "correct chunk text"
