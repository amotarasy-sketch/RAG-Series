from typing import Any, Dict, List


def print_used_fragments(results: List[Dict[str, Any]]) -> None:
    print("\n=== USED FRAGMENTS ===\n")
    for i, r in enumerate(results, start=1):
        print("=" * 90)
        print(f"Источник #{i}")
        print("Книга:", r.get("book_title"))
        print("Файл:", r.get("book_file"))
        print("Глава:", r.get("chapter"))
        print("Чанк в книге:", r.get("book_chunk_number"))
        print("Text offset:", r.get("text_offset"))
        print("Соседний чанк:", "да" if r.get("is_neighbor") else "нет")
        print("Retrieval source:", r.get("retrieval_source", "neighbor"))
        print("Semantic score:", round(r["semantic_score"], 4) if r.get("semantic_score") is not None else None)
        print("Semantic rank:", r.get("semantic_rank"))
        print("Semantic threshold passed:", r.get("semantic_passed_threshold"))
        print("BM25 score:", round(r["bm25_score"], 4) if r.get("bm25_score") is not None else None)
        print("BM25 rank:", r.get("bm25_rank"))
        print("RRF semantic:", round(r["rrf_semantic"], 6) if r.get("rrf_semantic") is not None else None)
        print("RRF BM25:", round(r["rrf_bm25"], 6) if r.get("rrf_bm25") is not None else None)
        print("Hybrid/RRF score:", round(r["hybrid_score"], 6) if r.get("hybrid_score") is not None else None)
        print("Rerank score:", round(r["rerank_score"], 4) if r.get("rerank_score") is not None else None)
        print("Fast mode:", r.get("fast_mode"))
        print("-" * 90)
        print(r["text"])
        print()
