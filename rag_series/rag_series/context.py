from typing import Any, Dict, List, Tuple

from .config import CFG
from .text_store import ChunkTextStore, chunk_text_for_metadata
from .text_utils import count_tokens


def build_index_by_book_chunk(metadata: List[Dict[str, Any]]) -> Dict[Tuple[str, int], int]:
    return {(m["book_file"], int(m["book_chunk_number"])): idx for idx, m in enumerate(metadata)}


def expand_with_neighbors(
    results: List[Dict[str, Any]],
    metadata: List[Dict[str, Any]],
    by_book_chunk: Dict[Tuple[str, int], int],
    neighbor_window: int,
    chunk_texts: ChunkTextStore,
) -> List[Dict[str, Any]]:
    if not results:
        return []

    results_by_index = {r["index"]: r for r in results}
    selected_indexes = set()
    center_order = {r["index"]: pos for pos, r in enumerate(results)}

    for r in results:
        book = r["book_file"]
        n = int(r["book_chunk_number"])
        for delta in range(-neighbor_window, neighbor_window + 1):
            idx = by_book_chunk.get((book, n + delta))
            if idx is not None:
                selected_indexes.add(idx)

    expanded = []
    for idx in selected_indexes:
        item = dict(metadata[idx])
        item["index"] = idx
        item["text"] = chunk_text_for_metadata(item, chunk_texts)

        original = results_by_index.get(idx)
        if original:
            for key in [
                "semantic_score", "semantic_rank", "semantic_passed_threshold",
                "bm25_score", "bm25_rank", "rrf_semantic", "rrf_bm25",
                "hybrid_score", "rerank_score", "retrieval_source", "fast_mode",
            ]:
                if key in original:
                    item[key] = original[key]
            item["is_neighbor"] = False
        else:
            item["is_neighbor"] = True
            item["retrieval_source"] = "neighbor"

        nearest_center_rank = 10**9
        for center_idx, rank in center_order.items():
            c = metadata[center_idx]
            if c["book_file"] == item["book_file"]:
                dist = abs(int(c["book_chunk_number"]) - int(item["book_chunk_number"]))
                if dist <= neighbor_window:
                    nearest_center_rank = min(nearest_center_rank, rank)

        item["_sort_rank"] = nearest_center_rank
        expanded.append(item)

    expanded.sort(key=lambda x: (x["_sort_rank"], x["book_order"], x["book_chunk_number"]))

    final = []
    total_tokens = 0
    for item in expanded:
        if len(final) >= CFG.max_context_chunks:
            break
        t = int(item.get("token_count") or count_tokens(item["text"]))
        if final and total_tokens + t > CFG.max_context_tokens:
            break
        final.append(item)
        total_tokens += t

    for item in final:
        item.pop("_sort_rank", None)

    return final
