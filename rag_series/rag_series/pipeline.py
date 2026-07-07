from typing import Any, Dict, List, Tuple

import faiss

from .config import CFG
from .context import expand_with_neighbors
from .llm import ask_llm
from .logging_utils import logger, timed
from .reranking import rerank, should_skip_rerank
from .retrieval import hybrid_search
from .text_store import load_needed_chunk_texts


def answer_query(
    query: str,
    metadata: List[Dict[str, Any]],
    faiss_index: faiss.Index,
    lexical_db_path: str,
    by_book_chunk: Dict[Tuple[str, int], int],
) -> Tuple[str, List[Dict[str, Any]]]:
    with timed("Retrieval / hybrid_search"):
        candidates = hybrid_search(query, metadata, faiss_index, lexical_db_path)
    logger.info("Hybrid candidates: %d", len(candidates))

    with timed("Load candidate chunk texts"):
        candidate_texts = load_needed_chunk_texts(candidates)

    if should_skip_rerank(candidates):
        logger.info("Fast mode: skipping reranker")
        reranked = [dict(r, rerank_score=None, fast_mode=True) for r in candidates[:CFG.rerank_top_k]]
    else:
        with timed("Rerank"):
            reranked = rerank(query, candidates, top_k=CFG.rerank_top_k, chunk_texts=candidate_texts)
    logger.info("Reranked results: %d", len(reranked))

    with timed("Context expansion"):
        needed_indexes = set()
        for r in reranked:
            book = r["book_file"]
            n = int(r["book_chunk_number"])
            for delta in range(-CFG.neighbor_window, CFG.neighbor_window + 1):
                idx = by_book_chunk.get((book, n + delta))
                if idx is not None:
                    needed_indexes.add(idx)

        context_metas = [metadata[idx] for idx in needed_indexes]
        context_texts = load_needed_chunk_texts(context_metas)

        context_results = expand_with_neighbors(
            reranked,
            metadata,
            by_book_chunk=by_book_chunk,
            neighbor_window=CFG.neighbor_window,
            chunk_texts=context_texts,
        )
    logger.info("Context chunks: %d", len(context_results))

    with timed("LLM answer"):
        answer = ask_llm(query, context_results)
    return answer, context_results
