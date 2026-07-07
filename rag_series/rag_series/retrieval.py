from functools import lru_cache
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from .config import CFG
from .indexes import lexical_search_fts5
from .logging_utils import logger, timed
from .models import get_embedding_model

QUERY_EMBEDDING_CACHE_SIZE = 1024


@lru_cache(maxsize=QUERY_EMBEDDING_CACHE_SIZE)
def encode_query_cached_bytes(query: str) -> bytes:
    emb = get_embedding_model().encode(query, normalize_embeddings=True)
    return np.asarray(emb, dtype=np.float32).tobytes()


def encode_query_cached(query: str) -> np.ndarray:
    raw = encode_query_cached_bytes(query)
    emb = np.frombuffer(raw, dtype=np.float32)
    return emb.reshape(1, -1)


def semantic_search_faiss(query: str, faiss_index: faiss.Index, top_k: int) -> List[Dict[str, Any]]:
    if hasattr(faiss_index, "hnsw"):
        faiss_index.hnsw.efSearch = CFG.hnsw_ef_search

    query_embedding = encode_query_cached(query)
    top_k = min(top_k, faiss_index.ntotal)
    scores, indexes = faiss_index.search(query_embedding, top_k)
    scores = scores[0]
    indexes = indexes[0]

    valid = indexes >= 0
    indexes = indexes[valid]
    scores = scores[valid]

    return [
        {
            "index": int(idx),
            "semantic_score": float(scores[pos]),
            "semantic_rank": pos + 1,
            "semantic_passed_threshold": bool(float(scores[pos]) >= CFG.semantic_score_threshold),
        }
        for pos, idx in enumerate(indexes)
    ]


def bm25_search(query: str, lexical_db_path: str, top_k: int) -> List[Dict[str, Any]]:
    return lexical_search_fts5(query, lexical_db_path, top_k)


def rrf_score(rank: Optional[int], k: int) -> float:
    if rank is None:
        return 0.0
    return 1.0 / (k + rank)


def hybrid_search(
    query: str,
    metadata: List[Dict[str, Any]],
    faiss_index: faiss.Index,
    lexical_db_path: str,
) -> List[Dict[str, Any]]:
    with timed("Semantic search / FAISS"):
        sem = semantic_search_faiss(query, faiss_index, CFG.semantic_top_k)
    with timed("Lexical search / FTS5 BM25"):
        lex = bm25_search(query, lexical_db_path, CFG.bm25_top_k)
    logger.info("Semantic results: %d; BM25 results: %d", len(sem), len(lex))

    combined: Dict[int, Dict[str, Any]] = {}
    for r in sem:
        idx = r["index"]
        combined.setdefault(idx, {"index": idx})
        combined[idx].update(r)
    for r in lex:
        idx = r["index"]
        combined.setdefault(idx, {"index": idx})
        combined[idx].update(r)

    results = []
    for idx, r in combined.items():
        semantic_score = r.get("semantic_score")
        has_semantic = semantic_score is not None
        has_bm25 = "bm25_score" in r
        semantic_passed = bool(r.get("semantic_passed_threshold", False))

        if has_semantic and not semantic_passed and not has_bm25:
            continue
        if not has_semantic and has_bm25 and not CFG.allow_bm25_only_without_semantic:
            continue

        semantic_rrf = CFG.semantic_rrf_weight * rrf_score(r.get("semantic_rank"), CFG.rrf_k)
        bm25_rrf = CFG.bm25_rrf_weight * rrf_score(r.get("bm25_rank"), CFG.rrf_k)
        hybrid_score = semantic_rrf + bm25_rrf

        item = dict(metadata[idx])
        item.update({
            "index": idx,
            "semantic_score": semantic_score,
            "semantic_rank": r.get("semantic_rank"),
            "semantic_passed_threshold": semantic_passed if has_semantic else None,
            "bm25_score": r.get("bm25_score"),
            "bm25_rank": r.get("bm25_rank"),
            "rrf_semantic": float(semantic_rrf),
            "rrf_bm25": float(bm25_rrf),
            "hybrid_score": float(hybrid_score),
            "retrieval_source": "+".join([
                name for name, present in [
                    ("faiss_semantic", has_semantic),
                    ("fts5_bm25_lemmas", has_bm25),
                ] if present
            ]),
        })
        results.append(item)

    results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return results[:CFG.hybrid_top_k]
