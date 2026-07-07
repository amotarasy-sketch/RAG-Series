import os
import sqlite3
from typing import Any, Dict, List

import faiss
import numpy as np

from .config import CFG, FAISS_INDEX_PATH, LEXICAL_INDEX_PATH
from .text_utils import tokenize_bm25_query, tokenize_bm25_uncached


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, CFG.hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = CFG.hnsw_ef_construction
    index.add(embeddings)
    return index


def save_faiss_index(index: faiss.Index) -> None:
    faiss.write_index(index, FAISS_INDEX_PATH)


def load_faiss_index() -> faiss.Index:
    index = faiss.read_index(FAISS_INDEX_PATH)
    if hasattr(index, "hnsw"):
        index.hnsw.efSearch = CFG.hnsw_ef_search
    return index


def build_fts_query(tokens: List[str]) -> str:
    safe_tokens = []
    for token in tokens:
        token = token.strip()
        if token:
            safe_tokens.append(f'"{token.replace(chr(34), chr(34) * 2)}"')
    return " OR ".join(safe_tokens)


def build_lexical_index_from_chunks(chunks: List[str], db_path: str = LEXICAL_INDEX_PATH) -> None:
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(
            "CREATE VIRTUAL TABLE chunks_fts USING fts5("
            "chunk_id UNINDEXED, "
            "lemmas, "
            "tokenize='unicode61'"
            ")"
        )

        rows = []
        for chunk_id, chunk in enumerate(chunks, start=1):
            rows.append((chunk_id, " ".join(tokenize_bm25_uncached(chunk))))

        conn.executemany(
            "INSERT INTO chunks_fts(chunk_id, lemmas) VALUES (?, ?)",
            rows,
        )
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('optimize')")
        conn.commit()
    finally:
        conn.close()


def lexical_search_fts5(query: str, db_path: str, top_k: int) -> List[Dict[str, Any]]:
    tokens = tokenize_bm25_query(query)
    if not tokens:
        return []
    fts_query = build_fts_query(tokens)
    if not fts_query:
        return []

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT chunk_id, bm25(chunks_fts) AS rank_score
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY rank_score
            LIMIT ?
            """,
            (fts_query, int(top_k)),
        ).fetchall()
    finally:
        conn.close()

    results = []
    for pos, (chunk_id, rank_score) in enumerate(rows, start=1):
        idx = int(chunk_id) - 1
        results.append({
            "index": idx,
            "bm25_score": float(-rank_score),
            "bm25_rank": pos,
        })
    return results
