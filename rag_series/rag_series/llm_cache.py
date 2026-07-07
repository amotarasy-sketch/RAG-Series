import hashlib
import json
import sqlite3
import time
from typing import Any, Dict, List, Optional

from .config import CFG, LLM_CACHE_PATH


def init_llm_cache() -> None:
    conn = sqlite3.connect(LLM_CACHE_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_cache (
                cache_key TEXT PRIMARY KEY,
                response TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def cache_key_for_llm(query: str, context_results: List[Dict[str, Any]]) -> str:
    payload = {
        "version": CFG.llm_cache_version,
        "model": CFG.llm_model,
        "temperature": 0.15,
        "query": query,
        "sources": [
            {
                "global_chunk_id": int(r["global_chunk_id"]),
                "text_offset": int(r["text_offset"]),
                "book_sha256": r.get("book_sha256"),
                "is_neighbor": bool(r.get("is_neighbor", False)),
            }
            for r in context_results
        ],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def get_llm_cache(cache_key: str) -> Optional[str]:
    if not CFG.enable_llm_cache:
        return None
    conn = sqlite3.connect(LLM_CACHE_PATH)
    try:
        row = conn.execute(
            "SELECT response FROM llm_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def set_llm_cache(cache_key: str, response: str) -> None:
    if not CFG.enable_llm_cache:
        return
    conn = sqlite3.connect(LLM_CACHE_PATH)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO llm_cache(cache_key, response, created_at)
            VALUES (?, ?, ?)
            """,
            (cache_key, response, int(time.time())),
        )
        conn.commit()
    finally:
        conn.close()
