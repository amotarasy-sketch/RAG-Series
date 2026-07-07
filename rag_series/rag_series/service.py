from __future__ import annotations

from threading import Lock
from time import perf_counter
from typing import Any

from .cli import Runtime, load_runtime
from .config import CFG
from .logging_utils import logger
from .models import DEVICE
from .exceptions import RuntimeLoadError
from .pipeline import answer_query
from .retrieval import hybrid_search
from .text_store import load_needed_chunk_texts


class RagService:
    """Reusable application service for CLI, API and benchmarks.

    Runtime loading is explicit and thread-safe. In production the API can load
    indexes during FastAPI lifespan startup via API_LOAD_ON_STARTUP=true, while
    local development can keep the lazy behavior.
    """

    def __init__(self) -> None:
        self._runtime: Runtime | None = None
        self._lock = Lock()
        self._last_load_error: str | None = None

    def load(self, rebuild: bool = False) -> Runtime:
        if self._runtime is None or rebuild:
            with self._lock:
                if self._runtime is None or rebuild:
                    try:
                        self._runtime = load_runtime(rebuild=rebuild)
                        self._last_load_error = None
                    except Exception as exc:
                        self._last_load_error = str(exc)
                        logger.exception("Failed to load RAG runtime")
                        raise RuntimeLoadError(str(exc)) from exc
        return self._runtime

    def require_loaded(self) -> Runtime:
        """Return a loaded runtime, loading it lazily when needed.

        Older builds returned HTTP 503 from /ask and /search unless
        API_LOAD_ON_STARTUP=true or /build had been called manually. For a
        Docker demo/API service the expected behavior is that requests work
        after startup, so we lazily load existing indexes as a safety net. If
        loading fails, RuntimeLoadError preserves a clear 503 with the real
        cause in logs and /health.
        """
        if self._runtime is None:
            return self.load(rebuild=False)
        return self._runtime

    def load_on_startup(self) -> None:
        if CFG.api_load_on_startup:
            self.load(rebuild=False)

    def is_ready(self) -> bool:
        return self._runtime is not None

    def readiness(self) -> dict[str, Any]:
        metadata, faiss_index, lexical_db_path, by_book_chunk = self._runtime or ([], None, "", {})
        return {
            "ready": self.is_ready(),
            "indexes_loaded": self._runtime is not None,
            "chunks_loaded": len(metadata),
            "faiss_vectors": int(getattr(faiss_index, "ntotal", 0) or 0),
            "lexical_index_loaded": bool(lexical_db_path),
            "neighbor_index_loaded": bool(by_book_chunk),
            "startup_load_enabled": CFG.api_load_on_startup,
            "last_load_error": self._last_load_error,
        }

    def ask(self, question: str) -> dict[str, Any]:
        started = perf_counter()
        metadata, faiss_index, lexical_db_path, by_book_chunk = self.require_loaded()
        answer, sources = answer_query(
            question,
            metadata,
            faiss_index,
            lexical_db_path,
            by_book_chunk,
        )
        return {
            "answer": answer,
            "sources": sources,
            "source_count": len(sources),
            "latency_ms": round((perf_counter() - started) * 1000, 2),
        }

    def search(self, query: str, top_k: int = 10, include_text: bool = False) -> dict[str, Any]:
        started = perf_counter()
        metadata, faiss_index, lexical_db_path, _ = self.require_loaded()
        results = hybrid_search(query, metadata, faiss_index, lexical_db_path)[:top_k]

        if include_text:
            texts = load_needed_chunk_texts(results)
            for item in results:
                chunk_id = int(item.get("global_chunk_id", int(item["index"]) + 1))
                item["text"] = texts.get(chunk_id, "")

        return {
            "query": query,
            "results": results,
            "count": len(results),
            "latency_ms": round((perf_counter() - started) * 1000, 2),
        }

    def rebuild(self) -> dict[str, Any]:
        started = perf_counter()
        metadata, faiss_index, lexical_db_path, by_book_chunk = self.load(rebuild=True)
        return {
            "status": "ready",
            "chunks": len(metadata),
            "faiss_vectors": int(faiss_index.ntotal),
            "lexical_db_path": lexical_db_path,
            "neighbor_index_size": len(by_book_chunk),
            "latency_ms": round((perf_counter() - started) * 1000, 2),
        }

    def build_summary(self) -> dict[str, Any]:
        metadata, faiss_index, lexical_db_path, by_book_chunk = self.load(rebuild=False)
        return {
            "status": "ready",
            "chunks": len(metadata),
            "faiss_vectors": int(faiss_index.ntotal),
            "lexical_db_path": lexical_db_path,
            "neighbor_index_size": len(by_book_chunk),
            "latency_ms": 0.0,
        }

    def config(self) -> dict[str, Any]:
        return {
            "books_dir": CFG.books_dir,
            "chunks_dir": CFG.chunks_dir,
            "embedding_model_name": CFG.embedding_model_name,
            "reranker_model_name": CFG.reranker_model_name,
            "device": DEVICE,
            "llm_base_url": CFG.llm_base_url,
            "llm_model": CFG.llm_model,
            "semantic_top_k": CFG.semantic_top_k,
            "bm25_top_k": CFG.bm25_top_k,
            "hybrid_top_k": CFG.hybrid_top_k,
            "rerank_top_k": CFG.rerank_top_k,
            "neighbor_window": CFG.neighbor_window,
            "max_context_chunks": CFG.max_context_chunks,
            "max_context_tokens": CFG.max_context_tokens,
            "enable_fast_mode": CFG.enable_fast_mode,
            "enable_llm_cache": CFG.enable_llm_cache,
            "api_load_on_startup": CFG.api_load_on_startup,
            "api_enable_build_endpoint": CFG.api_enable_build_endpoint,
        }


_service: RagService | None = None
_service_lock = Lock()


def get_service() -> RagService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = RagService()
    return _service
