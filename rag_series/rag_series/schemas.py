from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Stable machine-readable error code.")
    message: str = Field(..., description="Human-readable error message.")
    request_id: str | None = Field(None, description="Request correlation id, when available.")


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    ready: bool
    indexes_loaded: bool
    chunks_loaded: int = 0
    faiss_vectors: int = 0
    lexical_index_loaded: bool = False
    neighbor_index_loaded: bool = False
    startup_load_enabled: bool
    last_load_error: str | None = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question for the RAG assistant.")


class ChunkMetadata(BaseModel):
    index: int | None = Field(None, description="Zero-based runtime metadata index.")
    global_chunk_id: int | None = None
    book_file: str | None = None
    book_title: str | None = None
    authors: list[str] = Field(default_factory=list)
    sequence_name: str | None = None
    sequence_number: int | float | None = None
    book_chunk_number: int | None = None
    section_title: str | None = None
    text_offset: int | None = None
    text_length: int | None = None

    model_config = {"extra": "allow"}


class RetrievalResult(ChunkMetadata):
    semantic_score: float | None = None
    semantic_rank: int | None = None
    semantic_passed_threshold: bool | None = None
    bm25_score: float | None = None
    bm25_rank: int | None = None
    rrf_semantic: float | None = None
    rrf_bm25: float | None = None
    hybrid_score: float | None = None
    rerank_score: float | None = None
    retrieval_source: str | None = None
    fast_mode: bool | None = None
    text: str | None = None


class SourceResponse(RetrievalResult):
    pass


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    source_count: int
    latency_ms: float


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query.")
    top_k: int = Field(10, ge=1, le=50, description="Maximum number of results.")
    include_text: bool = Field(False, description="Include chunk text in search results.")


class SearchResponse(BaseModel):
    query: str
    results: list[RetrievalResult]
    count: int
    latency_ms: float


class BuildResponse(BaseModel):
    status: Literal["ready"] = "ready"
    chunks: int
    faiss_vectors: int
    lexical_db_path: str
    neighbor_index_size: int
    latency_ms: float


class ConfigResponse(BaseModel):
    books_dir: str
    chunks_dir: str
    embedding_model_name: str
    reranker_model_name: str
    device: str = "auto"
    llm_base_url: str
    llm_model: str
    semantic_top_k: int
    bm25_top_k: int
    hybrid_top_k: int
    rerank_top_k: int
    neighbor_window: int
    max_context_chunks: int
    max_context_tokens: int
    enable_fast_mode: bool
    enable_llm_cache: bool
    api_load_on_startup: bool
    api_enable_build_endpoint: bool

    model_config = {"extra": "allow"}


class EvalQuestion(BaseModel):
    id: str
    question: str = Field(..., min_length=1)
    relevant_chunk_ids: set[int]


def as_model(model_type: type[BaseModel], data: dict[str, Any]) -> BaseModel:
    return model_type.model_validate(data)
