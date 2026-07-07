import os
from dataclasses import dataclass
from pathlib import Path

from .env import env_bool, env_float, env_int, env_str, load_environment

load_environment()

# Repository root. This makes the default data folders portable across machines.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    # Folders
    # By default, read books and write generated indexes inside the project.
    # These paths can still be overridden with BOOKS_DIR and CHUNKS_DIR.
    books_dir: str = env_str("BOOKS_DIR", str(PROJECT_ROOT / "Books"))
    chunks_dir: str = env_str("CHUNKS_DIR", str(PROJECT_ROOT / "Chunks"))

    # LM Studio / OpenAI-compatible server
    llm_base_url: str = env_str("LLM_BASE_URL", "http://127.0.0.1:1234/v1")
    llm_api_key: str = env_str("LLM_API_KEY", "lm-studio")
    llm_model: str = env_str("LLM_MODEL", "local-model")

    # Models
    embedding_model_name: str = env_str("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
    reranker_model_name: str = env_str("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")

    # Device selection for local embedding/reranker models.
    # auto = cuda when available, otherwise cpu. You can force cpu/cuda with DEVICE=cpu/cuda.
    device: str = env_str("DEVICE", "auto").lower()

    # Token-based chunking
    chunk_tokens: int = env_int("CHUNK_TOKENS", 420)
    overlap_tokens: int = env_int("OVERLAP_TOKENS", 80)
    min_chunk_tokens: int = env_int("MIN_CHUNK_TOKENS", 80)

    # Retrieval
    semantic_top_k: int = env_int("SEMANTIC_TOP_K", 80)
    bm25_top_k: int = env_int("BM25_TOP_K", 60)
    hybrid_top_k: int = env_int("HYBRID_TOP_K", 50)
    rerank_top_k: int = env_int("RERANK_TOP_K", 6)
    reranker_batch_size: int = env_int("RERANKER_BATCH_SIZE", 16)
    reranker_max_length: int = env_int("RERANKER_MAX_LENGTH", 1024)

    # RRF fusion
    rrf_k: int = env_int("RRF_K", 60)
    semantic_rrf_weight: float = env_float("SEMANTIC_RRF_WEIGHT", 1.0)
    bm25_rrf_weight: float = env_float("BM25_RRF_WEIGHT", 1.0)
    semantic_score_threshold: float = env_float("SEMANTIC_SCORE_THRESHOLD", 0.31)

    # FAISS HNSW
    hnsw_m: int = env_int("HNSW_M", 32)
    hnsw_ef_construction: int = env_int("HNSW_EF_CONSTRUCTION", 200)
    hnsw_ef_search: int = env_int("HNSW_EF_SEARCH", 128)

    allow_bm25_only_without_semantic: bool = env_bool("ALLOW_BM25_ONLY_WITHOUT_SEMANTIC", True)

    # Context
    neighbor_window: int = env_int("NEIGHBOR_WINDOW", 1)
    max_context_chunks: int = env_int("MAX_CONTEXT_CHUNKS", 14)
    max_context_tokens: int = env_int("MAX_CONTEXT_TOKENS", 6500)

    # Embedding batch size
    embedding_batch_size: int = env_int("EMBEDDING_BATCH_SIZE", 16)

    # Files
    embeddings_file: str = env_str("EMBEDDINGS_FILE", "series_embeddings.npy")
    metadata_file: str = env_str("METADATA_FILE", "series_chunks_metadata.json")
    chunks_text_file: str = env_str("CHUNKS_TEXT_FILE", "chunks_text.jsonl")
    lexical_index_file: str = env_str("LEXICAL_INDEX_FILE", "series_fts5.sqlite3")
    faiss_index_file: str = env_str("FAISS_INDEX_FILE", "series_faiss_hnsw.index")
    manifest_file: str = env_str("MANIFEST_FILE", "series_manifest.json")
    llm_cache_file: str = env_str("LLM_CACHE_FILE", "llm_response_cache.sqlite3")

    rebuild_database: bool = env_bool("REBUILD_DATABASE", False)
    load_embeddings_for_debug: bool = env_bool("LOAD_EMBEDDINGS_FOR_DEBUG", False)
    write_debug_chunks: bool = env_bool("WRITE_DEBUG_CHUNKS", False)
    debug_print_fragments: bool = env_bool("DEBUG_PRINT_FRAGMENTS", False)

    # FB2 parsing
    include_annotation: bool = env_bool("INCLUDE_ANNOTATION", False)
    include_epigraphs: bool = env_bool("INCLUDE_EPIGRAPHS", True)
    include_poems: bool = env_bool("INCLUDE_POEMS", True)
    skip_notes_body: bool = env_bool("SKIP_NOTES_BODY", True)

    # Fast mode
    enable_fast_mode: bool = env_bool("ENABLE_FAST_MODE", True)
    fast_semantic_threshold: float = env_float("FAST_SEMANTIC_THRESHOLD", 0.55)
    fast_bm25_threshold: float = env_float("FAST_BM25_THRESHOLD", 12.0)
    fast_margin: float = env_float("FAST_MARGIN", 0.08)
    fast_min_candidates: int = env_int("FAST_MIN_CANDIDATES", 2)

    # LLM cache
    enable_llm_cache: bool = env_bool("ENABLE_LLM_CACHE", True)
    llm_cache_version: str = env_str("LLM_CACHE_VERSION", "rag_llm_cache_v1")

    # API runtime behavior
    api_load_on_startup: bool = env_bool("API_LOAD_ON_STARTUP", True)
    api_enable_build_endpoint: bool = env_bool("API_ENABLE_BUILD_ENDPOINT", True)

    # torch.compile for reranker
    compile_reranker: bool = env_bool("COMPILE_RERANKER", True)


CFG = Config()
os.makedirs(CFG.chunks_dir, exist_ok=True)

EMB_PATH = os.path.join(CFG.chunks_dir, CFG.embeddings_file)
META_PATH = os.path.join(CFG.chunks_dir, CFG.metadata_file)
CHUNKS_TEXT_PATH = os.path.join(CFG.chunks_dir, CFG.chunks_text_file)
LEXICAL_INDEX_PATH = os.path.join(CFG.chunks_dir, CFG.lexical_index_file)
FAISS_INDEX_PATH = os.path.join(CFG.chunks_dir, CFG.faiss_index_file)
MANIFEST_PATH = os.path.join(CFG.chunks_dir, CFG.manifest_file)
LLM_CACHE_PATH = os.path.join(CFG.chunks_dir, CFG.llm_cache_file)
