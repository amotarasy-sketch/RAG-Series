from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
import shutil
import sys
from threading import Lock
from typing import Any, Callable, Iterable, Tuple

import torch
import pymorphy3
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import snapshot_download

from .config import CFG
from .logging_utils import logger, timed



def _download_fresh_snapshot(model_name: str) -> str:
    """Download a fresh Hugging Face snapshot after detecting corrupt JSON cache files.

    Docker keeps model files in the named ``rag-series-models`` volume. If a
    previous download was interrupted, Transformers/SentenceTransformers may see
    an empty cached JSON file and fail with ``JSONDecodeError``. A forced
    snapshot refresh replaces the broken files while keeping the public API
    unchanged for the rest of the app.
    """
    logger.warning("Refreshing Hugging Face cache for %s after invalid cached JSON was detected.", model_name)
    return snapshot_download(repo_id=model_name, force_download=True)


def _load_tokenizer(model_name: str) -> Any:
    """Load tokenizer and repair stale/corrupt HF cache once if needed."""
    try:
        return AutoTokenizer.from_pretrained(model_name)
    except json.JSONDecodeError as exc:
        logger.warning("Tokenizer cache for %s is invalid: %s", model_name, exc)
        snapshot_path = _download_fresh_snapshot(model_name)
        return AutoTokenizer.from_pretrained(snapshot_path)


def _load_sequence_classifier(model_name: str, **kwargs: Any) -> Any:
    """Load sequence-classification model and repair corrupt HF cache once."""
    try:
        return AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
    except json.JSONDecodeError as exc:
        logger.warning("Model cache for %s is invalid: %s", model_name, exc)
        snapshot_path = _download_fresh_snapshot(model_name)
        return AutoModelForSequenceClassification.from_pretrained(snapshot_path, **kwargs)

def _select_device() -> str:
    requested = CFG.device
    cuda_available = torch.cuda.is_available()

    if requested in {"cpu", "cuda"}:
        if requested == "cuda" and not cuda_available:
            logger.warning(
                "DEVICE=cuda was requested, but CUDA is not available inside this process; falling back to CPU. "
                "For Docker, make sure NVIDIA Container Toolkit is installed and docker compose passes gpus: all."
            )
            return "cpu"
        return requested

    if requested != "auto":
        logger.warning("Unknown DEVICE=%r; using auto device selection.", requested)

    return "cuda" if cuda_available else "cpu"




def _has_cxx_compiler() -> bool:
    """Return True when a C/C++ compiler is available for torch.compile.

    PyTorch Inductor may defer native-code compilation until the first model
    invocation. Checking before torch.compile prevents runtime failures such as
    "RuntimeError: Failed to find C compiler" in slim Docker images.
    """
    return any(shutil.which(name) for name in ("c++", "g++", "clang++", "gcc", "clang"))


def _should_compile_reranker() -> bool:
    """Decide whether torch.compile can be enabled safely for the reranker."""
    if DEVICE != "cuda":
        return False
    if not CFG.compile_reranker:
        return False
    if not hasattr(torch, "compile"):
        logger.warning("COMPILE_RERANKER=true, but this PyTorch build has no torch.compile; using eager reranker.")
        return False
    if not _has_cxx_compiler():
        logger.warning(
            "COMPILE_RERANKER=true, but no C/C++ compiler was found in PATH; "
            "using eager reranker. Install gcc/g++ or build-essential in the Docker image "
            "to enable torch.compile."
        )
        return False
    return True




def _load_sentence_transformer() -> SentenceTransformer:
    """Load the embedding model with a workaround for text-only models.

    Recent versions of ``transformers`` may try to load an ``AutoProcessor``
    while SentenceTransformer is initializing the underlying Transformer
    module.  For text-only embedding models such as BAAI/bge-m3 a processor is
    optional, but a stale/empty Hugging Face negative-cache marker can make
    ``AutoProcessor.from_pretrained`` raise ``JSONDecodeError`` and abort the
    whole model load.  Falling back to ``None`` for the processor keeps loading
    the tokenizer/model weights normally and avoids repeatedly recreating the
    SentenceTransformer instance.
    """
    try:
        from transformers.models.auto.processing_auto import AutoProcessor
    except ImportError:
        # Unit-test stubs and older transformers layouts may not expose this
        # private module path. In that case there is nothing to patch.
        return SentenceTransformer(CFG.embedding_model_name, device=DEVICE)

    original_from_pretrained = AutoProcessor.from_pretrained

    @classmethod
    def safe_from_pretrained(cls, *args: Any, **kwargs: Any) -> Any:
        try:
            return original_from_pretrained(*args, **kwargs)
        except json.JSONDecodeError as exc:
            logger.warning(
                "AutoProcessor config is invalid or unavailable for %s; continuing without processor: %s",
                CFG.embedding_model_name,
                exc,
            )
            return None

    AutoProcessor.from_pretrained = safe_from_pretrained
    try:
        try:
            return SentenceTransformer(CFG.embedding_model_name, device=DEVICE)
        except json.JSONDecodeError as exc:
            logger.warning("SentenceTransformer cache for %s is invalid: %s", CFG.embedding_model_name, exc)
            snapshot_path = _download_fresh_snapshot(CFG.embedding_model_name)
            return SentenceTransformer(snapshot_path, device=DEVICE)
    finally:
        AutoProcessor.from_pretrained = original_from_pretrained

DEVICE = _select_device()
if DEVICE == "cuda":
    logger.info("Device: cuda (%s)", torch.cuda.get_device_name(0))
else:
    logger.info("Device: cpu")


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    """Return a cached OpenAI-compatible client."""
    return OpenAI(base_url=CFG.llm_base_url, api_key=CFG.llm_api_key)


_embedding_model: SentenceTransformer | None = None
_embedding_model_lock = Lock()


def get_embedding_model() -> SentenceTransformer:
    """Return a process-wide singleton embedding model.

    SentenceTransformer is expensive to initialize and holds model weights in
    RAM/VRAM, so every caller must reuse the same instance.  A plain
    @lru_cache can still execute the wrapped function more than once when
    concurrent requests hit an empty cache at the same time; the explicit lock
    makes the first initialization strictly single-flight.
    """
    global _embedding_model
    if _embedding_model is None:
        with _embedding_model_lock:
            if _embedding_model is None:
                with timed("Load embedding model"):
                    _embedding_model = _load_sentence_transformer()
    return _embedding_model


def clear_embedding_model_cache() -> None:
    """Clear the embedding singleton, primarily for tests and controlled reloads."""
    global _embedding_model
    with _embedding_model_lock:
        _embedding_model = None
    get_chunk_tokenizer.cache_clear()

    retrieval_module = sys.modules.get("rag_series.retrieval")
    if retrieval_module is not None:
        encode_query_cached_bytes = getattr(retrieval_module, "encode_query_cached_bytes", None)
        if encode_query_cached_bytes is not None:
            encode_query_cached_bytes.cache_clear()


@lru_cache(maxsize=1)
def get_chunk_tokenizer() -> Any:
    """Return a tokenizer for chunk token counting without loading model weights.

    Chunking only needs token ids, not the full SentenceTransformer. Loading the
    full embedding model here makes /build much more fragile and can duplicate
    GPU work before embeddings are actually encoded.
    """
    return _load_tokenizer(CFG.embedding_model_name)


@lru_cache(maxsize=1)
def get_reranker() -> Tuple[Any, Any]:
    """Load and return the reranker tokenizer and model lazily."""
    with timed("Load reranker"):
        tokenizer = _load_tokenizer(CFG.reranker_model_name)
        reranker_dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        model = _load_sequence_classifier(
            CFG.reranker_model_name,
            torch_dtype=reranker_dtype,
        )

    model.to(DEVICE)
    model.eval()

    if _should_compile_reranker():
        try:
            logger.info("Compiling reranker with torch.compile...")
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as exc:  # pragma: no cover
            logger.warning("torch.compile failed before first inference, using eager reranker: %s", exc)

    logger.info("Reranker dtype: %s", reranker_dtype)
    return tokenizer, model


@lru_cache(maxsize=1)
def get_morph() -> pymorphy3.MorphAnalyzer:
    """Load the Russian lemmatizer lazily."""
    logger.info("Loading Russian lemmatizer...")
    return pymorphy3.MorphAnalyzer()
