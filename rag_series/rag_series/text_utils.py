import hashlib
import os
import re
from functools import lru_cache
from typing import Dict, List, Tuple

from .models import get_chunk_tokenizer, get_morph

LEMMA_CACHE: Dict[str, str] = {}
BM25_QUERY_CACHE_SIZE = 4096


def safe_filename(name: str) -> str:
    name = os.path.splitext(name)[0]
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize_spaces(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def lemmatize_token(token: str) -> str:
    token = token.lower()
    if len(token) < 3:
        return token
    cached = LEMMA_CACHE.get(token)
    if cached is not None:
        return cached
    if re.fullmatch(r"[a-z0-9]+", token):
        LEMMA_CACHE[token] = token
        return token
    parsed = get_morph().parse(token)
    lemma = parsed[0].normal_form if parsed else token
    LEMMA_CACHE[token] = lemma
    return lemma


def tokenize_bm25_uncached(text: str) -> List[str]:
    tokens = re.findall(r"[а-яА-ЯёЁa-zA-Z0-9]+", text.lower())
    return [lemmatize_token(t) for t in tokens]


@lru_cache(maxsize=BM25_QUERY_CACHE_SIZE)
def tokenize_bm25_query_cached(text: str) -> Tuple[str, ...]:
    return tuple(tokenize_bm25_uncached(text))


def tokenize_bm25_query(text: str) -> List[str]:
    return list(tokenize_bm25_query_cached(text))


@lru_cache(maxsize=200_000)
def count_tokens(text: str) -> int:
    return len(get_chunk_tokenizer().encode(text, add_special_tokens=False))


def count_tokens_batch(texts: List[str]) -> List[int]:
    if not texts:
        return []
    encodings = get_chunk_tokenizer()(
        texts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )
    return [len(ids) for ids in encodings["input_ids"]]
