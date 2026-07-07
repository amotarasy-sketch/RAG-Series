"""Lightweight test doubles for optional heavyweight runtime dependencies."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

os.environ.setdefault("BOOKS_DIR", str(Path("/tmp/rag_series_test_books")))
os.environ.setdefault("CHUNKS_DIR", str(Path("/tmp/rag_series_test_chunks")))
Path(os.environ["BOOKS_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["CHUNKS_DIR"]).mkdir(parents=True, exist_ok=True)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



class DummyTokenizer:
    def __init__(self) -> None:
        self._last_tokens: list[str] = []

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        self._last_tokens = text.split()
        return list(range(len(self._last_tokens)))

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(self._last_tokens[i] for i in ids)

    def __call__(self, texts, add_special_tokens=False, padding=False, truncation=False):
        return {"input_ids": [list(range(len(text.split()))) for text in texts]}


class DummyEmbeddingModel:
    tokenizer = DummyTokenizer()

    def encode(self, *args, **kwargs):  # pragma: no cover - not used in unit tests
        raise RuntimeError("Embedding model should not be called from unit tests")


if "faiss" not in sys.modules:
    faiss = types.ModuleType("faiss")
    faiss.METRIC_INNER_PRODUCT = 0

    class Index:  # pragma: no cover - structural stub
        pass

    class _Hnsw:
        efConstruction = None
        efSearch = None

    class IndexHNSWFlat(Index):
        def __init__(self, dim, m, metric):
            self.dim = dim
            self.m = m
            self.metric = metric
            self.hnsw = _Hnsw()
            self.added = None

        def add(self, embeddings):
            self.added = embeddings

    faiss.Index = Index
    faiss.IndexHNSWFlat = IndexHNSWFlat
    faiss.write_index = lambda index, path: None
    faiss.read_index = lambda path: IndexHNSWFlat(1, 1, faiss.METRIC_INNER_PRODUCT)
    sys.modules["faiss"] = faiss


if "pymorphy3" not in sys.modules:
    pymorphy3 = types.ModuleType("pymorphy3")

    class MorphAnalyzer:
        def parse(self, token):
            return [types.SimpleNamespace(normal_form=token.lower())]

    pymorphy3.MorphAnalyzer = MorphAnalyzer
    sys.modules["pymorphy3"] = pymorphy3


if "openai" not in sys.modules:
    openai = types.ModuleType("openai")

    class OpenAI:  # pragma: no cover - structural stub
        def __init__(self, *args, **kwargs):
            pass

    openai.OpenAI = OpenAI
    sys.modules["openai"] = openai


if "sentence_transformers" not in sys.modules:
    sentence_transformers = types.ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = lambda *args, **kwargs: DummyEmbeddingModel()
    sys.modules["sentence_transformers"] = sentence_transformers


if "transformers" not in sys.modules:
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: DummyTokenizer())
    transformers.AutoModelForSequenceClassification = types.SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: types.SimpleNamespace(to=lambda device: None, eval=lambda: None)
    )
    sys.modules["transformers"] = transformers
