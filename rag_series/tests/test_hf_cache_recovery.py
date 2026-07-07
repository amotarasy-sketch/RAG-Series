import json
import sys
import types

from rag_series import models


def test_load_sentence_transformer_ignores_broken_optional_processor(monkeypatch) -> None:
    calls = []

    class FakeAutoProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append((args, kwargs))
            raise json.JSONDecodeError("Expecting value", "", 0)

    class FakeSentenceTransformer:
        def __init__(self, name: str, device: str) -> None:
            from transformers.models.auto.processing_auto import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(name)
            self.name = name
            self.device = device

    processing_auto = types.ModuleType("transformers.models.auto.processing_auto")
    processing_auto.AutoProcessor = FakeAutoProcessor
    monkeypatch.setitem(sys.modules, "transformers.models", types.ModuleType("transformers.models"))
    monkeypatch.setitem(sys.modules, "transformers.models.auto", types.ModuleType("transformers.models.auto"))
    monkeypatch.setitem(sys.modules, "transformers.models.auto.processing_auto", processing_auto)
    monkeypatch.setattr(models, "SentenceTransformer", FakeSentenceTransformer)

    loaded = models._load_sentence_transformer()

    assert isinstance(loaded, FakeSentenceTransformer)
    assert loaded.processor is None
    assert calls
