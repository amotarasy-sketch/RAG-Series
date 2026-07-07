from __future__ import annotations

import rag_series.models as models


class DummyModel:
    tokenizer = object()


def test_get_embedding_model_reuses_singleton(monkeypatch) -> None:
    models.clear_embedding_model_cache()
    calls: list[tuple[str, str]] = []

    def fake_sentence_transformer(model_name: str, device: str) -> DummyModel:
        calls.append((model_name, device))
        return DummyModel()

    monkeypatch.setattr(models, "SentenceTransformer", fake_sentence_transformer)

    first = models.get_embedding_model()
    second = models.get_embedding_model()

    assert first is second
    assert calls == [(models.CFG.embedding_model_name, models.DEVICE)]

    models.clear_embedding_model_cache()
