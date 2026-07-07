from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag_series import chunking


class WordTokenizer:
    def __init__(self) -> None:
        self.tokens: list[str] = []

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        self.tokens = text.split()
        return list(range(len(self.tokens)))

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(self.tokens[i] for i in ids)


def test_split_into_sentences_handles_russian_and_latin_text() -> None:
    text = "Привет! Как дела? Всё хорошо… Next sentence."

    assert chunking.split_into_sentences(text) == [
        "Привет!",
        "Как дела?",
        "Всё хорошо…",
        "Next sentence.",
    ]


def test_split_long_text_by_tokens_respects_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = WordTokenizer()
    monkeypatch.setattr(chunking, "get_chunk_tokenizer", lambda: tokenizer)

    chunks = chunking.split_long_text_by_tokens("one two three four five", max_tokens=3, overlap_tokens=1)

    assert chunks == ["one two three", "three four five"]


def test_chunk_section_by_tokens_adds_overlap_between_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        chunking,
        "CFG",
        SimpleNamespace(chunk_tokens=4, overlap_tokens=2, min_chunk_tokens=2),
    )
    monkeypatch.setattr(chunking, "count_tokens", lambda text: len(text.split()))
    monkeypatch.setattr(chunking, "count_tokens_batch", lambda texts: [len(text.split()) for text in texts])

    text = "Alpha beta. Gamma delta. Epsilon zeta. Tail."

    assert chunking.chunk_section_by_tokens(text) == [
        "Alpha beta. Gamma delta.",
        "Gamma delta. Epsilon zeta.",
        "Epsilon zeta. Tail.",
    ]
