import re
from typing import List

from .config import CFG
from .models import get_chunk_tokenizer
from .text_utils import count_tokens, count_tokens_batch, normalize_spaces


def split_into_sentences(text: str) -> List[str]:
    text = normalize_spaces(text)
    parts = re.split(r"(?<=[.!?…])\s+(?=[А-ЯЁA-Z0-9—\"«])", text)
    return [p.strip() for p in parts if p.strip()]


def split_long_text_by_tokens(text: str, max_tokens: int, overlap_tokens: int = 0) -> List[str]:
    ids = get_chunk_tokenizer().encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(ids):
        end = min(start + max_tokens, len(ids))
        piece = get_chunk_tokenizer().decode(ids[start:end], skip_special_tokens=True).strip()
        if piece:
            chunks.append(piece)
        if end >= len(ids):
            break
        start = max(0, end - overlap_tokens)
    return chunks


def chunk_section_by_tokens(section_text: str) -> List[str]:
    sentences = split_into_sentences(section_text)
    sentence_items = list(zip(sentences, count_tokens_batch(sentences)))

    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    def flush_current() -> None:
        nonlocal current, current_tokens
        if current:
            chunk = normalize_spaces(" ".join(current))
            if chunk:
                chunks.append(chunk)
        current = []
        current_tokens = 0

    for sentence, sent_tokens in sentence_items:
        if sent_tokens > CFG.chunk_tokens:
            flush_current()
            chunks.extend(split_long_text_by_tokens(sentence, CFG.chunk_tokens, CFG.overlap_tokens))
            continue

        if current_tokens + sent_tokens <= CFG.chunk_tokens:
            current.append(sentence)
            current_tokens += sent_tokens
        else:
            flush_current()
            if chunks and CFG.overlap_tokens > 0:
                prev_sentences = split_into_sentences(chunks[-1])
                overlap: List[str] = []
                overlap_count = 0
                for prev in reversed(prev_sentences):
                    t = count_tokens(prev)
                    if overlap_count + t <= CFG.overlap_tokens:
                        overlap.insert(0, prev)
                        overlap_count += t
                    else:
                        break
                current = overlap + [sentence]
                current_tokens = overlap_count + sent_tokens
            else:
                current = [sentence]
                current_tokens = sent_tokens

    flush_current()

    merged: List[str] = []
    merged_token_counts: List[int] = []
    for chunk in chunks:
        chunk_tokens = count_tokens(chunk)
        if merged and chunk_tokens < CFG.min_chunk_tokens:
            candidate = normalize_spaces(merged[-1] + " " + chunk)
            candidate_tokens = count_tokens(candidate)
            if candidate_tokens <= CFG.chunk_tokens + CFG.overlap_tokens:
                merged[-1] = candidate
                merged_token_counts[-1] = candidate_tokens
            else:
                merged.append(chunk)
                merged_token_counts.append(chunk_tokens)
        else:
            merged.append(chunk)
            merged_token_counts.append(chunk_tokens)

    return merged
