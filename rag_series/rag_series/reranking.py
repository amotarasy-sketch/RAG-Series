from typing import Any, Dict, List

import torch

from .config import CFG
from .models import DEVICE, get_reranker
from .text_store import ChunkTextStore, chunk_text_for_metadata


def should_skip_rerank(results: List[Dict[str, Any]]) -> bool:
    if not CFG.enable_fast_mode:
        return False
    if len(results) < CFG.fast_min_candidates:
        return False

    top1 = results[0]
    top2 = results[1]

    s1 = top1.get("semantic_score") or 0.0
    s2 = top2.get("semantic_score") or 0.0
    semantic_confident = s1 >= CFG.fast_semantic_threshold and (s1 - s2) >= CFG.fast_margin

    b1 = top1.get("bm25_score") or 0.0
    b2 = top2.get("bm25_score") or 0.0
    bm25_confident = b1 >= CFG.fast_bm25_threshold and (b1 - b2) >= CFG.fast_margin * 10

    return semantic_confident or bm25_confident


def rerank(
    query: str,
    results: List[Dict[str, Any]],
    top_k: int,
    chunk_texts: ChunkTextStore,
) -> List[Dict[str, Any]]:
    if not results:
        return []

    reranker_tokenizer, reranker_model = get_reranker()

    all_scores: List[float] = []
    batch_size = max(1, int(CFG.reranker_batch_size))

    for i in range(0, len(results), batch_size):
        batch = results[i:i + batch_size]
        pairs = [(query, chunk_text_for_metadata(r, chunk_texts)) for r in batch]

        inputs = reranker_tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=CFG.reranker_max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(DEVICE, non_blocking=True) for k, v in inputs.items()}

        with torch.inference_mode():
            if DEVICE == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = reranker_model(**inputs).logits
            else:
                logits = reranker_model(**inputs).logits
            scores = logits.view(-1).float().detach().cpu().numpy().tolist()

        all_scores.extend(float(s) for s in scores)

    reranked = []
    for r, score in zip(results, all_scores):
        item = dict(r)
        item["rerank_score"] = float(score)
        reranked.append(item)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]
