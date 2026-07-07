#!/usr/bin/env python3
"""Evaluate retrieval quality before and after reranking.

Input JSONL schema, one question per line:
{"id":"q001", "question":"...", "relevant_chunk_ids":[123, 456]}

Accepted aliases for relevant ids:
- relevant_chunk_ids
- expected_chunk_ids
- global_chunk_ids
- relevant_chunks
- relevant

`relevant` / `relevant_chunks` may contain integers or objects with `global_chunk_id` / `chunk_id` / `id`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

# Allow running as `python eval/run_eval.py` from the repository root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_series.config import CFG  # noqa: E402
from rag_series.database import load_database  # noqa: E402
from rag_series.retrieval import hybrid_search  # noqa: E402
from rag_series.reranking import rerank  # noqa: E402
from rag_series.schemas import EvalQuestion  # noqa: E402
from rag_series.text_store import load_needed_chunk_texts  # noqa: E402


def _extract_relevant_ids(row: dict[str, Any]) -> set[int]:
    for key in (
        "relevant_chunk_ids",
        "expected_chunk_ids",
        "global_chunk_ids",
        "relevant_chunks",
        "relevant",
    ):
        if key in row:
            raw = row[key]
            break
    else:
        raise ValueError("missing relevant ids field")

    if not isinstance(raw, list):
        raise ValueError("relevant ids field must be a list")

    ids: set[int] = set()
    for item in raw:
        if isinstance(item, int):
            ids.add(item)
        elif isinstance(item, str) and item.isdigit():
            ids.add(int(item))
        elif isinstance(item, dict):
            for field in ("global_chunk_id", "chunk_id", "id"):
                if field in item:
                    ids.add(int(item[field]))
                    break
            else:
                raise ValueError(f"relevant object has no chunk id field: {item!r}")
        else:
            raise ValueError(f"unsupported relevant id item: {item!r}")

    if not ids:
        raise ValueError("relevant ids field is empty")
    return ids


def load_questions(path: Path) -> list[EvalQuestion]:
    questions: list[EvalQuestion] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            question = str(row.get("question", "")).strip()
            if not question:
                raise ValueError(f"{path}:{line_no}: missing question")
            qid = str(row.get("id") or f"line_{line_no}")
            try:
                relevant_ids = _extract_relevant_ids(row)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            questions.append(EvalQuestion(qid, question, relevant_ids))

    if not questions:
        raise ValueError(f"No questions found in {path}")
    return questions


def chunk_ids(results: Iterable[dict[str, Any]]) -> list[int]:
    return [int(r["global_chunk_id"]) for r in results]


def recall_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int) -> float:
    return 1.0 if relevant_ids.intersection(ranked_ids[:k]) else 0.0


def reciprocal_rank(ranked_ids: list[int], relevant_ids: set[int]) -> float:
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_one(
    item: EvalQuestion,
    metadata: list[dict[str, Any]],
    faiss_index: Any,
    lexical_db_path: str,
    rerank_top_k: int,
) -> dict[str, Any]:
    before = hybrid_search(item.question, metadata, faiss_index, lexical_db_path)
    before_ids = chunk_ids(before)

    # Rerank the full hybrid candidate set, then keep enough results to compute Recall@10.
    rerank_limit = max(rerank_top_k, 10)
    chunk_texts = load_needed_chunk_texts(before)
    after = rerank(item.question, before, top_k=rerank_limit, chunk_texts=chunk_texts)
    after_ids = chunk_ids(after)

    return {
        "id": item.id,
        "question": item.question,
        "relevant_chunk_ids": sorted(item.relevant_chunk_ids),
        "before": {
            "top_ids": before_ids[:10],
            "recall@5": recall_at_k(before_ids, item.relevant_chunk_ids, 5),
            "recall@10": recall_at_k(before_ids, item.relevant_chunk_ids, 10),
            "mrr": reciprocal_rank(before_ids, item.relevant_chunk_ids),
        },
        "after": {
            "top_ids": after_ids[:10],
            "recall@5": recall_at_k(after_ids, item.relevant_chunk_ids, 5),
            "recall@10": recall_at_k(after_ids, item.relevant_chunk_ids, 10),
            "mrr": reciprocal_rank(after_ids, item.relevant_chunk_ids),
        },
    }


def aggregate(rows: list[dict[str, Any]], stage: str) -> dict[str, float]:
    return {
        "recall@5": mean(row[stage]["recall@5"] for row in rows),
        "recall@10": mean(row[stage]["recall@10"] for row in rows),
        "mrr": mean(row[stage]["mrr"] for row in rows),
    }


def print_summary(rows: list[dict[str, Any]]) -> None:
    before = aggregate(rows, "before")
    after = aggregate(rows, "after")

    print("\n=== Evaluation summary ===")
    print(f"Questions: {len(rows)}")
    print("\nStage              Recall@5  Recall@10  MRR")
    print("--------------------------------------------")
    print(f"Before reranker   {before['recall@5']:.4f}    {before['recall@10']:.4f}     {before['mrr']:.4f}")
    print(f"After reranker    {after['recall@5']:.4f}    {after['recall@10']:.4f}     {after['mrr']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval before/after reranker.")
    parser.add_argument(
        "--questions",
        default="eval/questions.example.jsonl",
        help="Path to JSONL questions file.",
    )
    parser.add_argument(
        "--output",
        default="eval/results.jsonl",
        help="Where to write per-question JSONL results.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild indexes before evaluation.",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=CFG.rerank_top_k,
        help="Reranker top_k; internally at least 10 are kept for Recall@10.",
    )
    args = parser.parse_args()

    questions_path = Path(args.questions)
    output_path = Path(args.output)

    questions = load_questions(questions_path)
    _, metadata, faiss_index, lexical_db_path = load_database(rebuild=args.rebuild)

    rows: list[dict[str, Any]] = []
    for pos, item in enumerate(questions, start=1):
        print(f"[{pos}/{len(questions)}] {item.id}: {item.question}")
        rows.append(
            evaluate_one(
                item,
                metadata=metadata,
                faiss_index=faiss_index,
                lexical_db_path=lexical_db_path,
                rerank_top_k=args.rerank_top_k,
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print_summary(rows)
    print(f"\nPer-question results written to: {output_path}")


if __name__ == "__main__":
    main()
