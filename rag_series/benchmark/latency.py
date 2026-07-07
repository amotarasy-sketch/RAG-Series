#!/usr/bin/env python3
"""Benchmark /ask latency through RagService without running HTTP server.

Input JSONL schema:
{"question":"..."}

Example:
python benchmark/latency.py --questions eval/questions.example.jsonl --output benchmark/results/latency.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_series.service import RagService  # noqa: E402


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[idx]


def load_questions(path: Path) -> list[str]:
    questions: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            question = str(row.get("question", "")).strip()
            if not question:
                raise ValueError(f"{path}:{line_no}: missing question")
            questions.append(question)
    if not questions:
        raise ValueError(f"No questions found in {path}")
    return questions


def summarize(latencies_ms: list[float]) -> dict[str, Any]:
    return {
        "count": len(latencies_ms),
        "avg_ms": round(mean(latencies_ms), 2),
        "median_ms": round(median(latencies_ms), 2),
        "p50_ms": round(percentile(latencies_ms, 0.50), 2),
        "p95_ms": round(percentile(latencies_ms, 0.95), 2),
        "min_ms": round(min(latencies_ms), 2),
        "max_ms": round(max(latencies_ms), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark RAG answer latency.")
    parser.add_argument("--questions", default="eval/questions.example.jsonl")
    parser.add_argument("--output", default="benchmark/results/latency.json")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup requests before measuring.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of measured questions; 0 = all.")
    args = parser.parse_args()

    questions = load_questions(Path(args.questions))
    if args.limit > 0:
        questions = questions[: args.limit]

    service = RagService()
    service.load(rebuild=False)

    for question in questions[: args.warmup]:
        service.ask(question)

    rows: list[dict[str, Any]] = []
    for pos, question in enumerate(questions, start=1):
        started = perf_counter()
        result = service.ask(question)
        latency_ms = (perf_counter() - started) * 1000
        rows.append({
            "position": pos,
            "question": question,
            "latency_ms": round(latency_ms, 2),
            "service_latency_ms": result.get("latency_ms"),
            "source_count": result.get("source_count"),
        })
        print(f"[{pos}/{len(questions)}] {latency_ms:.2f} ms | {question}")

    summary = summarize([row["latency_ms"] for row in rows])
    payload = {"summary": summary, "rows": rows}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Latency summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"\nResults written to: {output_path}")


if __name__ == "__main__":
    main()
