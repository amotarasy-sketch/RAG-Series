#!/usr/bin/env python3
"""Generate a portfolio-ready benchmark report from latency and retrieval eval outputs.

Examples:
python benchmark/latency.py --questions eval/questions.example.jsonl --output benchmark/results/latency.json
python eval/run_eval.py --questions eval/questions.example.jsonl --output benchmark/results/retrieval_eval.jsonl
python benchmark/report.py --latency benchmark/results/latency.json --eval benchmark/results/retrieval_eval.jsonl --output benchmark/results/report.md
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def aggregate_eval(rows: list[dict[str, Any]], stage: str) -> dict[str, float]:
    if not rows:
        return {"recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0}
    return {
        "recall@5": mean(float(row[stage]["recall@5"]) for row in rows),
        "recall@10": mean(float(row[stage]["recall@10"]) for row in rows),
        "mrr": mean(float(row[stage]["mrr"]) for row in rows),
    }


def fmt_ms(value: Any) -> str:
    try:
        return f"{float(value):.2f} ms"
    except (TypeError, ValueError):
        return "n/a"


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def markdown_report(latency: dict[str, Any], eval_rows: list[dict[str, Any]]) -> str:
    latency_summary = latency.get("summary", {})
    before = aggregate_eval(eval_rows, "before")
    after = aggregate_eval(eval_rows, "after")
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    return f"""# RAG Series Benchmark Report

Generated: `{generated_at}`

## Executive summary

This report demonstrates production-oriented benchmarking for a local FastAPI RAG service over FB2 book collections. It covers request latency, retrieval quality before/after reranking, and operational observability signals suitable for portfolio review and backend/ML interviews.

## System under test

| Component | Implementation |
| --- | --- |
| API | FastAPI `/ask`, `/search`, `/build`, `/health` |
| Dense retrieval | FAISS vector search |
| Lexical retrieval | SQLite FTS5 / BM25 |
| Rank fusion | Reciprocal Rank Fusion |
| Optional reranking | Cross-encoder reranker |
| Context strategy | Neighbor chunk expansion |
| Observability | request IDs, structured logs, `/observability`, Prometheus-style `/metrics` |
| Runtime | Python {platform.python_version()} on {platform.platform()} |

## Latency results

| Metric | Value |
| --- | ---: |
| Requests measured | {latency_summary.get('count', 0)} |
| Average | {fmt_ms(latency_summary.get('avg_ms'))} |
| Median | {fmt_ms(latency_summary.get('median_ms'))} |
| P50 | {fmt_ms(latency_summary.get('p50_ms'))} |
| P95 | {fmt_ms(latency_summary.get('p95_ms'))} |
| Min | {fmt_ms(latency_summary.get('min_ms'))} |
| Max | {fmt_ms(latency_summary.get('max_ms'))} |

## Retrieval quality

| Stage | Recall@5 | Recall@10 | MRR |
| --- | ---: | ---: | ---: |
| Hybrid search before reranker | {pct(before['recall@5'])} | {pct(before['recall@10'])} | {before['mrr']:.3f} |
| After cross-encoder reranker | {pct(after['recall@5'])} | {pct(after['recall@10'])} | {after['mrr']:.3f} |

Evaluation questions: **{len(eval_rows)}**

## How to reproduce

```bash
rag-series build --rebuild
python benchmark/latency.py --questions eval/questions.example.jsonl --output benchmark/results/latency.json
python eval/run_eval.py --questions eval/questions.example.jsonl --output benchmark/results/retrieval_eval.jsonl
python benchmark/report.py --latency benchmark/results/latency.json --eval benchmark/results/retrieval_eval.jsonl --output benchmark/results/report.md
```

## Observability checks

Start the API:

```bash
uvicorn rag_series.api:app --host 0.0.0.0 --port 8000
```

Inspect runtime metrics:

```bash
curl http://127.0.0.1:8000/observability
curl http://127.0.0.1:8000/metrics
```

The API emits a stable `x-request-id` response header and JSON structured request logs with method, route, status code, client, and latency.

## Interview talking points

- Built a local-first RAG backend with separate semantic and lexical retrieval paths.
- Used RRF and optional reranking to improve answer grounding.
- Added benchmark automation for latency and retrieval quality rather than relying on anecdotal demos.
- Added dependency-light observability appropriate for local deployments and easy Prometheus/Grafana integration later.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark report Markdown and JSON summary.")
    parser.add_argument("--latency", default="benchmark/results/latency.json")
    parser.add_argument("--eval", default="benchmark/results/retrieval_eval.jsonl")
    parser.add_argument("--output", default="benchmark/results/report.md")
    parser.add_argument("--json-output", default="benchmark/results/report_summary.json")
    args = parser.parse_args()

    latency = load_json(Path(args.latency))
    eval_rows = load_jsonl(Path(args.eval))
    report = markdown_report(latency, eval_rows)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")

    summary = {
        "latency": latency.get("summary", {}),
        "retrieval_before": aggregate_eval(eval_rows, "before"),
        "retrieval_after": aggregate_eval(eval_rows, "after"),
        "eval_questions": len(eval_rows),
        "report_path": str(output),
    }
    json_output = Path(args.json_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Benchmark report written to: {output}")
    print(f"JSON summary written to: {json_output}")


if __name__ == "__main__":
    main()
