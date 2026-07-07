# RAG Series Benchmark Report

Generated: `2026-06-30T05:21:16+00:00`

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
| Runtime | Python 3.13.5 on Linux-4.4.0-x86_64-with-glibc2.41 |

## Latency results

| Metric | Value |
| --- | ---: |
| Requests measured | 0 |
| Average | n/a |
| Median | n/a |
| P50 | n/a |
| P95 | n/a |
| Min | n/a |
| Max | n/a |

## Retrieval quality

| Stage | Recall@5 | Recall@10 | MRR |
| --- | ---: | ---: | ---: |
| Hybrid search before reranker | 0.0% | 0.0% | 0.000 |
| After cross-encoder reranker | 0.0% | 0.0% | 0.000 |

Evaluation questions: **0**

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
