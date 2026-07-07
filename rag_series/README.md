# RAG Series — Local RAG Backend

A production-style **local Retrieval-Augmented Generation (RAG) service** for searching and asking questions over private FB2 book collections. The project combines semantic search, lexical search, reranking, context expansion, caching, CLI tooling, a FastAPI REST API, Docker, tests, evaluation scripts, latency benchmarking, observability endpoints, and benchmark report generation.

> Built as a portfolio project to demonstrate practical backend, ML/NLP, search, and API engineering skills.

## What this project does

RAG Series indexes a local folder of `.fb2` books and lets you ask natural-language questions about the collection. It retrieves relevant passages with a hybrid search pipeline, optionally reranks them, expands the context around the best chunks, and sends a grounded prompt to an OpenAI-compatible local LLM server such as LM Studio, Ollama, or vLLM.

The system is designed for private/offline-first usage: books and generated indexes stay on your machine.

## Highlights

- **Hybrid retrieval**: FAISS vector search + SQLite FTS5/BM25 keyword search.
- **RRF fusion**: semantic and lexical results are merged with Reciprocal Rank Fusion.
- **Reranking**: optional cross-encoder reranker for better top results.
- **Context expansion**: neighboring chunks are added around retrieved passages.
- **Grounded answers**: responses include source chunks and metadata.
- **LLM cache**: repeated questions can be served from a SQLite response cache.
- **FastAPI API**: `/ask`, `/search`, `/build`, `/health`, and `/config` endpoints.
- **CLI**: build indexes, ask one-off questions, or run an interactive chat.
- **Docker support**: reproducible local runtime with mounted books/indexes.
- **Quality tooling**: unit tests, ruff config, retrieval evaluation, latency benchmark, observability, and benchmark reporting.




## Architecture



Python
│
├── FastAPI
├── Typer
├── SQLite
├── FAISS
├── Sentence Transformers
├── Docker
├── PyTest
└── Ruff



```text
FB2 files
   │
   ▼
Parser → cleaning → sentence/token chunking
   │
   ├──────────────► FAISS dense vector index
   │
   └──────────────► SQLite FTS5 / BM25 lexical index
                       │
                       ▼
            Reciprocal Rank Fusion (RRF)
                       │
                       ▼
              Cross-encoder reranking
                       │
                       ▼
             Neighbor context expansion
                       │
                       ▼
        OpenAI-compatible local LLM server
                       │
                       ▼
          Answer + source chunk references
```

## Tech stack

| Area | Tools |
| --- | --- |
| Language | Python 3.10+ |
| API | FastAPI, Uvicorn, Pydantic |
| CLI | Typer |
| Vector search | FAISS |
| Lexical search | SQLite FTS5 / BM25 |
| NLP / ML | sentence-transformers, transformers, pymorphy3 |
| LLM integration | OpenAI-compatible API client |
| Packaging | pyproject.toml, setuptools |
| Dev quality | pytest, ruff, benchmark scripts, structured observability |
| Runtime | Docker, docker compose |




## Roadmap

See [ROADMAP.md](ROADMAP.md) for a GitHub issue-style roadmap with milestones, priorities, labels, scope, and acceptance criteria.

## Repository layout

```text
rag_series/
  api.py            # FastAPI application and request/response models
  cli.py            # Typer CLI entry points
  service.py        # Shared service layer for CLI/API/benchmarks
  config.py         # Environment-driven settings
  database.py       # Index build/load flow and manifest checks
  fb2_parser.py     # FB2 parsing
  chunking.py       # Sentence and token-based chunking
  indexes.py        # FAISS + SQLite FTS5 index creation/search
  retrieval.py      # Hybrid search and RRF fusion
  reranking.py      # Cross-encoder reranking
  context.py        # Neighbor chunk expansion
  llm.py            # Prompt construction and LLM calls
  llm_cache.py      # SQLite cache for repeated answers
  text_store.py     # JSONL chunk text storage

eval/               # Retrieval evaluation scripts
benchmark/          # Latency benchmarking scripts
tests/              # Unit tests
Dockerfile          # Container image
docker-compose.yml  # Local orchestration via Docker Compose
```

> Note: the actual file is named `docker-compose.yml` for compatibility with common Docker Compose workflows.

## Key Engineering Features

Modular backend architecture
• Dependency injection
• Configuration via environment
• Persistent cache
• Index versioning
• Retrieval evaluation
• Structured logging
• Docker deployment

## Engineering Challenges

Challenges solved
• Hybrid retrieval
• Duplicate removal
• Chunk overlap tuning
• Latency optimization
• Context expansion
• Reranking quality
• Response caching

## Quick start

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2. Install the project

```bash
pip install -e ".[dev]"
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

The project is portable by default:

- `Books/` contains demo `.fb2` books and can be replaced with your own books.
- `Chunks/` is created automatically and stores generated indexes, metadata, and caches.

You only need to edit `.env` if you want to override paths or connect your local LLM server:

```env
# Optional path overrides. By default these point to ./Books and ./Chunks.
# BOOKS_DIR=/absolute/path/to/fb2/books
# CHUNKS_DIR=/absolute/path/to/generated/indexes

LLM_BASE_URL=http://127.0.0.1:1234/v1
LLM_API_KEY=lm-studio
LLM_MODEL=local-model
```

### 4. Build indexes

```bash
rag-series build --rebuild
```

### 5. Ask a question

```bash
rag-series ask "Кто главный герой и что с ним произошло?"
```

### 6. Start interactive chat

```bash
rag-series chat
```

## REST API

Start the API:

```bash
uvicorn rag_series.api:app --host 0.0.0.0 --port 8000
```

Open interactive docs:

```text
http://127.0.0.1:8000/docs
```

### Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Runtime readiness check |
| `GET` | `/observability` | JSON runtime metrics for local dashboards/demos |
| `GET` | `/metrics` | Prometheus-compatible text metrics |
| `GET` | `/config` | Effective RAG configuration |
| `POST` | `/build?rebuild=true` | Build or load indexes |
| `POST` | `/search` | Retrieve relevant chunks without LLM generation |
| `POST` | `/ask` | Full RAG answer with sources |

### Example request

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Кто главный герой и что с ним произошло?"}'
```

Example response shape:

```json
{
  "answer": "...",
  "sources": [
    {
      "book": "...",
      "global_chunk_id": 42,
      "hybrid_score": 0.031,
      "retrieval_source": "faiss_semantic+fts5_bm25_lemmas"
    }
  ],
  "source_count": 1,
  "latency_ms": 1234.56
}
```

## Frontend demo

A small static frontend is included in `frontend-demo/` for quick local API demos.

Start the API:

```bash
uvicorn rag_series.api:app --host 0.0.0.0 --port 8000
```

Serve the frontend:

```bash
python -m http.server 5173 --directory frontend-demo
```

Open `http://127.0.0.1:5173` and use the API base URL field if your backend runs somewhere other than `http://127.0.0.1:8000`.


## Observability

The API includes dependency-light observability that is useful for demos, debugging, and portfolio discussions:

- `x-request-id` response header for request tracing.
- Structured JSON request logs with method, path, status code, client, and latency.
- `/observability` for human-readable runtime metrics.
- `/metrics` for Prometheus-style text metrics without adding another dependency.

Example:

```bash
curl http://127.0.0.1:8000/observability
curl http://127.0.0.1:8000/metrics
```

You can also write logs to a file by setting `LOG_FILE=rag.log`.

## Benchmark report

Generate latency and retrieval quality artifacts, then compile them into a Markdown report that can be shared with recruiters or discussed in interviews.

```bash
rag-series build --rebuild
python benchmark/latency.py --questions eval/questions.example.jsonl --output benchmark/results/latency.json
python eval/run_eval.py --questions eval/questions.example.jsonl --output benchmark/results/retrieval_eval.jsonl
python benchmark/report.py \
  --latency benchmark/results/latency.json \
  --eval benchmark/results/retrieval_eval.jsonl \
  --output benchmark/results/report.md
```

The report includes latency summary, Recall@5, Recall@10, MRR before/after reranking, reproducibility commands, and observability checks.

## Docker

Build the image:

```bash
docker compose build
```

Build indexes:

```bash
docker compose run --rm rag-series build --rebuild
```

Ask a question:

```bash
docker compose run --rm rag-series ask "Ваш вопрос"
```

Start the API:

```bash
docker compose up rag-api
```

In Docker, books and generated indexes are mounted into the container:

```text
./Books  → /data/books
./Chunks → /data/chunks
```

Do not commit private books or generated indexes. They are intentionally ignored by `.gitignore`.

## Evaluation

Create a JSONL file with questions and relevant chunk IDs:

```jsonl
{"id":"q001","question":"...","relevant_chunk_ids":[123,456]}
```

Run retrieval evaluation:

```bash
python eval/run_eval.py --questions eval/questions.example.jsonl
```

## Latency benchmark

```bash
python benchmark/latency.py \
  --questions eval/questions.example.jsonl \
  --output benchmark/results/latency.json
```

## Development

Run checks:

```bash
ruff check .
ruff format .
pytest
```

Useful CLI commands:

```bash
rag-series info
rag-series build --rebuild
rag-series ask "..." --debug
rag-series chat --debug
```

## Configuration reference

Most behavior is controlled through environment variables. Key settings include:

| Variable | Purpose |
| --- | --- |
| `BOOKS_DIR` | Source folder with `.fb2` files. Defaults to `./Books` |
| `CHUNKS_DIR` | Folder for generated chunks, indexes, metadata, and caches. Defaults to `./Chunks` |
| `LLM_BASE_URL` | OpenAI-compatible LLM API URL |
| `LLM_API_KEY` | API key or local placeholder key |
| `LLM_MODEL` | Model name exposed by the local LLM server |
| `EMBEDDING_MODEL_NAME` | Sentence-transformers embedding model |
| `RERANKER_MODEL_NAME` | Cross-encoder reranker model |
| `SEMANTIC_TOP_K` | Number of FAISS candidates |
| `BM25_TOP_K` | Number of lexical candidates |
| `HYBRID_TOP_K` | Number of fused candidates |
| `RERANK_TOP_K` | Final reranked results used for context |
| `MAX_CONTEXT_TOKENS` | Maximum prompt context size |
| `ENABLE_LLM_CACHE` | Enable/disable SQLite answer cache |

See `.env.example` for the full list.

## Why this project is useful for employers

This repository demonstrates the ability to design and implement a real backend/ML application end to end:

- building a retrieval pipeline from raw documents to queryable indexes;
- combining vector search, lexical search, reranking, and context construction;
- exposing the same business logic through CLI, API, evaluation, and benchmark layers;
- keeping runtime configuration externalized and Docker-friendly;
- writing tests around indexing, text processing, and query handling;
- separating private local data from source code and deployment artifacts.

## License

MIT. See [LICENSE](LICENSE).

## Production API behavior

The FastAPI app now separates liveness from readiness:

- `GET /health` always returns API liveness plus runtime state.
- `GET /ready` returns `200` only when indexes are loaded and `503` otherwise.
- `API_LOAD_ON_STARTUP=true` loads indexes during the FastAPI lifespan startup hook.
- `API_LOAD_ON_STARTUP=false` keeps startup lightweight; call `POST /build` or load indexes from your own deployment workflow before using `/ask` and `/search`.
- `API_ENABLE_BUILD_ENDPOINT=false` disables the HTTP build endpoint for production deployments where indexing should not be triggerable over the network.

Error responses use a stable shape:

```json
{
  "error": "runtime_not_ready",
  "message": "RAG indexes are not loaded yet. Build or load indexes before calling this endpoint.",
  "request_id": "optional-correlation-id"
}
```

The API schema includes typed models for chunk metadata, retrieval results, source responses and eval questions.

## CI

GitHub Actions runs on pushes and pull requests to `main`:

- `ruff check .`
- `ruff format --check .`
- `pytest`
- `python -m build`



