# Исправления Docker/API

## Что исправлено

1. `torch.compile` для reranker больше не падает при отсутствии C/C++ компилятора:
   - в Dockerfile добавлены `gcc`, `g++`, `build-essential`;
   - перед включением `torch.compile` код проверяет наличие C/C++ компилятора;
   - если компилятора нет или `torch.compile` недоступен, reranker автоматически работает в eager-режиме.

2. `/ask` больше не требует ручного вызова `/build` после старта:
   - `API_LOAD_ON_STARTUP=true` выставлен по умолчанию;
   - `docker-compose.yml` также передает `API_LOAD_ON_STARTUP=true`;
   - добавлена lazy-load страховка в `/ask` и `/search`, если startup-load не выполнился.

3. Лог FAISS `No module named 'faiss.swigfaiss_avx2'` подавлен настройкой:
   - `FAISS_DISABLE_CPU_FEATURES=AVX2`.

   Это сообщение не было фатальной ошибкой: FAISS после него загружал обычную версию. Настройка нужна, чтобы не путать при чтении логов Docker.

4. Порт API вынесен в переменную:
   - `API_PORT=8000` по умолчанию;
   - если порт занят, можно запустить так: `API_PORT=8001 docker compose up --build`.

## Проверка

```bash
python -m compileall -q rag_series tests
pytest -q
```

Результат: `24 passed`.

## 2026-07-07 fixes

- Replaced embedding model `@lru_cache` initialization with an explicit process-wide singleton guarded by `threading.Lock`, so concurrent requests cannot create multiple `SentenceTransformer` instances before the cache is populated.
- Added `clear_embedding_model_cache()` for tests/controlled reloads and to clear dependent tokenizer/query caches safely.
- Fixed `/search` with `include_text=true`: chunk text is now looked up by `global_chunk_id` instead of zero-based FAISS/result `index`.
- Added regression tests for embedding singleton reuse and search text mapping.
- Verified with `python -m compileall` and `pytest`: 26 tests passing.

## 2026-07-07 cache-recovery fixes

- Fixed the failure shown by Docker logs where `SentenceTransformer(BAAI/bge-m3)` could crash with `json.decoder.JSONDecodeError` while `transformers` reads a cached Hugging Face JSON file.
- `get_embedding_model()` now removes empty/corrupt JSON files from the local HF cache for the configured embedding model and retries model loading once.
- Runtime load failures are now logged with a full traceback before returning HTTP 503, so `/build` errors are visible in Docker logs instead of only showing `503 Service Unavailable`.
- Added regression tests for Hugging Face cache cleanup and retry-on-JSONDecodeError.
- Verified with `pytest -q`: 28 tests passing.

## v3 - Hugging Face AutoProcessor workaround

- Removed the previous broad Hugging Face cache cleanup approach. Hugging Face stores zero-byte marker files under `.no_exist`; deleting or treating those as corrupt can make model loading unstable.
- Added a targeted workaround around `AutoProcessor.from_pretrained`: for text-only embedding models such as `BAAI/bge-m3`, a broken optional processor config no longer aborts loading the SentenceTransformer.
- Changed `get_chunk_tokenizer()` to load only `AutoTokenizer` instead of calling `get_embedding_model().tokenizer`, so chunking no longer forces the full embedding model to load before embeddings are actually created.
- Added a regression test for the broken optional `AutoProcessor` path.

## 2026-07-07 — Hugging Face cache recovery

Fixed Docker/API failures where `/build` and `/ask` crashed with `json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)` while loading `BAAI/bge-m3`. The app now detects invalid cached Hugging Face JSON files, refreshes the model snapshot once, and retries tokenizer, SentenceTransformer, and reranker loading. This avoids manual deletion of the `rag-series-models` Docker volume after interrupted downloads.

