# Frontend demo

A dependency-free static UI for the local RAG Series FastAPI backend.

## Run

Start the API from the project root:

```bash
uvicorn rag_series.api:app --host 0.0.0.0 --port 8000
```

Serve this folder with any static file server:

```bash
python -m http.server 5173 --directory frontend-demo
```

Open:

```text
http://127.0.0.1:5173
```

The default API base URL is `http://127.0.0.1:8000`. Change it in the UI if the backend runs elsewhere.
