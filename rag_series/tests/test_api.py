from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rag_series.api import app
from rag_series.service import get_service


class FakeFaissIndex:
    ntotal = 2


class FakeService:
    def __init__(self) -> None:
        self.rebuilt = False

    def is_ready(self) -> bool:
        return True

    def readiness(self):
        return {
            "ready": True,
            "indexes_loaded": True,
            "chunks_loaded": 2,
            "faiss_vectors": 2,
            "lexical_index_loaded": True,
            "neighbor_index_loaded": True,
            "startup_load_enabled": False,
        }

    def ask(self, question: str):
        return {
            "answer": f"Answer for: {question}",
            "sources": [{"book": "demo.fb2", "global_chunk_id": 1}],
            "source_count": 1,
            "latency_ms": 12.5,
        }

    def search(self, query: str, top_k: int = 10, include_text: bool = False):
        results = [
            {
                "book": "demo.fb2",
                "index": 1,
                "hybrid_score": 0.42,
                **({"text": "Demo chunk text"} if include_text else {}),
            }
        ][:top_k]
        return {"query": query, "results": results, "count": len(results), "latency_ms": 3.0}

    def rebuild(self):
        self.rebuilt = True
        return {
            "status": "ready",
            "chunks": 2,
            "faiss_vectors": 2,
            "lexical_db_path": "/tmp/demo.sqlite3",
            "neighbor_index_size": 2,
            "latency_ms": 10.0,
        }

    def load(self, rebuild: bool = False):
        return ([{"book": "demo.fb2"}, {"book": "demo.fb2"}], FakeFaissIndex(), "/tmp/demo.sqlite3", {0: []})

    def build_summary(self):
        return {
            "status": "ready",
            "chunks": 2,
            "faiss_vectors": 2,
            "lexical_db_path": "/tmp/demo.sqlite3",
            "neighbor_index_size": 2,
            "latency_ms": 0.0,
        }

    def config(self):
        return {
            "books_dir": "/tmp/books",
            "chunks_dir": "/tmp/chunks",
            "embedding_model_name": "fake-embedding",
            "reranker_model_name": "fake-reranker",
            "llm_base_url": "http://localhost:1234/v1",
            "llm_model": "fake-llm",
            "semantic_top_k": 10,
            "bm25_top_k": 10,
            "hybrid_top_k": 10,
            "rerank_top_k": 5,
            "neighbor_window": 1,
            "max_context_chunks": 4,
            "max_context_tokens": 1024,
            "enable_fast_mode": True,
            "enable_llm_cache": False,
            "api_load_on_startup": False,
            "api_enable_build_endpoint": True,
        }


@pytest.fixture
def client():
    fake_service = FakeService()
    app.dependency_overrides[get_service] = lambda: fake_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health_returns_ready_status(client: TestClient) -> None:
    response = client.get("/health", headers={"x-request-id": "test-request-id"})

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["ready"] is True
    assert response.json()["indexes_loaded"] is True
    assert response.json()["chunks_loaded"] == 2
    assert response.headers["x-request-id"] == "test-request-id"


def test_ready_returns_503_when_indexes_are_not_loaded() -> None:
    class NotReadyService(FakeService):
        def is_ready(self) -> bool:
            return False

        def readiness(self):
            return {
                "ready": False,
                "indexes_loaded": False,
                "chunks_loaded": 0,
                "faiss_vectors": 0,
                "lexical_index_loaded": False,
                "neighbor_index_loaded": False,
                "startup_load_enabled": False,
            }

    app.dependency_overrides[get_service] = lambda: NotReadyService()
    with TestClient(app) as test_client:
        response = test_client.get("/ready")
    app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json()["ready"] is False


def test_ask_trims_question_and_returns_sources(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "  What happened?  "})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Answer for: What happened?"
    assert payload["source_count"] == 1
    assert payload["sources"][0]["book"] == "demo.fb2"


def test_ask_rejects_empty_question(client: TestClient) -> None:
    response = client.post("/ask", json={"question": ""})

    assert response.status_code == 422


def test_search_supports_top_k_and_include_text(client: TestClient) -> None:
    response = client.post("/search", json={"query": "dragon", "top_k": 1, "include_text": True})

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "dragon"
    assert payload["count"] == 1
    assert payload["results"][0]["text"] == "Demo chunk text"


def test_search_rejects_invalid_top_k(client: TestClient) -> None:
    response = client.post("/search", json={"query": "dragon", "top_k": 0})

    assert response.status_code == 422


def test_build_loads_existing_runtime_by_default(client: TestClient) -> None:
    response = client.post("/build")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["faiss_vectors"] == 2


def test_build_can_force_rebuild(client: TestClient) -> None:
    response = client.post("/build?rebuild=true")

    assert response.status_code == 200
    assert response.json()["chunks"] == 2


def test_config_returns_effective_settings(client: TestClient) -> None:
    response = client.get("/config")

    assert response.status_code == 200
    assert response.json()["embedding_model_name"] == "fake-embedding"


def test_metrics_endpoint_returns_prometheus_text(client: TestClient) -> None:
    client.get("/health")
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "rag_series_http_requests_total" in response.text
    assert 'path="/health"' in response.text


def test_observability_returns_snapshot(client: TestClient) -> None:
    client.get("/health")
    response = client.get("/observability")

    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "rag-series-api"
    assert "GET /health" in payload["endpoints"]
