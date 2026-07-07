from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Query, Request, Response
from fastapi.responses import JSONResponse

from .config import CFG
from .exceptions import FeatureDisabledError, RagSeriesError
from .observability import RequestObservabilityMiddleware, metrics
from .schemas import (
    AskRequest,
    AskResponse,
    BuildResponse,
    ConfigResponse,
    ErrorResponse,
    HealthResponse,
    SearchRequest,
    SearchResponse,
)
from .logging_utils import logger
from .service import RagService, get_service


@asynccontextmanager
async def lifespan(app_: FastAPI) -> AsyncIterator[None]:
    """Load runtime during startup when API_LOAD_ON_STARTUP=true.

    Startup loading is best-effort: a missing/invalid index should not prevent
    /health, /config or /docs from opening. /ask and /search still try a lazy
    load and return a clear 503 if the runtime cannot be built or loaded.
    """
    if CFG.api_load_on_startup:
        service_factory = app_.dependency_overrides.get(get_service, get_service)
        service = service_factory()
        load_on_startup = getattr(service, "load_on_startup", None)
        if callable(load_on_startup):
            try:
                load_on_startup()
            except RagSeriesError as exc:
                logger.warning("Startup runtime load failed; API will retry lazily on request: %s", exc)
    yield


app = FastAPI(
    title="RAG Series API",
    version="0.3.0",
    description="REST API for the local FB2 book-series RAG assistant.",
    lifespan=lifespan,
    responses={
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
app.add_middleware(RequestObservabilityMiddleware)


def _request_id(request: Request) -> str | None:
    return request.headers.get("x-request-id")


@app.exception_handler(RagSeriesError)
async def rag_series_exception_handler(request: Request, exc: RagSeriesError) -> JSONResponse:
    payload = ErrorResponse(
        error=exc.error_code,
        message=exc.public_message,
        request_id=_request_id(request),
    )
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    payload = ErrorResponse(
        error="internal_error",
        message="Internal server error.",
        request_id=_request_id(request),
    )
    return JSONResponse(status_code=500, content=payload.model_dump())


@app.get("/health", response_model=HealthResponse)
def health(service: RagService = Depends(get_service)) -> dict[str, Any]:
    readiness = service.readiness() if hasattr(service, "readiness") else {"ready": service.is_ready()}
    return {
        "status": "ok",
        "ready": bool(readiness.get("ready", False)),
        "indexes_loaded": bool(readiness.get("indexes_loaded", readiness.get("ready", False))),
        "chunks_loaded": int(readiness.get("chunks_loaded", 0)),
        "faiss_vectors": int(readiness.get("faiss_vectors", 0)),
        "lexical_index_loaded": bool(readiness.get("lexical_index_loaded", False)),
        "neighbor_index_loaded": bool(readiness.get("neighbor_index_loaded", False)),
        "startup_load_enabled": bool(readiness.get("startup_load_enabled", CFG.api_load_on_startup)),
    }


@app.get("/ready", response_model=HealthResponse)
def ready(service: RagService = Depends(get_service)) -> tuple[dict[str, Any], int] | dict[str, Any]:
    payload = health(service)
    if not payload["ready"]:
        return JSONResponse(status_code=503, content=payload)  # type: ignore[return-value]
    return payload


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, service: RagService = Depends(get_service)) -> dict[str, Any]:
    return service.ask(payload.question.strip())


@app.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest, service: RagService = Depends(get_service)) -> dict[str, Any]:
    return service.search(
        payload.query.strip(),
        top_k=payload.top_k,
        include_text=payload.include_text,
    )


@app.post("/build", response_model=BuildResponse)
def build(
    rebuild: bool = Query(False, description="Force index rebuild."),
    service: RagService = Depends(get_service),
) -> dict[str, Any]:
    if not CFG.api_enable_build_endpoint:
        raise FeatureDisabledError("The build endpoint is disabled. Set API_ENABLE_BUILD_ENDPOINT=true.")
    if rebuild:
        return service.rebuild()
    return service.build_summary()


@app.get("/observability")
def observability() -> dict[str, Any]:
    """Human-readable runtime metrics for dashboards and demos."""
    return metrics.snapshot()


@app.get("/metrics")
def prometheus_metrics() -> Response:
    """Prometheus-compatible text exposition without extra dependencies."""
    return Response(content=metrics.prometheus_text(), media_type="text/plain; version=0.0.4")


@app.get("/config", response_model=ConfigResponse)
def config(service: RagService = Depends(get_service)) -> dict[str, Any]:
    return service.config()
