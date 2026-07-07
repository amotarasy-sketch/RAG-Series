from __future__ import annotations

import json
import logging
import os
import platform
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = logging.getLogger("rag_series.api")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EndpointStats:
    count: int = 0
    errors: int = 0
    total_latency_ms: float = 0.0
    min_latency_ms: float | None = None
    max_latency_ms: float = 0.0
    latencies_ms: deque[float] = field(default_factory=lambda: deque(maxlen=1000))

    def observe(self, latency_ms: float, is_error: bool) -> None:
        self.count += 1
        self.errors += int(is_error)
        self.total_latency_ms += latency_ms
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)
        self.min_latency_ms = latency_ms if self.min_latency_ms is None else min(self.min_latency_ms, latency_ms)
        self.latencies_ms.append(latency_ms)

    def summary(self) -> dict[str, Any]:
        values = sorted(self.latencies_ms)
        avg = self.total_latency_ms / self.count if self.count else 0.0
        return {
            "count": self.count,
            "errors": self.errors,
            "error_rate": round(self.errors / self.count, 4) if self.count else 0.0,
            "avg_ms": round(avg, 2),
            "p50_ms": round(percentile(values, 0.50), 2),
            "p95_ms": round(percentile(values, 0.95), 2),
            "p99_ms": round(percentile(values, 0.99), 2),
            "min_ms": round(self.min_latency_ms or 0.0, 2),
            "max_ms": round(self.max_latency_ms, 2),
        }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
    return values[index]


class MetricsRegistry:
    """Small dependency-free metrics registry for local portfolio/demo deployments."""

    def __init__(self) -> None:
        self.started_at = utc_now_iso()
        self._lock = Lock()
        self._stats: dict[str, EndpointStats] = defaultdict(EndpointStats)
        self._status_counts: dict[int, int] = defaultdict(int)

    def observe_request(self, method: str, path: str, status_code: int, latency_ms: float) -> None:
        key = f"{method} {path}"
        with self._lock:
            self._stats[key].observe(latency_ms, status_code >= 500)
            self._status_counts[status_code] += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "service": "rag-series-api",
                "started_at": self.started_at,
                "generated_at": utc_now_iso(),
                "runtime": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "pid": os.getpid(),
                },
                "status_codes": dict(sorted(self._status_counts.items())),
                "endpoints": {name: stats.summary() for name, stats in sorted(self._stats.items())},
            }

    def prometheus_text(self) -> str:
        snapshot = self.snapshot()
        lines = [
            "# HELP rag_series_http_requests_total Total HTTP requests by endpoint and status.",
            "# TYPE rag_series_http_requests_total counter",
        ]
        for endpoint, stats in snapshot["endpoints"].items():
            method, path = endpoint.split(" ", 1)
            lines.append(
                f' rag_series_http_requests_total{{method="{method}",path="{path}"}} {stats["count"]}'
            )
        lines.extend([
            "# HELP rag_series_http_errors_total Total HTTP 5xx responses by endpoint.",
            "# TYPE rag_series_http_errors_total counter",
        ])
        for endpoint, stats in snapshot["endpoints"].items():
            method, path = endpoint.split(" ", 1)
            lines.append(
                f' rag_series_http_errors_total{{method="{method}",path="{path}"}} {stats["errors"]}'
            )
        lines.extend([
            "# HELP rag_series_http_latency_ms Latency summary in milliseconds.",
            "# TYPE rag_series_http_latency_ms gauge",
        ])
        for endpoint, stats in snapshot["endpoints"].items():
            method, path = endpoint.split(" ", 1)
            for metric in ("avg_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms"):
                lines.append(
                    f' rag_series_http_latency_ms{{method="{method}",path="{path}",quantile="{metric}"}} {stats[metric]}'
                )
        return "\n".join(line.strip() for line in lines) + "\n"


metrics = MetricsRegistry()


class RequestObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            latency_ms = (time.perf_counter() - started) * 1000
            route_path = getattr(request.scope.get("route"), "path", request.url.path)
            metrics.observe_request(request.method, route_path, status_code, latency_ms)
            logger.info(
                json.dumps(
                    {
                        "event": "http_request",
                        "request_id": request_id,
                        "method": request.method,
                        "path": route_path,
                        "status_code": status_code,
                        "latency_ms": round(latency_ms, 2),
                        "client": request.client.host if request.client else None,
                    },
                    ensure_ascii=False,
                )
            )
            try:
                response.headers["x-request-id"] = request_id  # type: ignore[name-defined]
            except UnboundLocalError:
                pass
