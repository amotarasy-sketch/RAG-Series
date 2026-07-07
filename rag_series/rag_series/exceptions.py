from __future__ import annotations


class RagSeriesError(Exception):
    """Base class for domain errors that can be returned safely by the API."""

    status_code = 500
    error_code = "internal_error"
    public_message = "Internal server error."

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.public_message)
        self.public_message = message or self.public_message


class RuntimeNotReadyError(RagSeriesError):
    status_code = 503
    error_code = "runtime_not_ready"
    public_message = "RAG indexes are not loaded yet. Build or load indexes before calling this endpoint."


class RuntimeLoadError(RagSeriesError):
    status_code = 503
    error_code = "runtime_load_failed"
    public_message = "Failed to load RAG runtime indexes."


class IndexConsistencyError(RagSeriesError):
    status_code = 500
    error_code = "index_consistency_error"
    public_message = "RAG indexes are inconsistent. Rebuild the database indexes."


class FeatureDisabledError(RagSeriesError):
    status_code = 404
    error_code = "feature_disabled"
    public_message = "This API feature is disabled by configuration."
