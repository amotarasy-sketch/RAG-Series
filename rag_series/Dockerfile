# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BOOKS_DIR=/data/books \
    CHUNKS_DIR=/data/chunks \
    HF_HOME=/models/huggingface \
    TORCH_HOME=/models/torch \
    TRANSFORMERS_CACHE=/models/huggingface \
    FAISS_DISABLE_CPU_FEATURES=AVX2

WORKDIR /app

# libgomp1 is required by FAISS / scientific Python wheels.
# gcc/g++/build-essential are required when COMPILE_RERANKER=true because
# torch.compile / PyTorch Inductor compiles native code at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        gcc \
        g++ \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt README.md ./
COPY rag_series ./rag_series
COPY eval ./eval
COPY run.py ./run.py

RUN pip install --upgrade pip \
    && pip install .

RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /data/books /data/chunks /models \
    && chown -R appuser:appuser /app /data /models

USER appuser

VOLUME ["/data/books", "/data/chunks", "/models"]

ENTRYPOINT ["rag-series"]
CMD ["--help"]
