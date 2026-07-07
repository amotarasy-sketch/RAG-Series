import logging
import os
import time
from contextlib import contextmanager

from .env import env_str

LOG_LEVEL = env_str("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=(
        [logging.StreamHandler(), logging.FileHandler(LOG_FILE, encoding="utf-8")]
        if LOG_FILE else [logging.StreamHandler()]
    ),
)
logger = logging.getLogger("rag_series")


@contextmanager
def timed(step_name: str):
    start = time.perf_counter()
    logger.info("%s: started", step_name)
    try:
        yield
    finally:
        logger.info("%s: %.3f sec", step_name, time.perf_counter() - start)
