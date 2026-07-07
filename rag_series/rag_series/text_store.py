import json
from functools import lru_cache
from typing import Any, Dict, List

from .config import CHUNKS_TEXT_PATH

ChunkTextStore = Dict[int, str]
CHUNK_TEXT_CACHE_SIZE = 4096


def save_chunks_text_jsonl_and_attach_offsets(
    chunks: List[str],
    metadata: List[Dict[str, Any]],
    path: str = CHUNKS_TEXT_PATH,
) -> None:
    if len(chunks) != len(metadata):
        raise RuntimeError(
            f"Chunk/text metadata mismatch before save: chunks={len(chunks)}, metadata={len(metadata)}"
        )

    with open(path, "wb") as f:
        for idx, (text, meta) in enumerate(zip(chunks, metadata), start=1):
            if int(meta["global_chunk_id"]) != idx:
                raise RuntimeError(
                    f"global_chunk_id mismatch at metadata row {idx}: {meta['global_chunk_id']}"
                )
            text_offset = f.tell()
            meta["text_offset"] = int(text_offset)
            line = json.dumps({"chunk_id": idx, "text": text}, ensure_ascii=False)
            f.write(line.encode("utf-8"))
            f.write(b"\n")


@lru_cache(maxsize=CHUNK_TEXT_CACHE_SIZE)
def load_chunk_text_cached(
    chunk_id: int,
    text_offset: int,
    path: str = CHUNKS_TEXT_PATH,
) -> str:
    chunk_id = int(chunk_id)
    text_offset = int(text_offset)
    with open(path, "rb") as f:
        f.seek(text_offset)
        line = f.readline()

    if not line.strip():
        raise RuntimeError(f"Missing chunk text in {path}: chunk_id={chunk_id}, offset={text_offset}")

    obj = json.loads(line.decode("utf-8"))
    actual_chunk_id = int(obj["chunk_id"])
    if actual_chunk_id != chunk_id:
        raise RuntimeError(
            f"Chunk text offset is stale/corrupt: expected chunk_id {chunk_id}, "
            f"got {actual_chunk_id}. Set CFG.rebuild_database = True."
        )
    return obj["text"]


def validate_metadata_text_offsets(metadata: List[Dict[str, Any]], path: str = CHUNKS_TEXT_PATH) -> None:
    import os

    if not os.path.exists(path):
        raise RuntimeError(f"Missing chunk text store: {path}")
    for meta in metadata[:5]:
        if "text_offset" not in meta:
            raise RuntimeError(
                "metadata has no text_offset. Rebuild database with CFG.rebuild_database = True."
            )
        load_chunk_text_cached(int(meta["global_chunk_id"]), int(meta["text_offset"]), path)


def load_needed_chunk_texts(
    metas: List[Dict[str, Any]],
    path: str = CHUNKS_TEXT_PATH,
) -> ChunkTextStore:
    found: ChunkTextStore = {}
    for meta in metas:
        chunk_id = int(meta["global_chunk_id"])
        if chunk_id in found:
            continue
        found[chunk_id] = load_chunk_text_cached(chunk_id, int(meta["text_offset"]), path)
    return found


def chunk_text_for_metadata(meta: Dict[str, Any], text_store: ChunkTextStore) -> str:
    return text_store[int(meta["global_chunk_id"])]
