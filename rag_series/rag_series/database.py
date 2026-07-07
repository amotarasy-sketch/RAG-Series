import json
import os
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from .chunking import chunk_section_by_tokens
from .config import CFG, CHUNKS_TEXT_PATH, EMB_PATH, FAISS_INDEX_PATH, LEXICAL_INDEX_PATH, MANIFEST_PATH, META_PATH
from .fb2_parser import parse_fb2_sections
from .indexes import build_faiss_index, build_lexical_index_from_chunks, load_faiss_index, save_faiss_index
from .logging_utils import logger
from .models import get_embedding_model
from .text_store import save_chunks_text_jsonl_and_attach_offsets, validate_metadata_text_offsets
from .text_utils import count_tokens, file_sha256, safe_filename


def current_books_manifest() -> Dict[str, Any]:
    files = [f for f in os.listdir(CFG.books_dir) if f.lower().endswith(".fb2")]
    files.sort()
    items = []
    for f in files:
        path = os.path.join(CFG.books_dir, f)
        items.append({"file": f, "sha256": file_sha256(path), "size": os.path.getsize(path)})

    return {
        "books_dir": CFG.books_dir,
        "chunk_tokens": CFG.chunk_tokens,
        "overlap_tokens": CFG.overlap_tokens,
        "min_chunk_tokens": CFG.min_chunk_tokens,
        "embedding_model": CFG.embedding_model_name,
        "lexical_index": "sqlite_fts5_pymorphy3_lemmas_v2_no_bm25_tokens_file",
        "chunk_text_store": "chunks_text_jsonl_offsets_in_metadata_v3",
        "parser": "lxml_fb2_v2",
        "faiss": {
            "type": "IndexHNSWFlat",
            "metric": "IP_normalized",
            "m": CFG.hnsw_m,
            "ef_construction": CFG.hnsw_ef_construction,
        },
        "items": items,
    }


def load_manifest() -> Optional[Dict[str, Any]]:
    if not os.path.exists(MANIFEST_PATH):
        return None
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def database_cache_exists() -> bool:
    return all(os.path.exists(p) for p in [
        EMB_PATH,
        META_PATH,
        CHUNKS_TEXT_PATH,
        LEXICAL_INDEX_PATH,
        FAISS_INDEX_PATH,
        MANIFEST_PATH,
    ])


def is_cache_valid() -> bool:
    if not database_cache_exists():
        return False
    return load_manifest() == current_books_manifest()


def build_database() -> Tuple[Optional[np.ndarray], List[Dict[str, Any]], faiss.Index, str]:
    manifest = current_books_manifest()
    book_files = [item["file"] for item in manifest["items"]]

    logger.info("Found FB2 books: %d", len(book_files))
    if not book_files:
        raise RuntimeError("No .fb2 files found. Check CFG.books_dir.")

    all_chunks: List[str] = []
    metadata: List[Dict[str, Any]] = []
    global_chunk_id = 0

    for book_order, book_file in enumerate(book_files, start=1):
        book_path = os.path.join(CFG.books_dir, book_file)
        logger.info("Reading book %d/%d: %s", book_order, len(book_files), book_file)

        try:
            book_info, sections = parse_fb2_sections(book_path)
        except Exception:
            logger.exception("Failed to parse %s", book_file)
            continue

        logger.info("Book title: %s", book_info.get("title"))
        logger.info("Sections: %d", len(sections))

        book_chunks_for_debug: List[Dict[str, Any]] = []
        book_chunk_number = 0

        for section in sections:
            section_chunks = chunk_section_by_tokens(section["text"])
            for local_idx, chunk in enumerate(section_chunks, start=1):
                global_chunk_id += 1
                book_chunk_number += 1
                token_count = count_tokens(chunk)
                all_chunks.append(chunk)

                meta = {
                    "global_chunk_id": global_chunk_id,
                    "text_offset": None,
                    "book_order": book_order,
                    "book_file": book_file,
                    "book_title": book_info.get("title"),
                    "authors": book_info.get("authors") or [],
                    "sequence_name": book_info.get("sequence_name"),
                    "sequence_number": book_info.get("sequence_number"),
                    "book_sha256": book_info.get("sha256"),
                    "book_chunk_number": book_chunk_number,
                    "section_index": section["section_index"],
                    "section_chunk_number": local_idx,
                    "chapter": section["chapter"],
                    "token_count": token_count,
                }
                metadata.append(meta)

                if CFG.write_debug_chunks:
                    debug_meta = dict(meta)
                    debug_meta["text"] = chunk
                    book_chunks_for_debug.append(debug_meta)

        logger.info("Chunks in book: %d", book_chunk_number)

        if CFG.write_debug_chunks:
            debug_path = os.path.join(CFG.chunks_dir, safe_filename(book_file) + "_chunks.txt")
            with open(debug_path, "w", encoding="utf-8") as f:
                for m in book_chunks_for_debug:
                    f.write(f"=== CHUNK {m['book_chunk_number']} ===\n")
                    f.write(f"Book: {m['book_title']}\n")
                    f.write(f"Chapter: {m['chapter']}\n")
                    f.write(f"Tokens: {m['token_count']}\n")
                    f.write(m["text"].replace("\n", " "))
                    f.write("\n\n")

    if not all_chunks:
        raise RuntimeError("No chunks created. Check FB2 files and parser.")

    logger.info("Total chunks: %d", len(all_chunks))
    logger.info("Creating embeddings...")
    embeddings = get_embedding_model().encode(
        all_chunks,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=CFG.embedding_batch_size,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    logger.info("Building FAISS index...")
    faiss_index = build_faiss_index(embeddings)

    logger.info("Building SQLite FTS5 lexical index...")
    build_lexical_index_from_chunks(all_chunks)

    logger.info("Saving chunk text store and metadata offsets...")
    save_chunks_text_jsonl_and_attach_offsets(all_chunks, metadata)

    logger.info("Saving database...")
    np.save(EMB_PATH, embeddings)
    save_faiss_index(faiss_index)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info("Database ready.")
    return embeddings, metadata, faiss_index, LEXICAL_INDEX_PATH


def load_database(
    rebuild: bool = False,
    load_embeddings_for_debug: bool = False,
) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]], faiss.Index, str]:
    if not rebuild and is_cache_valid():
        logger.info("Loading cached database...")
        embeddings: Optional[np.ndarray] = None
        if load_embeddings_for_debug:
            logger.info("Loading raw embeddings for debug...")
            embeddings = np.load(EMB_PATH).astype(np.float32)

        with open(META_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        validate_metadata_text_offsets(metadata)
        faiss_index = load_faiss_index()
        return embeddings, metadata, faiss_index, LEXICAL_INDEX_PATH

    if not rebuild and database_cache_exists():
        logger.info("Cache exists but manifest changed. Rebuilding database...")

    return build_database()
