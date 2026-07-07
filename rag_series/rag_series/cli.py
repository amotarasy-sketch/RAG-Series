from __future__ import annotations

from typing import Any, Dict, List, Tuple

import typer

from .config import CFG
from .exceptions import IndexConsistencyError, RuntimeLoadError
from .logging_utils import logger

app = typer.Typer(
    help="Local RAG assistant for FB2 book series.",
    no_args_is_help=True,
)

Runtime = Tuple[List[Dict[str, Any]], Any, str, Dict[Tuple[str, int], int]]


def load_runtime(rebuild: bool = False) -> Runtime:
    """Initialize cache, load indexes and build neighbor lookup."""
    from .context import build_index_by_book_chunk
    from .database import load_database
    from .llm_cache import init_llm_cache

    init_llm_cache()

    try:
        _, metadata, faiss_index, lexical_db_path = load_database(
            rebuild=rebuild,
            load_embeddings_for_debug=CFG.load_embeddings_for_debug,
        )
    except Exception as exc:
        raise RuntimeLoadError(str(exc)) from exc

    if faiss_index.ntotal != len(metadata):
        raise IndexConsistencyError(
            f"FAISS index size mismatch: index has {faiss_index.ntotal}, "
            f"metadata has {len(metadata)}. Set REBUILD_DATABASE=true or run build --rebuild."
        )

    for i, meta in enumerate(metadata, start=1):
        if int(meta["global_chunk_id"]) != i:
            raise IndexConsistencyError(f"Metadata global_chunk_id mismatch at row {i}")
        if "text_offset" not in meta or meta["text_offset"] is None:
            raise IndexConsistencyError("Metadata missing text_offset. Run build --rebuild.")

    logger.info("SQLite FTS5 lexical index ready.")
    by_book_chunk = build_index_by_book_chunk(metadata)
    logger.info("Neighbor lookup index ready: %d chunks.", len(by_book_chunk))

    return metadata, faiss_index, lexical_db_path, by_book_chunk


@app.command()
def build(
    rebuild: bool = typer.Option(
        False,
        "--rebuild",
        help="Force rebuild even if the cached database manifest is valid.",
    ),
) -> None:
    """Build indexes, metadata and caches for the configured FB2 directory."""
    load_runtime(rebuild=rebuild or CFG.rebuild_database)
    typer.echo("Database is ready.")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask the RAG assistant."),
    debug: bool = typer.Option(False, "--debug", help="Print retrieved context fragments."),
) -> None:
    """Ask one question and print one answer."""
    from .debug import print_used_fragments
    from .pipeline import answer_query

    metadata, faiss_index, lexical_db_path, by_book_chunk = load_runtime(rebuild=False)

    answer, used = answer_query(
        question,
        metadata,
        faiss_index,
        lexical_db_path,
        by_book_chunk,
    )

    typer.echo("\n=== ANSWER ===\n")
    typer.echo(answer)

    if debug:
        print_used_fragments(used)


@app.command()
def chat(
    debug: bool = typer.Option(False, "--debug", help="Print retrieved context fragments."),
) -> None:
    """Start an interactive question-answering session."""
    from .debug import print_used_fragments
    from .pipeline import answer_query

    metadata, faiss_index, lexical_db_path, by_book_chunk = load_runtime(rebuild=False)

    typer.echo("\nReady. Type 'exit' to quit.")

    while True:
        query = typer.prompt("\nQuestion").strip()
        if query.lower() in {"exit", "quit", "выход"}:
            break
        if not query:
            continue

        try:
            answer, used = answer_query(
                query,
                metadata,
                faiss_index,
                lexical_db_path,
                by_book_chunk,
            )
        except Exception as exc:  # keep the chat alive after one bad query
            logger.exception("Failed to process query")
            typer.echo(f"Error: {exc}", err=True)
            continue

        typer.echo("\n=== ANSWER ===\n")
        typer.echo(answer)

        if debug:
            print_used_fragments(used)


@app.command()
def info() -> None:
    """Show the effective configuration and storage paths."""
    typer.echo(f"Books dir: {CFG.books_dir}")
    typer.echo(f"Chunks dir: {CFG.chunks_dir}")
    typer.echo(f"Embedding model: {CFG.embedding_model_name}")
    typer.echo(f"Reranker model: {CFG.reranker_model_name}")
    typer.echo(f"LLM base URL: {CFG.llm_base_url}")
    typer.echo(f"LLM model: {CFG.llm_model}")
    typer.echo(f"Chunk tokens: {CFG.chunk_tokens}")
    typer.echo(f"Overlap tokens: {CFG.overlap_tokens}")
    typer.echo(f"Rerank top k: {CFG.rerank_top_k}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
