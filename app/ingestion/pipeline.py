"""
Ingestion pipeline orchestrator.

Ties together: parse -> clean -> chunk into a single call.
Designed to handle any number of documents — pass a directory or a
single file path. Each document is processed independently and all
chunks are tagged with their source document name.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.ingestion.chunker import Chunk, chunk_pages
from app.ingestion.cleaner import clean_pages
from app.ingestion.parser import extract_pdf_pages

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf"}


def ingest_document(
    file_path: str | Path,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    """
    Run the full ingestion pipeline on a single document.

    Returns a list of Chunk objects ready for embedding.
    """
    file_path = Path(file_path)
    logger.info("Ingesting: %s", file_path.name)

    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {file_path.suffix}. "
            f"Supported: {SUPPORTED_EXTENSIONS}"
        )

    pages = extract_pdf_pages(file_path)
    cleaned = clean_pages(pages)
    chunks = chunk_pages(
        cleaned,
        source_document=file_path.name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    logger.info(
        "Ingested %s: %d pages -> %d chunks",
        file_path.name,
        len(pages),
        len(chunks),
    )
    return chunks


def ingest_directory(
    directory: str | Path,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    """
    Ingest all supported documents in a directory.

    Processes each file independently, returns all chunks combined.
    Files that fail are logged and skipped — one bad file does not
    block the rest.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    all_chunks: list[Chunk] = []
    files = sorted(
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        logger.warning("No supported files found in %s", directory)
        return []

    logger.info("Found %d documents in %s", len(files), directory)

    for file_path in files:
        try:
            chunks = ingest_document(file_path, chunk_size, chunk_overlap)
            all_chunks.extend(chunks)
        except Exception:
            logger.exception("Failed to ingest %s, skipping", file_path.name)

    logger.info(
        "Directory ingestion complete: %d files -> %d total chunks",
        len(files),
        len(all_chunks),
    )
    return all_chunks
