"""
Text chunking for the ingestion pipeline.

Takes cleaned page text and splits it into retrieval-sized chunks with
metadata (source document, page numbers, position). Designed to handle
any number of documents — the caller passes a document identifier and
the chunker tags every chunk with it.

Strategy: paragraph-aware recursive splitting.
  1. Split on double newlines (paragraph boundaries) first.
  2. If a paragraph is still too large, split on single newlines.
  3. If a single line is still too large, split on sentence boundaries.
  4. Merge small adjacent pieces up to chunk_size, keeping overlap.

This preserves semantic boundaries as much as possible while
guaranteeing every chunk stays within the size limit.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150


@dataclass
class Chunk:
    chunk_id: str
    text: str
    char_count: int
    source_document: str
    page_numbers: list[int]
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def _split_text(text: str, separators: list[str], chunk_size: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    if not separators:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep = separators[0]
    remaining_seps = separators[1:]

    if sep == "<sentence>":
        parts = _SENTENCE_BOUNDARY.split(text)
    else:
        parts = text.split(sep)

    pieces: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) <= chunk_size:
            pieces.append(part)
        else:
            pieces.extend(_split_text(part, remaining_seps, chunk_size))

    return pieces


def _merge_pieces_with_overlap(
    pieces: list[str],
    chunk_size: int,
    overlap: int,
) -> list[str]:
    if not pieces:
        return []

    chunks: list[str] = []
    current = pieces[0]

    for piece in pieces[1:]:
        candidate = current + "\n\n" + piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            chunks.append(current)
            if overlap > 0 and len(current) > overlap:
                tail = current[-overlap:]
                boundary = tail.find("\n")
                if boundary != -1:
                    tail = tail[boundary + 1 :]
                current = tail + "\n\n" + piece if tail.strip() else piece
            else:
                current = piece

    if current.strip():
        chunks.append(current)

    return chunks


def chunk_pages(
    pages: list[dict],
    source_document: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """
    Chunk a list of cleaned page dicts into retrieval-sized pieces.

    Args:
        pages: Output of clean_pages() — each dict must have "cleaned_text"
               and "page_number".
        source_document: Identifier for the source (e.g. filename).
        chunk_size: Target maximum characters per chunk.
        chunk_overlap: Characters of overlap between consecutive chunks.

    Returns:
        List of Chunk objects, ordered by position in the document.
    """
    page_pieces: list[tuple[str, int]] = []
    for page in pages:
        text = page.get("cleaned_text", page.get("text", "")).strip()
        if not text:
            continue
        pieces = _split_text(text, ["\n\n", "\n", "<sentence>"], chunk_size)
        for piece in pieces:
            page_pieces.append((piece, page["page_number"]))

    if not page_pieces:
        return []

    merged_texts: list[str] = []
    merged_pages: list[list[int]] = []

    current_text = page_pieces[0][0]
    current_pages = {page_pieces[0][1]}

    for piece_text, page_num in page_pieces[1:]:
        candidate = current_text + "\n\n" + piece_text
        if len(candidate) <= chunk_size:
            current_text = candidate
            current_pages.add(page_num)
        else:
            merged_texts.append(current_text)
            merged_pages.append(sorted(current_pages))

            if chunk_overlap > 0 and len(current_text) > chunk_overlap:
                tail = current_text[-chunk_overlap:]
                boundary = tail.find("\n")
                if boundary != -1:
                    tail = tail[boundary + 1 :]
                if tail.strip():
                    current_text = tail + "\n\n" + piece_text
                    current_pages = {max(current_pages), page_num}
                else:
                    current_text = piece_text
                    current_pages = {page_num}
            else:
                current_text = piece_text
                current_pages = {page_num}

    if current_text.strip():
        merged_texts.append(current_text)
        merged_pages.append(sorted(current_pages))

    chunks: list[Chunk] = []
    for idx, (text, page_nums) in enumerate(zip(merged_texts, merged_pages)):
        chunks.append(
            Chunk(
                chunk_id=uuid.uuid4().hex,
                text=text,
                char_count=len(text),
                source_document=source_document,
                page_numbers=page_nums,
                chunk_index=idx,
            )
        )

    return chunks


def summarize_chunks(chunks: list[Chunk]) -> dict:
    """Quick diagnostic summary for development — not part of the production pipeline."""
    if not chunks:
        return {"total_chunks": 0}

    char_counts = [c.char_count for c in chunks]
    return {
        "total_chunks": len(chunks),
        "avg_chars": round(sum(char_counts) / len(char_counts), 1),
        "min_chars": min(char_counts),
        "max_chars": max(char_counts),
        "source_documents": list({c.source_document for c in chunks}),
        "page_range": [
            min(p for c in chunks for p in c.page_numbers),
            max(p for c in chunks for p in c.page_numbers),
        ],
    }
