"""
Embedding generation using Sentence Transformers.

Converts text chunks into dense vector embeddings for semantic search.
The model runs locally — no API key, no cost, no network dependency
after the initial download.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

from app.config.settings import settings

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", settings.embedding_model)
        _model = SentenceTransformer(settings.embedding_model)
        logger.info("Model loaded (dimension=%d)", _model.get_sentence_embedding_dimension())
    return _model


@dataclass
class EmbeddedChunk:
    chunk_id: str
    text: str
    embedding: list[float]
    source_document: str
    page_numbers: list[int]
    chunk_index: int
    metadata: dict


def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.

    Args:
        texts: List of strings to embed.
        batch_size: Number of texts to process in each batch.

    Returns:
        List of embedding vectors (each a list of floats).
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 100,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


def embed_chunks(chunks: list, batch_size: int = 64) -> list[EmbeddedChunk]:
    """
    Generate embeddings for a list of Chunk objects.

    Args:
        chunks: List of Chunk objects (from chunker.py).
        batch_size: Batch size for the embedding model.

    Returns:
        List of EmbeddedChunk objects with embedding vectors attached.
    """
    if not chunks:
        return []

    texts = [c.text for c in chunks]
    logger.info("Embedding %d chunks (batch_size=%d)...", len(texts), batch_size)

    embeddings = embed_texts(texts, batch_size=batch_size)

    embedded: list[EmbeddedChunk] = []
    for chunk, embedding in zip(chunks, embeddings):
        embedded.append(
            EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                embedding=embedding,
                source_document=chunk.source_document,
                page_numbers=chunk.page_numbers,
                chunk_index=chunk.chunk_index,
                metadata=chunk.metadata,
            )
        )

    logger.info("Embedding complete: %d chunks -> %d vectors", len(chunks), len(embedded))
    return embedded
