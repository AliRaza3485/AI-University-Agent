"""
Vector storage using Qdrant.

Stores EmbeddedChunk objects (from app.embeddings.embedder) in a Qdrant
collection and provides similarity search over them. Qdrant runs as a
separate service (see docker-compose.yml) - this module only talks to
it over its client API, it never touches disk directly.
"""

from __future__ import annotations

import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from app.config.settings import settings
from app.embeddings.embedder import EmbeddedChunk

logger = logging.getLogger(__name__)

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        if settings.qdrant_url:
            logger.info("Connecting to Qdrant Cloud at %s", settings.qdrant_url)
            _client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )
        else:
            logger.info(
                "Connecting to local Qdrant at %s:%d",
                settings.qdrant_host,
                settings.qdrant_port,
            )
            _client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    return _client


def ensure_collection(
    collection_name: str = settings.qdrant_collection,
    vector_size: int = settings.embedding_dimension,
) -> None:
    """
    Create the collection if it does not already exist. Safe to call
    on every startup - does nothing if the collection is already there.
    """
    client = get_client()
    existing = {c.name for c in client.get_collections().collections}

    if collection_name in existing:
        logger.info("Collection '%s' already exists", collection_name)
        return

    logger.info("Creating collection '%s' (size=%d)", collection_name, vector_size)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def _chunk_id_to_point_id(chunk_id: str) -> str:
    """
    Qdrant point IDs must be an unsigned int or a UUID. Our chunk_ids are
    already UUID hex strings (from uuid.uuid4().hex in chunker.py), so we
    just reformat them into canonical UUID form.
    """
    return str(uuid.UUID(chunk_id))


def upsert_embedded_chunks(
    embedded_chunks: list[EmbeddedChunk],
    collection_name: str = settings.qdrant_collection,
) -> int:
    """
    Save (or update) a list of EmbeddedChunk objects into the collection.

    Returns the number of points upserted. Calling this again with the
    same chunk_ids overwrites the existing points - upsert is idempotent,
    which makes re-running ingestion on the same document safe.
    """
    if not embedded_chunks:
        return 0

    ensure_collection(collection_name)
    client = get_client()

    points = [
        PointStruct(
            id=_chunk_id_to_point_id(chunk.chunk_id),
            vector=chunk.embedding,
            payload={
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source_document": chunk.source_document,
                "page_numbers": chunk.page_numbers,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata,
            },
        )
        for chunk in embedded_chunks
    ]

    client.upsert(collection_name=collection_name, points=points)
    logger.info("Upserted %d points into '%s'", len(points), collection_name)
    return len(points)


def search(
    query_embedding: list[float],
    top_k: int = 5,
    collection_name: str = settings.qdrant_collection,
) -> list[dict]:
    """
    Find the top_k chunks most similar to a query embedding.

    Returns a list of dicts with the original chunk payload plus a
    "score" field (cosine similarity, higher = more similar).
    """
    client = get_client()

    results = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=top_k,
    ).points

    return [
        {
            **point.payload,
            "score": point.score,
        }
        for point in results
    ]
