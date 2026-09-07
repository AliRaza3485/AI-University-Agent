import math

from app.embeddings.embedder import EmbeddedChunk, embed_chunks, embed_texts
from app.ingestion.chunker import Chunk

EMBEDDING_DIMENSION = 384


def _make_chunk(chunk_id: str, text: str, chunk_index: int = 0) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        char_count=len(text),
        source_document="test.pdf",
        page_numbers=[1, 2],
        chunk_index=chunk_index,
    )


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


# --- embed_texts: basics ----------------------------------------------------


def test_embed_texts_returns_one_vector_per_input():
    texts = ["First sentence.", "Second sentence.", "Third sentence."]
    embeddings = embed_texts(texts)

    assert len(embeddings) == len(texts)


def test_embed_texts_vectors_have_correct_dimension():
    embeddings = embed_texts(["A short sentence about admissions."])

    assert len(embeddings[0]) == EMBEDDING_DIMENSION


def test_embed_texts_returns_empty_list_for_empty_input():
    embeddings = embed_texts([])

    assert embeddings == []


# --- embed_texts: normalization ---------------------------------------------


def test_embed_texts_vectors_are_normalized():
    embeddings = embed_texts(["The university offers a computer science degree."])
    norm = math.sqrt(sum(v * v for v in embeddings[0]))

    assert math.isclose(norm, 1.0, abs_tol=1e-3)


# --- embed_texts: semantic similarity ---------------------------------------


def test_similar_sentences_have_higher_similarity_than_unrelated_ones():
    similar_a = "The university offers a computer science degree."
    similar_b = "The university has a CS program."
    unrelated = "The cafeteria serves lunch at noon."

    emb_similar_a, emb_similar_b, emb_unrelated = embed_texts(
        [similar_a, similar_b, unrelated]
    )

    similar_score = _cosine_similarity(emb_similar_a, emb_similar_b)
    unrelated_score = _cosine_similarity(emb_similar_a, emb_unrelated)

    assert similar_score > 0.5
    assert similar_score > unrelated_score


# --- embed_chunks: basics ----------------------------------------------------


def test_embed_chunks_returns_one_embedded_chunk_per_input_chunk():
    chunks = [
        _make_chunk("id-1", "Admission requirements for Fall 2026.", chunk_index=0),
        _make_chunk(
            "id-2", "Eligibility criteria for the BSCS program.", chunk_index=1
        ),
    ]
    embedded = embed_chunks(chunks)

    assert len(embedded) == len(chunks)
    assert all(isinstance(e, EmbeddedChunk) for e in embedded)


def test_embed_chunks_returns_empty_list_for_empty_input():
    embedded = embed_chunks([])

    assert embedded == []


def test_embed_chunks_vectors_have_correct_dimension():
    chunks = [_make_chunk("id-1", "Some prospectus text.")]
    embedded = embed_chunks(chunks)

    assert len(embedded[0].embedding) == EMBEDDING_DIMENSION


# --- embed_chunks: metadata preservation -------------------------------------


def test_embed_chunks_preserves_chunk_metadata():
    chunk = _make_chunk(
        "chunk-abc", "Fee structure for the Fall 2026 intake.", chunk_index=3
    )
    embedded = embed_chunks([chunk])[0]

    assert embedded.chunk_id == chunk.chunk_id
    assert embedded.text == chunk.text
    assert embedded.source_document == chunk.source_document
    assert embedded.page_numbers == chunk.page_numbers
    assert embedded.chunk_index == chunk.chunk_index


def test_embed_chunks_preserves_order_across_multiple_chunks():
    chunks = [
        _make_chunk("id-1", "First chunk text.", chunk_index=0),
        _make_chunk("id-2", "Second chunk text.", chunk_index=1),
        _make_chunk("id-3", "Third chunk text.", chunk_index=2),
    ]
    embedded = embed_chunks(chunks)

    assert [e.chunk_id for e in embedded] == [c.chunk_id for c in chunks]
    assert [e.chunk_index for e in embedded] == [c.chunk_index for c in chunks]
