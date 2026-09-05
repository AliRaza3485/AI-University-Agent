from app.ingestion.chunker import Chunk, chunk_pages, summarize_chunks


# --- chunk_pages basics -------------------------------------------------------


def test_chunk_pages_returns_list_of_chunks():
    pages = [
        {"page_number": 1, "cleaned_text": "Hello world.", "cleaned_char_count": 12},
        {"page_number": 2, "cleaned_text": "Second page.", "cleaned_char_count": 12},
    ]
    chunks = chunk_pages(pages, source_document="test.pdf")

    assert isinstance(chunks, list)
    assert all(isinstance(c, Chunk) for c in chunks)
    assert len(chunks) >= 1


def test_chunk_pages_tags_source_document():
    pages = [{"page_number": 1, "cleaned_text": "Some text.", "cleaned_char_count": 10}]
    chunks = chunk_pages(pages, source_document="prospectus.pdf")

    for chunk in chunks:
        assert chunk.source_document == "prospectus.pdf"


def test_chunk_pages_tracks_page_numbers():
    pages = [
        {"page_number": 5, "cleaned_text": "Page five content.", "cleaned_char_count": 18},
        {"page_number": 6, "cleaned_text": "Page six content.", "cleaned_char_count": 17},
    ]
    chunks = chunk_pages(pages, source_document="test.pdf")

    all_pages = set()
    for chunk in chunks:
        all_pages.update(chunk.page_numbers)

    assert 5 in all_pages
    assert 6 in all_pages


def test_chunk_pages_skips_empty_pages():
    pages = [
        {"page_number": 1, "cleaned_text": "Content here.", "cleaned_char_count": 13},
        {"page_number": 2, "cleaned_text": "", "cleaned_char_count": 0},
        {"page_number": 3, "cleaned_text": "More content.", "cleaned_char_count": 13},
    ]
    chunks = chunk_pages(pages, source_document="test.pdf")

    all_pages = set()
    for chunk in chunks:
        all_pages.update(chunk.page_numbers)

    assert 2 not in all_pages


def test_chunk_pages_returns_empty_for_all_empty_pages():
    pages = [
        {"page_number": 1, "cleaned_text": "", "cleaned_char_count": 0},
        {"page_number": 2, "cleaned_text": "   ", "cleaned_char_count": 3},
    ]
    chunks = chunk_pages(pages, source_document="test.pdf")

    assert chunks == []


# --- size constraints ---------------------------------------------------------


def test_chunks_respect_max_size():
    long_text = "This is a sentence. " * 200
    pages = [{"page_number": 1, "cleaned_text": long_text, "cleaned_char_count": len(long_text)}]
    chunks = chunk_pages(pages, source_document="test.pdf", chunk_size=500, chunk_overlap=50)

    for chunk in chunks:
        assert chunk.char_count <= 550, (
            f"Chunk {chunk.chunk_index} has {chunk.char_count} chars, expected <=550"
        )


def test_small_pages_merge_into_single_chunk():
    pages = [
        {"page_number": i, "cleaned_text": f"Short line {i}.", "cleaned_char_count": 14}
        for i in range(1, 6)
    ]
    chunks = chunk_pages(pages, source_document="test.pdf", chunk_size=800)

    assert len(chunks) == 1


# --- chunk_index ordering -----------------------------------------------------


def test_chunk_indices_are_sequential():
    long_text = "Paragraph one.\n\nParagraph two.\n\n" * 50
    pages = [{"page_number": 1, "cleaned_text": long_text, "cleaned_char_count": len(long_text)}]
    chunks = chunk_pages(pages, source_document="test.pdf", chunk_size=200, chunk_overlap=0)

    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


# --- unique IDs ---------------------------------------------------------------


def test_chunk_ids_are_unique():
    pages = [
        {"page_number": 1, "cleaned_text": "A. " * 100, "cleaned_char_count": 300},
        {"page_number": 2, "cleaned_text": "B. " * 100, "cleaned_char_count": 300},
    ]
    chunks = chunk_pages(pages, source_document="test.pdf", chunk_size=100, chunk_overlap=0)

    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


# --- summarize_chunks --------------------------------------------------------


def test_summarize_chunks_returns_stats():
    pages = [
        {"page_number": 1, "cleaned_text": "Hello world content.", "cleaned_char_count": 20},
        {"page_number": 2, "cleaned_text": "More content here.", "cleaned_char_count": 18},
    ]
    chunks = chunk_pages(pages, source_document="test.pdf")
    summary = summarize_chunks(chunks)

    assert summary["total_chunks"] >= 1
    assert "avg_chars" in summary
    assert "min_chars" in summary
    assert "max_chars" in summary
    assert "test.pdf" in summary["source_documents"]


def test_summarize_chunks_handles_empty():
    summary = summarize_chunks([])
    assert summary == {"total_chunks": 0}
