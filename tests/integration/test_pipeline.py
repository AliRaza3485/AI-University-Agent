from pathlib import Path

import pytest

from app.ingestion.chunker import summarize_chunks
from app.ingestion.pipeline import ingest_directory, ingest_document

PDF_PATH = Path("data/raw_documents/Prospectus - FALL 2026 (29-07-2026).pdf")
DATA_DIR = Path("data/raw_documents")


@pytest.fixture
def require_pdf():
    if not PDF_PATH.exists():
        pytest.skip(f"Test PDF not found at {PDF_PATH}")


# --- ingest_document ---------------------------------------------------------


def test_ingest_document_end_to_end(require_pdf):
    chunks = ingest_document(PDF_PATH)

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.text.strip() != ""
        assert chunk.source_document == PDF_PATH.name
        assert len(chunk.page_numbers) >= 1
        assert chunk.char_count == len(chunk.text)


def test_ingest_document_no_leftover_headers(require_pdf):
    chunks = ingest_document(PDF_PATH)

    leftover = [
        c.chunk_index
        for c in chunks
        if "UNIVERSITY OF EDUCATION, LAHORE" in c.text
    ]
    assert leftover == [], (
        f"Header/footer text found in {len(leftover)} chunks: {leftover[:10]}"
    )


def test_ingest_document_chunk_sizes(require_pdf):
    chunk_size = 800
    chunks = ingest_document(PDF_PATH, chunk_size=chunk_size, chunk_overlap=150)

    oversized = [c for c in chunks if c.char_count > chunk_size + 200]
    assert oversized == [], (
        f"{len(oversized)} chunks exceed size limit: "
        f"{[(c.chunk_index, c.char_count) for c in oversized[:5]]}"
    )


def test_ingest_document_prints_summary(require_pdf, capsys):
    chunks = ingest_document(PDF_PATH)
    summary = summarize_chunks(chunks)

    print("\n--- Chunk Summary ---")
    for key, val in summary.items():
        print(f"  {key}: {val}")

    assert summary["total_chunks"] > 50


# --- ingest_directory ---------------------------------------------------------


def test_ingest_directory_processes_all_files(require_pdf):
    chunks = ingest_directory(DATA_DIR)

    assert len(chunks) > 0
    sources = {c.source_document for c in chunks}
    assert PDF_PATH.name in sources


def test_ingest_directory_raises_for_missing_dir():
    with pytest.raises(FileNotFoundError):
        ingest_directory("data/does_not_exist")


def test_ingest_unsupported_file_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        ingest_document("data/raw_documents/.gitkeep")
