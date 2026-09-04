from pathlib import Path

import pytest

from app.ingestion.parser import extract_pdf_pages, summarize_extraction

PDF_PATH = Path("data/raw_documents/Prospectus - FALL 2026 (29-07-2026).pdf")


def test_extract_pdf_pages_returns_list_of_pages():
    pages = extract_pdf_pages(PDF_PATH)

    assert isinstance(pages, list)
    assert len(pages) > 0

    first_page = pages[0]
    assert "page_number" in first_page
    assert "text" in first_page
    assert "char_count" in first_page
    assert first_page["page_number"] == 1


def test_page_numbers_are_sequential():
    pages = extract_pdf_pages(PDF_PATH)

    page_numbers = [p["page_number"] for p in pages]
    assert page_numbers == list(range(1, len(pages) + 1))


def test_char_count_matches_text_length():
    pages = extract_pdf_pages(PDF_PATH)

    for page in pages:
        assert page["char_count"] == len(page["text"])


def test_missing_file_raises_file_not_found_error():
    with pytest.raises(FileNotFoundError):
        extract_pdf_pages("data/raw_documents/does_not_exist.pdf")


def test_non_pdf_file_raises_value_error(tmp_path):
    fake_file = tmp_path / "not_a_pdf.txt"
    fake_file.write_text("hello")

    with pytest.raises(ValueError):
        extract_pdf_pages(fake_file)


def test_summarize_extraction_reports_useful_stats():
    pages = extract_pdf_pages(PDF_PATH)
    summary = summarize_extraction(pages)

    assert summary["total_pages"] == len(pages)
    assert "empty_pages" in summary
    assert "low_text_pages" in summary
    assert "avg_chars_per_page" in summary

    # Print for manual inspection when running with `-s`
    print("\n--- Extraction Summary ---")
    print(summary)
