from pathlib import Path

from app.ingestion.cleaner import (
    clean_page_text,
    clean_pages,
    normalize_whitespace,
    strip_running_header,
)
from app.ingestion.parser import extract_pdf_pages

PDF_PATH = Path("data/raw_documents/Prospectus - FALL 2026 (29-07-2026).pdf")


# --- strip_running_header -----------------------------------------------


def test_strip_running_header_removes_header_when_present():
    text = "UNIVERSITY OF EDUCATION, LAHORE\nFALL 2026 PROSPECTUS\nActual page content here."
    result = strip_running_header(text)

    assert "UNIVERSITY OF EDUCATION" not in result
    assert result.strip() == "Actual page content here."


def test_strip_running_header_leaves_text_unchanged_when_absent():
    text = "Mr. Muhammad Tehseen\nLecturer\nDepartment of Computer Science"
    result = strip_running_header(text)

    assert result == text


def test_strip_running_header_requires_both_lines_to_match():
    # Line 1 matches but line 2 does not -> should NOT strip.
    text = "UNIVERSITY OF EDUCATION, LAHORE\nSome unrelated line\nContent."
    result = strip_running_header(text)

    assert result == text


# --- normalize_whitespace -------------------------------------------------


def test_normalize_whitespace_collapses_multiple_blank_lines():
    text = "Line 1\n\n\n\nLine 2"
    result = normalize_whitespace(text)

    assert result == "Line 1\n\nLine 2"


def test_normalize_whitespace_trims_trailing_spaces_and_edges():
    text = "  \nLine with trailing spaces   \n  \n"
    result = normalize_whitespace(text)

    assert result == "Line with trailing spaces"


# --- clean_page_text (combined) -------------------------------------------


def test_clean_page_text_applies_both_steps():
    text = "UNIVERSITY OF EDUCATION, LAHORE\nFALL 2026\n\n\n\nBody text here.   "
    result = clean_page_text(text)

    assert result == "Body text here."


# --- clean_pages (integration with real PDF) -------------------------------


def test_clean_pages_preserves_raw_text_and_adds_cleaned_fields():
    pages = extract_pdf_pages(PDF_PATH)
    cleaned = clean_pages(pages)

    assert len(cleaned) == len(pages)
    for original, result in zip(pages, cleaned):
        assert result["text"] == original["text"]  # raw text untouched
        assert "cleaned_text" in result
        assert "cleaned_char_count" in result
        assert result["cleaned_char_count"] == len(result["cleaned_text"])


def test_clean_pages_removes_header_from_pages_that_had_it():
    pages = extract_pdf_pages(PDF_PATH)
    cleaned = clean_pages(pages)

    # Page 13 (index 12) was confirmed in the exploration notebook to
    # start with the running header.
    header_page = cleaned[12]
    assert "UNIVERSITY OF EDUCATION, LAHORE" not in header_page["cleaned_text"]


def test_clean_pages_does_not_alter_pages_without_header():
    pages = extract_pdf_pages(PDF_PATH)
    cleaned = clean_pages(pages)

    # Page 53 (index 52) was confirmed NOT to have the header.
    no_header_page = cleaned[52]
    # cleaned_char_count should be close to original char_count -
    # only whitespace normalization should have changed it, not content
    # removal.
    assert no_header_page["cleaned_char_count"] <= no_header_page["char_count"]
    assert no_header_page["cleaned_char_count"] > no_header_page["char_count"] * 0.9
