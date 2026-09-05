from pathlib import Path

from app.ingestion.cleaner import (
    clean_page_text,
    clean_pages,
    normalize_whitespace,
    strip_header_footer_blocks,
)
from app.ingestion.parser import extract_pdf_pages

PDF_PATH = Path("data/raw_documents/Prospectus - FALL 2026 (29-07-2026).pdf")


# --- strip_header_footer_blocks --------------------------------------------


def test_strip_header_footer_removes_block_at_start():
    text = "UNIVERSITY OF EDUCATION, LAHORE\nFALL 2026\nActual page content here."
    result = strip_header_footer_blocks(text)

    assert "UNIVERSITY OF EDUCATION" not in result
    assert "Actual page content here." in result


def test_strip_header_footer_removes_block_in_middle_of_text():
    # This is the real-world pattern found in the actual PDF: the block
    # appears embedded inside body text, not just at the top.
    text = (
        "Some paragraph text ends here. More\n"
        "UNIVERSITY OF EDUCATION, LAHORE\n"
        "FALL\n"
        "107\n"
        "Next paragraph continues here."
    )
    result = strip_header_footer_blocks(text)

    assert "UNIVERSITY OF EDUCATION" not in result
    assert "Some paragraph text ends here. More" in result
    assert "Next paragraph continues here." in result
    assert "107" not in result


def test_strip_header_footer_removes_back_to_back_duplicate_block():
    # Matches the observed pattern where the block repeats twice in a row
    # before a trailing page number.
    text = (
        "...competence, commitment, and integrity. More\n"
        "UNIVERSITY OF EDUCATION, LAHORE\n"
        "FALL\n"
        "UNIVERSITY OF EDUCATION, LAHORE\n"
        "FALL\n"
        "107\n"
        "Next section starts here."
    )
    result = strip_header_footer_blocks(text)

    assert result.count("UNIVERSITY OF EDUCATION") == 0
    assert "Next section starts here." in result


def test_strip_header_footer_leaves_text_unchanged_when_absent():
    text = "Mr. Muhammad Tehseen\nLecturer\nDepartment of Computer Science"
    result = strip_header_footer_blocks(text)

    assert result == text


def test_strip_header_footer_does_not_touch_unrelated_numbers():
    text = "The program requires 107 credit hours to graduate."
    result = strip_header_footer_blocks(text)

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


def test_clean_pages_removes_header_footer_from_entire_document():
    pages = extract_pdf_pages(PDF_PATH)
    cleaned = clean_pages(pages)

    still_has_phrase = [
        p["page_number"]
        for p in cleaned
        if "UNIVERSITY OF EDUCATION, LAHORE" in p["cleaned_text"]
    ]

    assert still_has_phrase == [], (
        f"Expected 0 pages with leftover header/footer text, "
        f"found {len(still_has_phrase)}: {still_has_phrase[:20]}"
    )
