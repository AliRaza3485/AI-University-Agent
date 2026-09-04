"""
PDF parsing utilities for the ingestion pipeline.

Responsible ONLY for extracting raw, page-wise text from a PDF file.
No cleaning, no chunking, no metadata enrichment happens here -
those are separate stages (cleaner.py, chunker.py) by design, so each
stage can be tested and reasoned about independently.
"""

from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def extract_pdf_pages(file_path: str | Path) -> list[dict]:
    """
    Extract text page-by-page from a PDF.

    Args:
        file_path: Path to the PDF file.

    Returns:
        A list of dicts, one per page, in page order:
            [
                {
                    "page_number": 1,       # 1-indexed
                    "text": "...",          # raw extracted text (may be empty)
                    "char_count": 123,      # len(text), useful for spotting
                                             # image-only / blank pages
                },
                ...
            ]

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a .pdf, or if it cannot be opened
            (corrupted / encrypted / not actually a PDF despite extension).
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if file_path.suffix.lower() != ".pdf":
        raise ValueError(f"Only PDF files are supported, got: {file_path.suffix}")

    pages: list[dict] = []

    try:
        with fitz.open(file_path) as document:
            if document.is_encrypted:
                raise ValueError(
                    f"PDF is encrypted and cannot be read without a password: {file_path}"
                )

            total_pages = document.page_count
            logger.info("Opened PDF '%s' with %d pages", file_path.name, total_pages)

            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text").strip()

                pages.append(
                    {
                        "page_number": page_number,
                        "text": text,
                        "char_count": len(text),
                    }
                )

    except fitz.FileDataError as exc:
        # Raised by PyMuPDF when the file is corrupted or not a valid PDF.
        raise ValueError(f"Could not open PDF (corrupted or invalid): {file_path}") from exc

    return pages


def summarize_extraction(pages: list[dict]) -> dict:
    """
    Produce a quick diagnostic summary of an extraction result.

    This is NOT part of the production pipeline - it exists purely to help
    us sanity-check a new document during development. Specifically it
    tells us how many pages came back with little/no text, which is the
    strongest signal that a document (or some of its pages) is scanned
    images and will need OCR rather than direct text extraction.

    Args:
        pages: Output of extract_pdf_pages().

    Returns:
        {
            "total_pages": int,
            "empty_pages": int,          # char_count == 0
            "low_text_pages": int,       # 0 < char_count < 20 (likely noise)
            "empty_page_numbers": list[int],
            "avg_chars_per_page": float,
        }
    """
    total_pages = len(pages)
    char_counts = [p["char_count"] for p in pages]

    empty_page_numbers = [p["page_number"] for p in pages if p["char_count"] == 0]
    low_text_pages = sum(1 for c in char_counts if 0 < c < 20)

    avg_chars = sum(char_counts) / total_pages if total_pages else 0.0

    return {
        "total_pages": total_pages,
        "empty_pages": len(empty_page_numbers),
        "low_text_pages": low_text_pages,
        "empty_page_numbers": empty_page_numbers,
        "avg_chars_per_page": round(avg_chars, 1),
    }
