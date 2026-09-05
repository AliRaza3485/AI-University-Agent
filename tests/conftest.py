from pathlib import Path

import pytest

DATA_DIR = Path("data/raw_documents")
PDF_PATH = DATA_DIR / "Prospectus - FALL 2026 (29-07-2026).pdf"


@pytest.fixture
def pdf_path() -> Path:
    if not PDF_PATH.exists():
        pytest.skip(f"Test PDF not found at {PDF_PATH}")
    return PDF_PATH


@pytest.fixture
def sample_raw_pages():
    return [
        {"page_number": 1, "text": "UNIVERSITY OF EDUCATION, LAHORE\nFALL 2026\nWelcome.", "char_count": 50},
        {"page_number": 2, "text": "Some body text here.", "char_count": 20},
        {"page_number": 3, "text": "", "char_count": 0},
    ]
