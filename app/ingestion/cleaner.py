"""
Text cleaning utilities for the ingestion pipeline.

IMPORTANT: The rules in this module are NOT generic/speculative. They are
derived directly from inspecting real extracted text from the Fall 2026
Prospectus PDF (see notebooks/01_ingestion_exploration.ipynb). Findings
from that inspection:

    - Repeated header: roughly half of sampled pages start with a
      two-line running header: "UNIVERSITY OF EDUCATION, LAHORE"
      followed by a line starting with "FALL ...". Pages WITHOUT this
      header (faculty listings, bullet-point pages, calendar pages)
      simply don't have it - so removal must be conditional on the
      pattern actually being present, never a blind positional strip.
    - No repeated footer pattern was found - footer content varies
      page to page, so no footer-stripping rule is implemented here.
    - No broken line breaks were observed.
    - No weird/control characters were observed.

If a future document shows different noise patterns, EXTEND this module
with new targeted rules based on evidence - do not add speculative
"just in case" cleaning logic now.
"""

from __future__ import annotations

import re

# Line 1: institution name, exact match (case-insensitive).
# Line 2: starts with "FALL" - the exact suffix (year/session wording)
# varies slightly across pages, so this is intentionally a prefix match
# rather than a full-line match.
_HEADER_LINE_1 = re.compile(r"^UNIVERSITY OF EDUCATION,?\s*LAHORE\s*$", re.IGNORECASE)
_HEADER_LINE_2_PREFIX = re.compile(r"^FALL\b", re.IGNORECASE)


def strip_running_header(text: str) -> str:
    """
    Remove the repeated 2-line running header from the top of a page's
    text, if present. Pages that don't have it are returned unchanged -
    this function never removes content that doesn't match the pattern.
    """
    lines = text.split("\n")

    if (
        len(lines) >= 2
        and _HEADER_LINE_1.match(lines[0].strip())
        and _HEADER_LINE_2_PREFIX.match(lines[1].strip())
    ):
        lines = lines[2:]

    return "\n".join(lines)


def normalize_whitespace(text: str) -> str:
    """
    Collapse redundant whitespace without destroying paragraph structure:
        - trims trailing whitespace on each line
        - collapses runs of 2+ blank lines down to a single blank line
        - strips leading/trailing blank lines from the whole page
    """
    lines = [line.rstrip() for line in text.split("\n")]

    cleaned_lines: list[str] = []
    prev_was_blank = False
    for line in lines:
        if line == "":
            if not prev_was_blank:
                cleaned_lines.append(line)
            prev_was_blank = True
        else:
            cleaned_lines.append(line)
            prev_was_blank = False

    return "\n".join(cleaned_lines).strip()


def clean_page_text(text: str) -> str:
    """Apply all cleaning steps, in order, to a single page's raw text."""
    text = strip_running_header(text)
    text = normalize_whitespace(text)
    return text


def clean_pages(pages: list[dict]) -> list[dict]:
    """
    Apply cleaning to a full list of page dicts (the output of
    extract_pdf_pages). Returns NEW dicts - the raw "text" field is left
    untouched, and cleaned output is added under "cleaned_text" plus a
    matching "cleaned_char_count".

    Keeping the raw text alongside the cleaned text (rather than
    overwriting it) is deliberate: it preserves an audit trail so we can
    always compare what was removed, useful when debugging retrieval
    quality issues later.
    """
    cleaned = []
    for page in pages:
        cleaned_text = clean_page_text(page["text"])
        cleaned.append(
            {
                **page,
                "cleaned_text": cleaned_text,
                "cleaned_char_count": len(cleaned_text),
            }
        )
    return cleaned
