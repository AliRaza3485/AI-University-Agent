"""
Text cleaning utilities for the ingestion pipeline.

IMPORTANT: The rules in this module are NOT generic/speculative. They are
derived directly from inspecting real extracted text from the Fall 2026
Prospectus PDF (see notebooks/01_ingestion_exploration.ipynb and
notebooks/02_cleaner_verification.ipynb).

Findings (v2, after deeper diagnostic investigation):

    - The phrase "UNIVERSITY OF EDUCATION, LAHORE" followed by a line
      starting with "FALL" is a repeating header/footer block emitted by
      the PDF's page furniture (running header/footer). Critically, this
      block does NOT always appear at a fixed position (first 2 lines).
      PyMuPDF extracts text blocks in the order they're stored in the
      PDF's content stream, not strictly top-to-bottom - so this block
      can appear at the very start, in the middle of body text, or
      duplicated back-to-back, sometimes followed by a standalone page
      number line (e.g. "...UNIVERSITY OF EDUCATION, LAHORE\nFALL\n107").
    - Because of this, removal MUST be pattern-based (regex, matched
      anywhere in the text) rather than positional (first-N-lines only).
      An earlier version of this module used positional stripping and
      left the phrase present on 302/382 pages - this version fixes that.
    - No broken line breaks were observed.
    - No weird/control characters were observed.

If a future document shows different noise patterns, EXTEND this module
with new targeted rules based on evidence - do not add speculative
"just in case" cleaning logic now.
"""

from __future__ import annotations

import re

# Matches one or more consecutive repeats of the running header/footer
# block ("UNIVERSITY OF EDUCATION, LAHORE" + a line starting with "FALL"),
# optionally followed by a standalone page-number line that belongs to
# the same footer (e.g. "...FALL\n107"). Matched anywhere in the text,
# not just at a fixed position, because extraction order is not reliably
# top-to-bottom.
_HEADER_FOOTER_BLOCK = re.compile(
    r"(?:UNIVERSITY OF EDUCATION,?\s*LAHORE[ \t]*\n[ \t]*FALL[^\n]*\n?)+"
    r"(?:[ \t]*\d{1,4}[ \t]*\n?)?",
    re.IGNORECASE,
)


def strip_header_footer_blocks(text: str) -> str:
    """
    Remove all occurrences of the recurring header/footer block, wherever
    they appear in the text (start, middle, or end - possibly repeated
    back-to-back). Text without the pattern is returned unchanged.
    """
    return _HEADER_FOOTER_BLOCK.sub("", text)


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
    text = strip_header_footer_blocks(text)
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
