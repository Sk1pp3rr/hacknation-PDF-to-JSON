"""PDF text extraction utilities."""

import io
from pathlib import Path

from pypdf import PdfReader


def extract_text_from_pdf(content: bytes) -> str:
    """Extract raw text from PDF bytes using pypdf."""
    reader = PdfReader(io.BytesIO(content))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n\n".join(page for page in pages if page)


def extract_text_from_pdf_file(path: Path) -> str:
    """Read a PDF file from disk and extract its text."""
    return extract_text_from_pdf(path.read_bytes())
