"""PDF text extraction utilities (text layer first, OCR fallback for scanned pages)."""

from __future__ import annotations

import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

from backend.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ParsedPDF:
    """Extracted text plus traceability info (per-page text, which pages were OCR-ed, timings)."""

    text: str
    page_count: int
    pages: list[str] = field(default_factory=list)
    ocr_pages: list[int] = field(default_factory=list)
    ocr_seconds: float = 0.0
    ocr_error: str | None = None


def _ocr_pages(content: bytes, page_indices: list[int]) -> dict[int, str]:
    from backend.services.parser.ocr import get_ocr_engine, render_pdf_page

    engine = get_ocr_engine(settings.ocr_engine, settings.ocr_languages, settings.ocr_tesseract_cmd)

    def run(idx: int) -> tuple[int, str]:
        image = render_pdf_page(content, idx, settings.ocr_dpi)
        return idx, engine.image_to_text(image).strip()

    workers = max(1, min(settings.ocr_workers, len(page_indices)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(run, page_indices))


def parse_pdf(content: bytes) -> ParsedPDF:
    """Extract text per page; pages without a usable text layer are OCR-ed (if enabled)."""
    reader = PdfReader(io.BytesIO(content))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    result = ParsedPDF(text="", page_count=len(pages))

    if settings.ocr_enabled:
        missing = [i for i, txt in enumerate(pages) if len(txt) < settings.ocr_min_chars]
        if settings.ocr_max_pages > 0:
            missing = missing[: settings.ocr_max_pages]
        if missing:
            t0 = time.perf_counter()
            try:
                for idx, txt in _ocr_pages(content, missing).items():
                    pages[idx] = txt
                result.ocr_pages = missing
            except Exception as exc:  # OCR is a best-effort fallback; never crash the pipeline
                logger.warning("OCR nie powiódł się: %s", exc)
                result.ocr_error = str(exc)
            result.ocr_seconds = time.perf_counter() - t0

    result.pages = pages
    result.text = "\n\n".join(page for page in pages if page)
    return result


def extract_text_from_pdf(content: bytes) -> str:
    """Extract raw text from PDF bytes (pypdf text layer + OCR fallback)."""
    return parse_pdf(content).text


def extract_text_from_pdf_file(path: Path) -> str:
    """Read a PDF file from disk and extract its text."""
    return extract_text_from_pdf(path.read_bytes())
