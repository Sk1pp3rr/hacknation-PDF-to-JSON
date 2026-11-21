"""PDF text extraction utilities (text layer first, OCR fallback for scanned pages)."""

from __future__ import annotations

import base64
import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


def _find_evidence_rects(tp: Any, page_w: float, page_h: float, ev: Any) -> list[dict[str, float]]:
    leaf = str(ev.field).rsplit(".", 1)[-1]
    val_s = str(ev.value).strip() if ev.value is not None else ""
    quote_s = str(ev.quote or "").strip()
    candidates: list[str] = []
    if len(val_s) >= 3 and leaf not in {"przez-pelnomocnika", "czynnosc_powod_id", "prowadzacy_postepowanie", "typ"}:
        candidates.append(val_s)
    if quote_s:
        candidates.append(quote_s)
        words = quote_s.split()
        if len(words) >= 4:
            candidates.append(" ".join(words[:5]))
            candidates.append(" ".join(words[-4:]))
        if len(quote_s) > 25:
            candidates.append(quote_s[:30].strip())

    for query in candidates:
        if not query or len(query) < 2:
            continue
        searcher = tp.search(query, match_case=False)
        hit = searcher.get_next()
        if hit:
            idx, count = hit
            n_rects = tp.count_rects(idx, count)
            rects: list[dict[str, float]] = []
            for r_idx in range(n_rects):
                left, bottom, right, top = tp.get_rect(r_idx)
                rects.append(
                    {
                        "x": round(max(0.0, (left / page_w) * 100.0 - 0.5), 2),
                        "y": round(max(0.0, ((page_h - top) / page_h) * 100.0 - 0.35), 2),
                        "w": round(min(100.0, ((right - left) / page_w) * 100.0 + 1.0), 2),
                        "h": round(min(100.0, ((top - bottom) / page_h) * 100.0 + 0.7), 2),
                    }
                )
            if rects:
                return rects
    return []


def render_pdf_explorer_data(
    content: bytes,
    evidence: list[Any],
    *,
    scale: float = 1.6,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Render PDF pages to PNG data URLs and locate normalized highlight rectangles for each field."""
    ev_dicts = [ev.model_dump() if hasattr(ev, "model_dump") else dict(ev) for ev in evidence]
    try:
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(content)
    except Exception as exc:
        logger.warning("Nie udało się wyrenderować stron PDF w pypdfium2: %s", exc)
        return [], ev_dicts

    try:
        page_images: list[str] = []
        textpages: list[Any] = []
        sizes: list[tuple[float, float]] = []
        for idx in range(len(doc)):
            page = doc[idx]
            sizes.append(page.get_size())
            textpages.append(page.get_textpage())
            pil_img = page.render(scale=scale).to_pil()
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG", optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            page_images.append(f"data:image/png;base64,{b64}")

        for ev_obj, ev_dict in zip(evidence, ev_dicts):
            p_idx = (getattr(ev_obj, "page", None) or 1) - 1
            if 0 <= p_idx < len(textpages):
                w, h = sizes[p_idx]
                ev_dict["rects"] = _find_evidence_rects(textpages[p_idx], w, h, ev_obj)
            else:
                ev_dict["rects"] = []
        return page_images, ev_dicts
    finally:
        doc.close()
