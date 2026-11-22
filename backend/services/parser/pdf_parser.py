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


def _find_evidence_rects(
    tp: Any,
    page_w: float,
    page_h: float,
    ev: Any,
    *,
    scale: float = 1.6,
    crop_top_px: int = 0,
    crop_h_px: int | None = None,
) -> list[dict[str, float]]:
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

    eff_h_px = float(crop_h_px) if crop_h_px and crop_h_px > 0 else (page_h * scale)

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
                px_top = (page_h - top) * scale - crop_top_px
                px_bottom = (page_h - bottom) * scale - crop_top_px
                rects.append(
                    {
                        "x": round(max(0.0, (left / page_w) * 100.0 - 0.5), 2),
                        "y": round(max(0.0, (px_top / eff_h_px) * 100.0 - 0.35), 2),
                        "w": round(min(100.0, ((right - left) / page_w) * 100.0 + 1.0), 2),
                        "h": round(min(100.0, ((px_bottom - px_top) / eff_h_px) * 100.0 + 0.7), 2),
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
    """Render PDF pages into seamless continuous strips and locate highlight rectangles for each field."""
    ev_dicts = [ev.model_dump() if hasattr(ev, "model_dump") else dict(ev) for ev in evidence]
    try:
        from PIL import ImageOps
        import pypdfium2 as pdfium
        from backend.services.parser.ocr import PDFIUM_LOCK
    except Exception as exc:
        logger.warning("Nie udało się wyrenderować stron PDF w pypdfium2: %s", exc)
        return [], ev_dicts

    with PDFIUM_LOCK:
        try:
            doc = pdfium.PdfDocument(content)
        except Exception as exc:
            logger.warning("Nie udało się wyrenderować stron PDF w pypdfium2: %s", exc)
            return [], ev_dicts

        try:
            page_images: list[str] = []
            textpages: list[Any] = []
            sizes: list[tuple[float, float]] = []
            crops: list[tuple[int, int]] = []
            n_pages = len(doc)

            for idx in range(n_pages):
                page = doc[idx]
                sizes.append(page.get_size())
                textpages.append(page.get_textpage())
                pil_img = page.render(scale=scale).to_pil()
                img_w, img_h = pil_img.size
                crop_top = 0
                crop_bottom = img_h

                # On multi-page documents, trim inter-page blank margins so pages join into one continuous sheet.
                if n_pages > 1:
                    bbox = ImageOps.invert(pil_img.convert("L")).getbbox()
                    if bbox:
                        _, content_top, _, content_bottom = bbox
                        pad = int(14 * scale)
                        if idx > 0:
                            crop_top = max(0, content_top - pad)
                        if idx < n_pages - 1:
                            crop_bottom = min(img_h, content_bottom + pad)
                        if crop_bottom > crop_top + 40:
                            pil_img = pil_img.crop((0, crop_top, img_w, crop_bottom))
                        else:
                            crop_top, crop_bottom = 0, img_h

                crops.append((crop_top, crop_bottom - crop_top))
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG", optimize=True)
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                page_images.append(f"data:image/png;base64,{b64}")

            for ev_obj, ev_dict in zip(evidence, ev_dicts):
                p_idx = (getattr(ev_obj, "page", None) or 1) - 1
                if 0 <= p_idx < len(textpages):
                    w, h = sizes[p_idx]
                    c_top, c_h = crops[p_idx]
                    ev_dict["rects"] = _find_evidence_rects(
                        textpages[p_idx],
                        w,
                        h,
                        ev_obj,
                        scale=scale,
                        crop_top_px=c_top,
                        crop_h_px=c_h,
                    )
                else:
                    ev_dict["rects"] = []
            return page_images, ev_dicts
        finally:
            doc.close()
