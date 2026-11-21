"""PDF page rendering, text-layer extraction, and the public ``extract_pages`` function.

Stack (all open-source, CPU-only):
    pypdfium2  (Apache-2.0 / BSD-3)  — render pages, read embedded text layer.
    Pillow     (HPND)                — image handling.
    pytesseract (Apache-2.0)         — OCR bridge to Tesseract.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import pypdfium2 as pdfium

from ocr.engine import OcrEngine, TesseractEngine

if TYPE_CHECKING:
    from PIL import Image

# ── Module-level constants (not parameters) ──────────────────────────────────
LANG: str = "pol"
DPI: int = 300
MIN_TEXT_CHARS: int = 50
WORKERS: int = min(4, os.cpu_count() or 1)
TESSERACT_CONFIG: str = "--oem 1 --psm 3 -c preserve_interword_spaces=1"

# Limit OpenMP threads per Tesseract subprocess to avoid oversubscription.
os.environ["OMP_THREAD_LIMIT"] = "1"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _is_text_layer_usable(text: str) -> bool:
    """Return *True* when *text* looks like a real embedded text layer.

    Criteria:
      1. At least ``MIN_TEXT_CHARS`` characters.
      2. Less than 2 % replacement characters (U+FFFD) or control chars (< 0x20
         excluding common whitespace \\n \\r \\t).
    """
    if len(text) < MIN_TEXT_CHARS:
        return False

    bad = 0
    for ch in text:
        cp = ord(ch)
        if cp == 0xFFFD:
            bad += 1
        elif cp < 0x20 and ch not in "\n\r\t":
            bad += 1
    return (bad / len(text)) < 0.02


def _get_text_layer(pdf_doc: pdfium.PdfDocument, page_index: int) -> str:
    """Extract the embedded text layer of one page via pypdfium2."""
    page = pdf_doc[page_index]
    try:
        textpage = page.get_textpage()
        try:
            return textpage.get_text_range()
        finally:
            textpage.close()
    finally:
        page.close()


def _render_page_grayscale(pdf_bytes: bytes, page_index: int) -> Image.Image:
    """Render a single page to a grayscale PIL image at ``DPI`` resolution."""
    doc = pdfium.PdfDocument(pdf_bytes)
    try:
        page = doc[page_index]
        try:
            bitmap = page.render(scale=DPI / 72.0, grayscale=True)
            return bitmap.to_pil()
        finally:
            page.close()
    finally:
        doc.close()


def _build_engine() -> OcrEngine:
    """Construct the default OCR engine (lazy, so import errors surface here)."""
    return TesseractEngine(lang=LANG, config=TESSERACT_CONFIG)


# ── Core: per-page extraction ────────────────────────────────────────────────

def _process_page(
    pdf_bytes: bytes,
    page_index: int,
    text_layers: dict[int, str],
    engine: OcrEngine | None,
) -> tuple[str, str, int, float]:
    """Return ``(key, text, char_count, elapsed)`` for one page.

    Uses the text layer when usable; otherwise OCRs the rendered bitmap.
    """
    key = f"page_{page_index + 1}"
    t0 = time.perf_counter()

    layer = text_layers.get(page_index, "")
    if _is_text_layer_usable(layer):
        method = "text-layer"
        text = layer
    else:
        method = "ocr"
        if engine is None:
            raise RuntimeError(
                "OCR engine is required but was not initialised. "
                "Ensure Tesseract is installed."
            )
        img = _render_page_grayscale(pdf_bytes, page_index)
        text = engine.ocr_image(img)

    elapsed = time.perf_counter() - t0
    char_count = len(text)

    # Log to stderr (never pollutes stdout / return value).
    print(
        f"[ocr] {key}: method={method}  chars={char_count}  time={elapsed:.2f}s",
        file=sys.stderr,
    )
    return key, text, char_count, elapsed


# ── Public API ───────────────────────────────────────────────────────────────

def extract_pages(pdf_path: str | Path) -> dict[str, str]:
    """Extract raw text from every page of *pdf_path*.

    Returns
    -------
    dict[str, str]
        ``{"page_1": "…", "page_2": "…", …}`` — keys in page order,
        values are **raw** text (no normalisation, no trimming).

    Raises
    ------
    FileNotFoundError
        *pdf_path* does not exist.
    ValueError
        File is not a PDF, or is encrypted / corrupted.
    RuntimeError
        Tesseract binary or Polish language pack missing.
    """
    path = Path(pdf_path)

    # ── Validate input ────────────────────────────────────────────────────
    if not path.exists():
        raise FileNotFoundError(f"Plik nie istnieje: {path}")
    if not path.is_file():
        raise ValueError(f"Ścieżka nie jest plikiem: {path}")

    pdf_bytes = path.read_bytes()

    # Quick magic-number check (PDF files start with %PDF).
    if not pdf_bytes[:5].startswith(b"%PDF"):
        raise ValueError(
            f"Plik nie jest dokumentem PDF (brak nagłówka %PDF): {path}"
        )

    # ── Open the document (catches encryption / corruption) ───────────────
    try:
        doc = pdfium.PdfDocument(pdf_bytes)
    except pdfium.PdfiumError as exc:
        raise ValueError(
            f"Nie można otworzyć pliku PDF (uszkodzony lub zaszyfrowany): {path}\n"
            f"Szczegóły: {exc}"
        ) from exc

    page_count = len(doc)
    if page_count == 0:
        doc.close()
        return {}

    # ── Read text layers (cheap, synchronous) ─────────────────────────────
    text_layers: dict[int, str] = {}
    for idx in range(page_count):
        try:
            text_layers[idx] = _get_text_layer(doc, idx)
        except Exception:
            text_layers[idx] = ""
    doc.close()

    # ── Decide if we need the OCR engine at all ───────────────────────────
    needs_ocr = any(
        not _is_text_layer_usable(text_layers.get(i, ""))
        for i in range(page_count)
    )
    engine: OcrEngine | None = _build_engine() if needs_ocr else None

    # ── Process pages in parallel ─────────────────────────────────────────
    workers = min(WORKERS, page_count)
    results: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_process_page, pdf_bytes, idx, text_layers, engine)
            for idx in range(page_count)
        ]
        for fut in futures:
            key, text, _cc, _el = fut.result()
            results[key] = text

    # Guarantee key order (page_1, page_2, …).
    return {f"page_{i + 1}": results[f"page_{i + 1}"] for i in range(page_count)}
