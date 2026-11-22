"""Thin OCR abstraction for scanned PDF pages (engine selected via configuration).

Engines are adapters behind the `OCREngine` protocol, so the pipeline does not depend
on a concrete OCR library. Default adapter: Tesseract (Apache-2.0, CPU-only) via
`pytesseract`; pages are rendered with `pypdfium2` (Apache-2.0 / BSD-3, CPU-only).
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache
from typing import Any, Callable, Protocol

# pypdfium2 wraps native PDFium C bindings that are not thread-safe across concurrent renders.
PDFIUM_LOCK = threading.Lock()


class OCREngine(Protocol):
    """Recognizes text on a single rendered page image (PIL.Image)."""

    name: str

    def image_to_text(self, image: Any) -> str: ...


class TesseractOCR:
    """Tesseract adapter (requires the `tesseract` binary with the configured language data)."""

    name = "tesseract"

    def __init__(self, languages: str = "pol", cmd: str | None = None, config: str = "--psm 3") -> None:
        import pytesseract

        # Each OCR call is single-threaded; throughput comes from parallel documents/pages.
        os.environ.setdefault("OMP_THREAD_LIMIT", "1")
        resolved = cmd or _default_tesseract_cmd()
        if resolved:
            pytesseract.pytesseract.tesseract_cmd = resolved
        self._pytesseract = pytesseract
        self.languages = languages
        self.config = config

    def image_to_text(self, image: Any) -> str:
        return self._pytesseract.image_to_string(image, lang=self.languages, config=self.config)


def _default_tesseract_cmd() -> str | None:
    candidate = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    return candidate if os.name == "nt" and os.path.isfile(candidate) else None


# Registry of available adapters: new engines can be added here and selected via `OCR_ENGINE`.
OCR_ENGINES: dict[str, Callable[..., OCREngine]] = {
    "tesseract": TesseractOCR,
}


@lru_cache(maxsize=4)
def get_ocr_engine(name: str, languages: str, cmd: str | None = None) -> OCREngine:
    try:
        factory = OCR_ENGINES[name]
    except KeyError as exc:
        raise ValueError(f"Nieznany silnik OCR: {name!r}. Dostępne: {sorted(OCR_ENGINES)}") from exc
    return factory(languages=languages, cmd=cmd)


def render_pdf_page(content: bytes, page_index: int, dpi: int) -> Any:
    """Render one PDF page to a grayscale PIL image."""
    import pypdfium2 as pdfium

    with PDFIUM_LOCK:
        pdf = pdfium.PdfDocument(content)
        try:
            page = pdf[page_index]
            bitmap = page.render(scale=dpi / 72.0, grayscale=True)
            return bitmap.to_pil()
        finally:
            pdf.close()
