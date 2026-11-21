"""OCR module for extracting raw text from Polish PDF documents.

Public API:
    extract_pages(pdf_path) -> dict[str, str]
        Keys: "page_1", "page_2", ...
        Values: raw text (UTF-8), no post-processing.
"""

from ocr.pdf import extract_pages

__all__ = ["extract_pages"]
