"""Tests for the ``ocr`` module.

Requirements:
    - Tesseract with the ``pol`` language pack (for live OCR tests).
    - DejaVu Sans font (in tests/fixtures) or Arial font.
"""

from __future__ import annotations

import io
import os
import re
import shutil
from pathlib import Path

import pypdfium2 as pdfium
import pytest
from PIL import Image, ImageDraw, ImageFont


def _tesseract_available() -> bool:
    """Return True if Tesseract binary is reachable."""
    try:
        import pytesseract

        tess_cmd = os.environ.get("TESSERACT_CMD")
        if tess_cmd:
            pytesseract.pytesseract.tesseract_cmd = tess_cmd
        elif os.name == "nt" and shutil.which(pytesseract.pytesseract.tesseract_cmd) is None:
            win_candidate = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.path.isfile(win_candidate):
                pytesseract.pytesseract.tesseract_cmd = win_candidate

        pytesseract.get_languages(config="")
        return True
    except Exception:
        return False


def _tesseract_has_pol() -> bool:
    """Return True if Tesseract has the 'pol' language pack."""
    try:
        import pytesseract

        tess_cmd = os.environ.get("TESSERACT_CMD")
        if tess_cmd:
            pytesseract.pytesseract.tesseract_cmd = tess_cmd
        elif os.name == "nt" and shutil.which(pytesseract.pytesseract.tesseract_cmd) is None:
            win_candidate = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.path.isfile(win_candidate):
                pytesseract.pytesseract.tesseract_cmd = win_candidate

        return "pol" in pytesseract.get_languages(config="")
    except Exception:
        return False


requires_tesseract = pytest.mark.skipif(
    not _tesseract_available(),
    reason="Tesseract OCR is not installed",
)
requires_tesseract_pol = pytest.mark.skipif(
    not _tesseract_has_pol(),
    reason="Tesseract 'pol' language pack is not installed",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

POLISH_TEXT = (
    'Odwołujący: Przedsiębiorstwo Budowlane „Żółw" Sp. z o.o., '
    "ul. Łąkowa 5, 90-001 Łódź"
)

# Polish chars that MUST survive the round-trip.
POLISH_CHARS = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")

# Subset of words we expect Tesseract to get right on a clean 300 DPI render.
EXPECTED_WORDS = [
    "Odwołujący",
    "Przedsiębiorstwo",
    "Budowlane",
    "Żółw",
    "Łąkowa",
    "Łódź",
]


def _find_font() -> str:
    """Return a path to a TrueType font that covers Polish glyphs."""
    fixture_font = Path(__file__).parent / "fixtures" / "DejaVuSans.ttf"
    candidates = [
        str(fixture_font),
        "arial.ttf",          # Windows
        "Arial.ttf",
        "DejaVuSans.ttf",    # Linux (fonts-dejavu-core)
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for name in candidates:
        try:
            ImageFont.truetype(name, 20)
            return name
        except OSError:
            continue
    pytest.skip("No TrueType font with Polish glyphs found on this system")


def _make_image_pdf(text: str, pages: int = 1) -> bytes:
    """Create an image-only PDF (no text layer) with *text* rendered on each page.

    Uses Pillow to draw text onto a 300 DPI A4-ish image, then embeds each
    image as a page in a PDF via pypdfium2.
    """
    font_path = _find_font()
    font = ImageFont.truetype(font_path, 36)

    width, height = 2480, 500  # ~A4 width at 300 DPI, short page
    images: list[Image.Image] = []

    for _ in range(pages):
        img = Image.new("L", (width, height), 255)
        draw = ImageDraw.Draw(img)
        draw.text((60, 60), text, fill=0, font=font)
        images.append(img)

    buf = io.BytesIO()
    if len(images) == 1:
        images[0].save(buf, format="PDF", resolution=300.0)
    else:
        images[0].save(
            buf,
            format="PDF",
            resolution=300.0,
            save_all=True,
            append_images=images[1:],
        )
    return buf.getvalue()


def _manual_text_pdf(texts: list[str]) -> bytes:
    """Build a minimal PDF with text layers using raw PDF operators."""
    buf = io.BytesIO()

    objects: list[bytes] = []

    def add_obj(content: bytes) -> int:
        obj_num = len(objects) + 1
        objects.append(content)
        return obj_num

    catalog = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    add_obj(catalog)

    font_obj = b"3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>\nendobj\n"

    page_obj_nums = []
    stream_obj_nums = []

    next_obj = 4
    for i, text in enumerate(texts):
        safe_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream_content = f"BT /F1 12 Tf 72 720 Td ({safe_text}) Tj ET".encode("latin-1", errors="replace")
        stream_obj = f"{next_obj} 0 obj\n<< /Length {len(stream_content)} >>\nstream\n".encode()
        stream_obj += stream_content
        stream_obj += b"\nendstream\nendobj\n"
        stream_obj_nums.append(next_obj)
        next_obj += 1

        page_obj = (
            f"{next_obj} 0 obj\n"
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {next_obj - 1} 0 R "
            f"/Resources << /Font << /F1 3 0 R >> >> >>\n"
            f"endobj\n"
        ).encode()
        page_obj_nums.append(next_obj)
        next_obj += 1

    page_refs = " ".join(f"{n} 0 R" for n in page_obj_nums)
    pages_obj = f"2 0 obj\n<< /Type /Pages /Kids [{page_refs}] /Count {len(texts)} >>\nendobj\n".encode()

    buf.write(b"%PDF-1.4\n")

    all_objs = [catalog, pages_obj, font_obj]
    for i, text in enumerate(texts):
        safe_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream_content = f"BT /F1 12 Tf 72 720 Td ({safe_text}) Tj ET".encode("latin-1", errors="replace")
        stream_num = stream_obj_nums[i]
        stream_obj = f"{stream_num} 0 obj\n<< /Length {len(stream_content)} >>\nstream\n".encode()
        stream_obj += stream_content
        stream_obj += b"\nendstream\nendobj\n"
        all_objs.append(stream_obj)

        page_num = page_obj_nums[i]
        page_obj = (
            f"{page_num} 0 obj\n"
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {stream_num} 0 R "
            f"/Resources << /Font << /F1 3 0 R >> >> >>\n"
            f"endobj\n"
        ).encode()
        all_objs.append(page_obj)

    obj_offsets: dict[int, int] = {}
    for obj_bytes in all_objs:
        first_line = obj_bytes.split(b"\n", 1)[0]
        obj_num = int(first_line.split(b" ", 1)[0])
        obj_offsets[obj_num] = buf.tell()
        buf.write(obj_bytes)

    xref_offset = buf.tell()
    max_obj = max(obj_offsets.keys())
    buf.write(b"xref\n")
    buf.write(f"0 {max_obj + 1}\n".encode())
    buf.write(b"0000000000 65535 f \n")
    for i in range(1, max_obj + 1):
        offset = obj_offsets.get(i, 0)
        buf.write(f"{offset:010d} 00000 n \n".encode())

    buf.write(b"trailer\n")
    buf.write(f"<< /Size {max_obj + 1} /Root 1 0 R >>\n".encode())
    buf.write(b"startxref\n")
    buf.write(f"{xref_offset}\n".encode())
    buf.write(b"%%EOF\n")

    return buf.getvalue()


def _collapse_whitespace(text: str) -> str:
    """Reduce all whitespace runs to a single space for fuzzy comparison."""
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@requires_tesseract_pol
class TestExtractPagesImagePDF:
    """OCR on an image-only PDF with Polish text."""

    @pytest.fixture(autouse=True)
    def _pdf(self, tmp_path: Path) -> None:
        self.pdf_path = tmp_path / "polish.pdf"
        self.pdf_path.write_bytes(_make_image_pdf(POLISH_TEXT))

    def test_polish_words_recognised(self) -> None:
        """Tesseract must recognise the key Polish words at 300 DPI."""
        from ocr import extract_pages

        result = extract_pages(self.pdf_path)
        assert "page_1" in result

        collapsed = _collapse_whitespace(result["page_1"])
        for word in EXPECTED_WORDS:
            assert word in collapsed, (
                f"Brak słowa {word!r} w tekście OCR:\n{collapsed}"
            )

    def test_polish_chars_present(self) -> None:
        """Polish diacritics must survive the OCR round-trip."""
        from ocr import extract_pages

        text = extract_pages(self.pdf_path)["page_1"]
        present = {ch for ch in text if ch in POLISH_CHARS}
        assert len(present) >= 5, (
            f"Za mało polskich znaków w wyniku OCR. Znalezione: {present}"
        )


class TestExtractPagesMultiPage:
    """Multi-page PDF returns exactly the right keys in order."""

    def test_three_pages_text_layer(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "three_text.pdf"
        pdf_path.write_bytes(
            _manual_text_pdf(
                [
                    "Pierwsza strona z wystarczającą liczbą znaków do zaliczenia warstwy tekstowej.",
                    "Druga strona z wystarczającą liczbą znaków do zaliczenia warstwy tekstowej.",
                    "Trzecia strona z wystarczającą liczbą znaków do zaliczenia warstwy tekstowej.",
                ]
            )
        )

        from ocr import extract_pages

        result = extract_pages(pdf_path)

        assert list(result.keys()) == ["page_1", "page_2", "page_3"]
        assert all(isinstance(v, str) for v in result.values())
        assert "Pierwsza" in result["page_1"]
        assert "Druga" in result["page_2"]
        assert "Trzecia" in result["page_3"]

    @requires_tesseract_pol
    def test_three_pages_image_pdf(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "three_img.pdf"
        pdf_path.write_bytes(_make_image_pdf("Strona testowa", pages=3))

        from ocr import extract_pages

        result = extract_pages(pdf_path)

        assert list(result.keys()) == ["page_1", "page_2", "page_3"]
        assert all(isinstance(v, str) for v in result.values())


class TestExtractPagesTextLayer:
    """When a usable text layer exists, OCR should NOT be called."""

    def test_text_layer_used(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        long_text = "Krajowa Izba Odwoławcza rozpatruje odwołanie złożone przez wykonawcę w postępowaniu o udzielenie zamówienia publicznego."
        pdf_path = tmp_path / "textlayer.pdf"
        pdf_path.write_bytes(_manual_text_pdf([long_text]))

        import ocr.pdf as ocr_pdf

        engine_built = False

        def _mock_build() -> ocr_pdf.OcrEngine:
            nonlocal engine_built
            engine_built = True

            class _Boom(ocr_pdf.OcrEngine):
                @property
                def name(self) -> str:
                    return "boom"

                def ocr_image(self, img: Image.Image) -> str:
                    raise AssertionError("OCR engine should not have been called!")

            return _Boom()

        monkeypatch.setattr(ocr_pdf, "_build_engine", _mock_build)

        from ocr import extract_pages

        result = extract_pages(pdf_path)

        assert "page_1" in result
        assert isinstance(result["page_1"], str)
        assert len(result["page_1"]) > 0
        assert not engine_built, "OCR engine was built even though text layer was usable"


class TestErrorHandling:
    """Descriptive errors for all 5 required failure modes."""

    def test_missing_file(self, tmp_path: Path) -> None:
        from ocr import extract_pages

        with pytest.raises(FileNotFoundError, match="nie istnieje"):
            extract_pages(tmp_path / "brak.pdf")

    def test_not_a_pdf(self, tmp_path: Path) -> None:
        from ocr import extract_pages

        fake = tmp_path / "fake.pdf"
        fake.write_text("This is not a PDF", encoding="utf-8")

        with pytest.raises(ValueError, match="nie jest dokumentem PDF"):
            extract_pages(fake)

    def test_corrupted_or_encrypted_pdf(self, tmp_path: Path) -> None:
        from ocr import extract_pages

        corrupted = tmp_path / "corrupted.pdf"
        corrupted.write_bytes(b"%PDF-1.4\n%%EOF_INVALID_CORRUPTED_DATA")

        with pytest.raises(ValueError, match="uszkodzony lub zaszyfrowany"):
            extract_pages(corrupted)

    def test_missing_tesseract(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from pytesseract import TesseractNotFoundError
        import pytesseract

        def _mock_get_languages(config: str = "") -> list[str]:
            raise TesseractNotFoundError()

        monkeypatch.setattr(pytesseract, "get_languages", _mock_get_languages)

        blank_img = Image.new("L", (100, 100), 255)
        buf = io.BytesIO()
        blank_img.save(buf, format="PDF", resolution=300.0)
        pdf_path = tmp_path / "needs_ocr.pdf"
        pdf_path.write_bytes(buf.getvalue())

        from ocr import extract_pages

        with pytest.raises(RuntimeError, match="Tesseract OCR nie jest zainstalowany"):
            extract_pages(pdf_path)

    def test_missing_pol_model(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import pytesseract

        def _mock_get_languages(config: str = "") -> list[str]:
            return ["eng", "osd"]

        monkeypatch.setattr(pytesseract, "get_languages", _mock_get_languages)

        blank_img = Image.new("L", (100, 100), 255)
        buf = io.BytesIO()
        blank_img.save(buf, format="PDF", resolution=300.0)
        pdf_path = tmp_path / "needs_pol.pdf"
        pdf_path.write_bytes(buf.getvalue())

        from ocr import extract_pages

        with pytest.raises(RuntimeError, match="Brak pakietu językowego Tesseract dla 'pol'"):
            extract_pages(pdf_path)

    def test_mocked_ocr_blank_page(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify that an OCR-processed page returns a str (even empty) without actual Tesseract."""
        blank_img = Image.new("L", (100, 100), 255)
        buf = io.BytesIO()
        blank_img.save(buf, format="PDF", resolution=300.0)
        pdf_path = tmp_path / "blank.pdf"
        pdf_path.write_bytes(buf.getvalue())

        import ocr.pdf as ocr_pdf

        class _MockEngine(ocr_pdf.OcrEngine):
            @property
            def name(self) -> str:
                return "mock"

            def ocr_image(self, img: Image.Image) -> str:
                return ""

        monkeypatch.setattr(ocr_pdf, "_build_engine", lambda: _MockEngine())

        from ocr import extract_pages

        result = extract_pages(pdf_path)
        assert result == {"page_1": ""}
        assert isinstance(result["page_1"], str)

    @requires_tesseract
    def test_empty_page_returns_string_live(self, tmp_path: Path) -> None:
        """Live test: blank page must return str with real Tesseract."""
        img = Image.new("L", (200, 200), 255)
        buf = io.BytesIO()
        img.save(buf, format="PDF", resolution=300.0)
        pdf_path = tmp_path / "blank_live.pdf"
        pdf_path.write_bytes(buf.getvalue())

        from ocr import extract_pages

        result = extract_pages(pdf_path)
        assert "page_1" in result
        assert isinstance(result["page_1"], str)
