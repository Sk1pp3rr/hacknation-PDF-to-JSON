"""OCR engine abstraction — swap Tesseract for another backend without touching the rest.

Classes:
    OcrEngine  — abstract base (interface).
    TesseractEngine — Tesseract adapter via pytesseract.
"""

from __future__ import annotations

import abc
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


class OcrEngine(abc.ABC):
    """Recognise text on a single rendered page image (PIL ``Image``)."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable engine identifier (used in logs)."""

    @abc.abstractmethod
    def ocr_image(self, img: Image.Image) -> str:
        """Return raw UTF-8 text recognised in *img*."""


class TesseractEngine(OcrEngine):
    """Tesseract OCR adapter (requires the ``tesseract`` binary + language data).

    Licence: Apache-2.0 (tesseract) · Apache-2.0 (pytesseract).
    """

    _name = "tesseract"

    def __init__(self, *, lang: str, config: str) -> None:
        import pytesseract  # lazy — fail fast with a clear message if missing

        # Honour env-var overrides --------------------------------------------------
        tess_cmd = os.environ.get("TESSERACT_CMD")
        if tess_cmd:
            pytesseract.pytesseract.tesseract_cmd = tess_cmd
        elif os.name == "nt" and shutil.which(pytesseract.pytesseract.tesseract_cmd) is None:
            for candidate in [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
            ]:
                if os.path.isfile(candidate):
                    pytesseract.pytesseract.tesseract_cmd = candidate
                    break

        tessdata = os.environ.get("TESSDATA_PREFIX")
        if tessdata:
            os.environ["TESSDATA_PREFIX"] = tessdata  # pytesseract reads it

        # Validate that the Tesseract binary is reachable ---------------------------
        try:
            available = pytesseract.get_languages(config="")
        except (pytesseract.TesseractNotFoundError, FileNotFoundError):
            raise RuntimeError(
                "Tesseract OCR nie jest zainstalowany lub brak go w PATH.\n"
                "Instrukcja instalacji:\n"
                "  Linux (Debian/Ubuntu) : apt install tesseract-ocr tesseract-ocr-pol\n"
                "  Windows                : winget install UB-Mannheim.TesseractOCR\n"
                "  Lokalizacja binarki    : ustaw zmienną środowiskową TESSERACT_CMD"
            )
        except Exception as exc:
            raise RuntimeError(
                f"Błąd podczas sprawdzania silnika Tesseract: {exc}\n"
                "Upewnij się, że Tesseract jest poprawnie zainstalowany lub ustaw TESSERACT_CMD."
            ) from exc

        if lang not in available:
            raise RuntimeError(
                f"Brak pakietu językowego Tesseract dla '{lang}'.\n"
                f"Dostępne pakiety: {available}\n"
                "Instrukcja instalacji:\n"
                f"  Linux (Debian/Ubuntu) : apt install tesseract-ocr tesseract-ocr-{lang}\n"
                f"  Windows                : uruchom instalator Tesseract i zaznacz '{lang}'\n"
                f"  Własny katalog danych : ustaw zmienną środowiskową TESSDATA_PREFIX"
            )

        self._pytesseract = pytesseract
        self._lang = lang
        self._config = config

    # -- public interface -----------------------------------------------------------

    @property
    def name(self) -> str:  # noqa: D401
        return self._name

    def ocr_image(self, img: Image.Image) -> str:
        """Run Tesseract on *img* and return the raw text."""
        return self._pytesseract.image_to_string(
            img, lang=self._lang, config=self._config
        )
