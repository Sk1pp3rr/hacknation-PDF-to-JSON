# OCR Module — PDF → Raw Text (Polish)

Moduł ekstrakcji surowego tekstu z wielostronicowych PDF-ów (cyfrowych i skanów)
w języku polskim, zoptymalizowany dla pism z zamówień publicznych / KIO.

## Wymagania

| Komponent | Licencja | Uwagi |
|-----------|----------|-------|
| [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) ≥ 4.0 | Apache-2.0 | binarka systemowa |
| [pytesseract](https://pypi.org/project/pytesseract/) | Apache-2.0 | Python wrapper |
| [pypdfium2](https://pypi.org/project/pypdfium2/) | Apache-2.0 / BSD-3 | renderowanie PDF |
| [Pillow](https://pypi.org/project/Pillow/) | HPND | obsługa obrazów |

> **Brak zależności AGPL/GPL.** Wszystkie biblioteki są open-source i działają na CPU.

## Instalacja Tesseracta z modelem `pol`

### Linux (Debian / Ubuntu)

```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-pol
```

Dla lepszej jakości (tessdata_best):
```bash
wget https://github.com/tesseract-ocr/tessdata_best/raw/main/pol.traineddata
sudo mv pol.traineddata /usr/share/tesseract-ocr/5/tessdata/
```

### Windows

1. Pobierz instalator z https://github.com/UB-Mannheim/tesseract/wiki
2. Podczas instalacji zaznacz język **Polish** (pol).
3. Dodaj do PATH lub ustaw zmienną `TESSERACT_CMD`.

### macOS

```bash
brew install tesseract
brew install tesseract-lang   # zawiera pol
```

## Instalacja zależności Pythona

```bash
pip install -r requirements.txt
```

## Użycie

### Python API

```python
from ocr import extract_pages

result = extract_pages("dane/przyklad_01/odwolanie.pdf")
# {"page_1": "...", "page_2": "...", ...}
```

### CLI

```bash
python -m ocr dane/przyklad_01/odwolanie.pdf
```

Wypisuje JSON na stdout z `ensure_ascii=False` (polskie znaki bez `\u`):
```json
{
  "page_1": "ODWOŁANIE\ndo Krajowej Izby Odwoławczej...",
  "page_2": "..."
}
```

Zapisanie do pliku:
```bash
python -m ocr plik.pdf > wynik.json
```

## Zmienne środowiskowe

| Zmienna | Opis | Domyślnie |
|---------|------|-----------|
| `TESSERACT_CMD` | Ścieżka do binarki `tesseract` | auto-detect |
| `TESSDATA_PREFIX` | Katalog z `*.traineddata` (np. tessdata_best) | systemowy |

## Stałe modułu (w `ocr/pdf.py`)

| Stała | Wartość | Opis |
|-------|---------|------|
| `LANG` | `"pol"` | Język Tesseracta |
| `DPI` | `300` | Rozdzielczość renderowania |
| `MIN_TEXT_CHARS` | `50` | Min. znaków dla warstwy tekstowej |
| `WORKERS` | `min(4, cpu_count)` | Wątki równoległe |
| `TESSERACT_CONFIG` | `--oem 1 --psm 3 -c preserve_interword_spaces=1` | Konfiguracja Tesseracta |

## Logika wyboru metody

Dla każdej strony PDF:

1. **Warstwa tekstowa** — jeśli zawiera ≥ 50 znaków i < 2% znaków `\ufffd` / sterujących → użyj jej.
2. **OCR** — w przeciwnym razie wyrenderuj stronę w 300 DPI (skala szarości) i uruchom Tesseract.

Logi diagnostyczne trafiają na stderr:
```
[ocr] page_1: method=text-layer  chars=1523  time=0.01s
[ocr] page_2: method=ocr         chars=892   time=1.34s
```

## Testy

```bash
pytest tests/test_ocr.py -v
```

## Struktura

```
ocr/
  __init__.py    # eksportuje extract_pages
  engine.py      # OcrEngine (ABC) + TesseractEngine
  pdf.py         # renderowanie, warstwa tekstowa, extract_pages()
  __main__.py    # CLI: python -m ocr <plik.pdf>
```
