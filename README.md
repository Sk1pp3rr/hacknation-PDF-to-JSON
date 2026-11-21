# Odwołania2json (hacknation-PDF-to-JSON)

Aplikacja webowa do automatycznej ekstrakcji danych ze spraw odwoławczych (odwołania, pełnomocnictwa, dowody wpłaty) z plików PDF do kanonicznego modelu JSON zgodnego ze specyfikacją UZP.

Projekt oparty na komponentach Open Source oraz europejskim API LLM (np. Mistral AI / Scaleway UE).

## Struktura repozytorium

```text
├── backend/
│   ├── main.py              # Aplikacja FastAPI i endpointy API
│   ├── config.py            # Konfiguracja i zmienne środowiskowe
│   ├── schemas.py           # Modele Pydantic (kanoniczny schemat JSON)
│   └── services/
│       ├── pdf_parser.py    # Parsowanie i podział plików PDF
│       ├── classifier.py    # Klasyfikacja i segmentacja dokumentów
│       └── llm_extractor.py # Ekstrakcja danych przez europejskie API LLM
├── ocr/                     # Samodzielny moduł OCR (pypdfium2 + Tesseract pol)
│   ├── __init__.py          # Eksportuje extract_pages(pdf_path)
│   ├── engine.py            # OcrEngine (ABC) + TesseractEngine
│   ├── pdf.py               # Warstwa tekstowa, renderowanie, extract_pages
│   ├── __main__.py          # CLI: python -m ocr <plik.pdf>
│   └── README.md            # Dokumentacja modułu OCR, instalacja, licencje
├── frontend/
│   └── app.py               # Interfejs webowy (Streamlit)
├── dane/
│   ├── negatywne/           # Próbki negatywne
│   └── przyklad_01/...      # Przykłady dokumentów i oczekiwanych JSON
├── scripts/
│   └── download_hacknation_data.py # Pobieranie danych źródłowych
├── tests/
│   └── test_smoke.py        # Podstawowe testy API
├── .env.example             # Przykładowa konfiguracja zmiennych środowiskowych
└── pyproject.toml           # Zależności i konfiguracja projektu
```

## Pobieranie danych

Katalog `dane/` można odtworzyć z repozytorium źródłowego:

```bash
python scripts/download_hacknation_data.py
```

Domyślnie pobierana jest gałąź `main` i zawartość zastępuje katalog `dane/`.
Można wskazać inną gałąź lub katalog docelowy:

```bash
python scripts/download_hacknation_data.py --branch main --target dane
```

Skrypt pobiera cały katalog `dane/`, w tym podkatalog `dane/skany/`.

## Szybki start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -e ".[dev]"
cp .env.example .env
```

Uruchomienie backendu:
```bash
fastapi dev backend/main.py
```

Uruchomienie frontendu:
```bash
streamlit run frontend/app.py
```

Interfejs webowy (React + TypeScript, Vite, tryb dzienny/nocny):
```bash
uvicorn backend.main:app --port 8000   # backend z endpointem /api/extract/document
cd frontend/web
npm install
npm run dev                            # http://localhost:5173 (proxy /api -> :8000, zmienna VITE_API_TARGET)
```

Testy:
```bash
pytest tests/test_ocr.py
```

## Ekstrakcja tekstu i OCR (`ocr`)

Moduł do ekstrakcji surowego tekstu z wielostronicowych PDF-ów (pisma UZP / KIO):

```bash
# Uruchomienie CLI (zwraca JSON z polskimi znakami):
python -m ocr dane/przyklad_01/odwolanie.pdf

# Użycie z poziomu Pythona:
# from ocr import extract_pages
# pages = extract_pages("plik.pdf")  # {"page_1": "...", ...}
```
Więcej informacji, instrukcje instalacji Tesseracta oraz opis konfiguracji: [ocr/README.md](file:///c:/Users/pawel/Desktop/hackaton_2026/hacknation-PDF-to-JSON/ocr/README.md).

## Klasyfikacja dokumentów

Klasyfikator korzysta z reprezentacji TF‑IDF (jedno- i dwuwyrazowe cechy) oraz
LinearSVC. Dokumenty o zbyt małej pewności są klasyfikowane jako `inny`.

Po instalacji projektu wytrenuj model na katalogu `dane/`:

```bash
python scripts/classify_documents.py --train
```

Następnie klasyfikuj tekst, JSON, PDF lub cały katalog:

```bash
python scripts/classify_documents.py --text "Odwołanie do Krajowej Izby Odwoławczej"
python scripts/classify_documents.py dane/przyklad_01/odwolanie.json
python scripts/classify_documents.py dane/przyklad_01
python scripts/classify_documents.py "plik.pdf" --output wynik.json
```
