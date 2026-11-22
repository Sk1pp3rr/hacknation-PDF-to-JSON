# Odwołania2json (hacknation-PDF-to-JSON)

Aplikacja webowa i wsadowe narzędzie CLI do automatycznej ekstrakcji danych ze spraw odwoławczych KIO (`odwolanie`, `pelnomocnictwo`, `przelew` oraz bezpieczne odrzucanie dokumentów spoza zakresu `inny -> {}`) z plików PDF do kanonicznego modelu JSON zgodnego ze specyfikacją UZP.

Projekt oparty w 100% na komponentach **Open Source** działających lokalnie na **CPU** oraz europejskim/zewnętrznym API LLM kompatybilnym ze standardem OpenAI.

## Architektura i przepływ przetwarzania

```text
PDF (cyfrowy lub skan)
   │
   ▼
[1. Parser & OCR]           pypdf (warstwa tekstowa) + pypdfium2 + Tesseract (`pol`) na CPU
   │
   ▼
[2. Klasyfikator 4-klasowy] TF-IDF + LinearSVC + reguły z YAML (`odwolanie`, `pelnomocnictwo`, `przelew`, `inny`)
   │
   ├── inny ──────────────────────────────────────────────► `{}` (bezpieczne odrzucenie bez halucynacji)
   │
   ▼
[3. Ekstraktor LLM]         OpenAI-compatible API, strict JSON Schema, `section_swarm`, surgical retry
   │
   ▼
[4. Normalizator & Walidator] Sumy kontrolne (IBAN, NIP, PESEL, REGON), korekty OCR, formaty dat/kwot
   │
   ▼
[5. Wynik]                  Czysty JSON UZP (`data`) + osobne metadane audytowe (`meta`: cytaty, bbox, pewność, tokeny)
```

### Kluczowe założenia projektowe
- **100% Open Source & CPU-only:** `FastAPI`, `pypdf`, `pypdfium2`, `pytesseract` + `tesseract-ocr-pol`, `scikit-learn`, `pydantic`, `httpx`, `React` + `Vite` (bez lokalnego GPU).
- **Config-Driven (bez hardcodowania):** definicje klas dokumentów, frazy klasyfikacyjne, podpowiedzi promptów i reguły normalizacji są ładowane dynamicznie z plików YAML w [`backend/config/documents/`](backend/config/documents/).
- **Niezależność od dostawcy (Adaptery):** zmiana dostawcy lub modelu LLM odbywa się wyłącznie przez `.env` (`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`), a silnik OCR implementuje abstrakcję `OcrEngine`.
- **Separacja danych UZP i metadanych:** wynikowy plik `.json` zawiera wyłącznie dozwolone pola schematu UZP; ślad audytowy (klasa, pewność, cytaty źródłowe, współrzędne na stronie PDF, błędy/ostrzeżenia walidacji, czas i tokeny) jest oddzielony w sekcji `meta`.

## Struktura repozytorium

```text
├── backend/
│   ├── main.py                      # Aplikacja FastAPI (/health, /api/extract/document, /api/extract/batch)
│   ├── config/
│   │   ├── settings.py              # Konfiguracja ze zmiennych środowiskowych (.env)
│   │   ├── loader.py                # Dynamiczne ładowanie profili YAML
│   │   ├── prompts.yaml             # Szablony promptów naprawczych
│   │   └── documents/               # Profile klas dokumentów (odwolanie, pelnomocnictwo, przelew)
│   ├── core/models.py               # Modele domenowe pipeline'u
│   ├── schemas/uzp.py               # Kanoniczne modele Pydantic zgodne ze schematem UZP
│   └── services/
│       ├── parser/                  # Parsowanie PDF per strona (pypdf), Eksplorator PDF (pypdfium2) i adapter OCR
│       ├── classifier/              # Klasyfikator TF-IDF + LinearSVC oraz reguły zapasowe
│       ├── normalizer/              # Walidator sum kontrolnych, deterministyczny augmenter i lokalizator cytatów
│       └── extractor/               # Klient OpenAI-compatible, strict JSON Schema, section_swarm i surgical retry
├── ocr/                             # Samodzielny moduł OCR (pypdfium2 + Tesseract pol)
│   ├── __init__.py                  # Eksportuje extract_pages(pdf_path)
│   ├── engine.py                    # OcrEngine (ABC) + TesseractEngine
│   ├── pdf.py                       # Warstwa tekstowa, renderowanie, extract_pages
│   ├── __main__.py                  # CLI: python -m ocr <plik.pdf>
│   └── README.md                    # Dokumentacja modułu OCR, instalacja, licencje
├── frontend/
│   ├── web/                         # Interfejs webowy React + TypeScript (Vite, tryb dzień/noc, Eksplorator PDF)
│   └── app.py                       # Pomocniczy interfejs Streamlit
├── dane/
│   ├── negatywne/                   # Próbki negatywne (spoza zakresu)
│   ├── skany/                       # Próbki zeskanowanych dokumentów PDF
│   └── przyklad_01/...              # 60 spraw treningowych z referencyjnymi plikami JSON
├── scripts/
│   ├── download_hacknation_data.py  # Pobieranie danych źródłowych
│   ├── fetch_uzp_negatives.py       # Pobieranie prawdziwych orzeczeń KIO z UZP jako negatywów (`inny`)
│   ├── classify_documents.py        # Trenowanie i uruchamianie klasyfikatora
│   ├── process_batch.py             # Przetwarzanie wsadowe katalogu input/ -> output/
│   ├── test_pipeline.py             # Test pojedynczego pliku PDF (CLI)
│   └── benchmark.py                 # Benchmark przepustowości (batch równoległy)
├── tests/                           # Testy jednostkowe i integracyjne (pytest)
├── Dockerfile                       # Obraz CPU-only (Python + Tesseract pol + zbudowany frontend React)
├── docker-compose.yml               # Usługi `app` (GUI + API) oraz `processor` (wsadowe CLI)
├── RUN.md                           # Instrukcja bezobsługowego uruchomienia na serwerze (/opt/solution)
└── Uruchom_Aplikacje.bat            # Jednoklikowy launcher GUI dla użytkownika końcowego (Windows)
├── .env.example                     # Przykładowa konfiguracja zmiennych środowiskowych
├── requirements.txt                 # Lista zależności Pythona (CPU-only, Open Source)
└── pyproject.toml                   # Zależności i konfiguracja projektu
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

## Uruchomienie produkcyjne (Docker — GUI + batch CLI)

Najprostszy sposób uruchomienia dla użytkownika końcowego oraz automatycznej oceny wsadowej opisany jest w [RUN.md](RUN.md):

```bash
cp .env.example .env
# Uzupełnij LLM_API_KEY, LLM_BASE_URL i LLM_MODEL w .env

docker compose build
docker compose up -d app          # Aplikacja webowa (GUI + API) na portach 80 i 8000
docker compose run --rm processor # Bezobsługowe przetwarzanie input/ -> output/
```

Dla użytkownika końcowego na Windows (bez terminala) wystarczy podwójne kliknięcie w [`Uruchom_Aplikacje.bat`](Uruchom_Aplikacje.bat) — skrypt uruchamia lokalny serwer i otwiera interfejs webowy w przeglądarce (`http://127.0.0.1:8000`).

### Endpointy HTTP (FastAPI)

| Metoda i ścieżka | Opis |
| --- | --- |
| `GET /health` | Status usługi (`{"status": "ok"}`) używany m.in. przez `healthcheck` Dockera |
| `POST /api/extract` | Szybka ekstrakcja i klasyfikacja przesłanego pliku PDF (`ExtractionResult`) |
| `POST /api/extract/document` | Pełne przetwarzanie pojedynczego PDF z podglądem stron i cytatami (`meta.evidence`, `meta.page_images`) |
| `POST /api/extract/batch` | Przetwarzanie wsadowe wielu plików PDF w jednym żądaniu |

## Szybki start lokalny (tryb developerski)

### 1. Środowisko Pythona i zależności
```bash
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
pip install -e ".[dev]"
cp .env.example .env
```

### 2. Konfiguracja modelu LLM w pliku `.env`

Aplikacja korzysta ze zdalnego API kompatybilnego z OpenAI (np. europejskie API Mistral AI / Scaleway / OVHcloud AI Endpoints, model Open Source):
```env
LLM_API_KEY=twoj_klucz_api
LLM_BASE_URL=https://api.mistral.ai/v1
LLM_MODEL=ministral-8b-latest
LLM_TIMEOUT_SECONDS=30
LLM_EXTRACTION_MODE=section_swarm
```

### 3. Uruchomienie Backendu (FastAPI) i GUI (React + TypeScript)

Otwórz **dwa terminale**:

**Terminal 1 – Backend (port `8000`):**
```bash
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --port 8000 --reload
```

**Terminal 2 – GUI React + TypeScript (port `5173`):**
```bash
cd frontend/web
npm install
npm run dev
```

Następnie otwórz w przeglądarce: **http://localhost:5173**
- Przeciągnij jeden lub wiele plików PDF (np. z `dane/przyklad_01/odwolanie.pdf`).
- W górnym pasku możesz przełączać **tryb dzienny/nocny**; równoległość przetwarzania kolejki dobiera się automatycznie do liczby wrzuconych plików.
- W zakładce **„Cytaty i Eksplorator PDF”** kliknięcie pola podświetla fragment w dokumencie PDF (z przybliżaniem i przesuwaniem myszką) oraz pozwala od razu poprawić wartość inline.
- W zakładce **„Edytuj pola”** możesz wygodnie poprawić wszystkie pola w formularzu (lub w surowym JSON-ie) i pobrać wynik pojedynczo albo hurtowo jako **ZIP (`wyniki_json.zip`)**.

### 4. Uruchamianie z linii komend (CLI) i testy

Pojedynczy dokument PDF (z porównaniem do pliku referencyjnego JSON):
```bash
python scripts/test_pipeline.py dane/przyklad_01/odwolanie.pdf --compare dane/przyklad_01/odwolanie.json
# Tryb bez wywołania LLM (samo parsowanie PDF + klasyfikator + podpowiedzi regex):
python scripts/test_pipeline.py dane/przyklad_01/odwolanie.pdf --dry-run
```

Benchmark przepustowości (wiele dokumentów równolegle):
```bash
python scripts/benchmark.py dane --limit 10 --concurrency 4
```

Testy jednostkowe:
```bash
pytest -v
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
Więcej informacji, instrukcje instalacji Tesseracta oraz opis konfiguracji: [ocr/README.md](ocr/README.md).

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

## Dodawanie nowej klasy dokumentu (bez modyfikacji kodu)

System jest w pełni sterowany konfiguracją YAML ([backend/config/documents/](backend/config/documents/)). Aby dodać obsługę nowej klasy dokumentu (np. `wezwanie`), wystarczy utworzyć plik `backend/config/documents/wezwanie.yaml`:

```yaml
id: "wezwanie"
nazwa: "Wezwanie do uzupełnienia braków"
klasyfikacja:
  filtry_nazw_plikow: ["wezwanie", "wezwanie.pdf"]
  frazy_kluczowe: ["wezwanie do uzupełnienia", "wzywa wykonawcę", "braki formalne"]
  prog_pewnosci: 0.62
ekstrakcja:
  max_tokens: 300
  max_chars: 6000
  system_prompt: "Ekstrahujesz dane z wezwania do ścisłego JSON."
  json_schema:
    type: object
    additionalProperties: false
    required: ["sygnatura", "termin_dni"]
    properties:
      sygnatura: { type: ["string", "null"] }
      termin_dni: { type: ["integer", "null"] }
```

Nowa klasa jest od razu rozpoznawana przez klasyfikator (`classify_document`) oraz ekstraktor LLM (`extract_document`) bez zmian w kodzie Pythona.
