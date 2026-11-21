# Odwołania2json (hacknation-PDF-to-JSON)

Aplikacja webowa do automatycznej ekstrakcji danych ze spraw odwoławczych (odwołania, pełnomocnictwa, dowody wpłaty) z plików PDF do kanonicznego modelu JSON zgodnego ze specyfikacją UZP.

Projekt oparty na komponentach Open Source oraz europejskim API LLM (np. Mistral AI / Scaleway UE).

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
├── frontend/
│   ├── web/                         # Interfejs webowy React + TypeScript (Vite, tryb dzień/noc, Eksplorator PDF)
│   └── app.py                       # Pomocniczy interfejs Streamlit
├── dane/
│   ├── negatywne/                   # Próbki negatywne (spoza zakresu)
│   ├── skany/                       # Próbki zeskanowanych dokumentów PDF
│   └── przyklad_01/...              # 60 spraw treningowych z referencyjnymi plikami JSON
├── scripts/
│   ├── download_hacknation_data.py  # Pobieranie danych źródłowych
│   ├── classify_documents.py        # Trenowanie i uruchamianie klasyfikatora
│   ├── test_pipeline.py             # Test pojedynczego pliku PDF (CLI)
│   └── benchmark.py                 # Benchmark przepustowości (batch równoległy)
├── tests/                           # Testy jednostkowe i integracyjne (pytest)
├── .env.example                     # Przykładowa konfiguracja zmiennych środowiskowych
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

## Szybki start (jak uruchomić aplikację lokalnie)

### 1. Środowisko Pythona i zależności
```bash
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

pip install -e ".[dev]"
pip install pypdfium2 pytesseract
cp .env.example .env
```

### 2. Konfiguracja modelu LLM w pliku `.env`

**Wariant A – lokalny model na CPU przez Ollama (np. polski Bielik-1.5B):**
```bash
ollama pull hf.co/speakleash/Bielik-1.5B-v3.0-Instruct-GGUF:Q8_0
```
W pliku `.env` ustaw:
```env
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=hf.co/speakleash/Bielik-1.5B-v3.0-Instruct-GGUF:Q8_0
LLM_TIMEOUT_SECONDS=180
LLM_EXTRACTION_MODE=section_swarm
```

**Wariant B – zdalne europejskie API (np. Mistral AI / Scaleway):**
```env
LLM_API_KEY=twoj_klucz_api
LLM_BASE_URL=https://api.mistral.ai/v1
LLM_MODEL=ministral-3b-latest
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
- W górnym pasku możesz przełączać **tryb dzienny/nocny** oraz liczbę dokumentów przetwarzanych **równolegle (`1 / 2 / 4 / 8`)**.
- W zakładce **„Cytaty i Eksplorator PDF”** kliknięcie dowolnego wyekstrahowanego pola przewija stronę wbudowanego eksploratora PDF i podświetla na niej dokładny fragment źródłowy.

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
