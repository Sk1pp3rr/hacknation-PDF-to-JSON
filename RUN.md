# Uruchomienie rozwiązania (`/opt/solution/RUN.md`)

## Komenda uruchamiająca przetwarzanie wsadowe (`input/` -> `output/`)

Uruchom z katalogu `/opt/solution`:

```bash
docker compose run --rm processor
```

Komenda działa całkowicie bezobsługowo (bez interakcji z użytkownikiem), przetwarza rekurencyjnie wszystkie pliki PDF z `/opt/solution/input` (warstwa tekstowa `pypdf` + automatyczny OCR `tesseract-ocr-pol` dla skanów, klasyfikacja 4-klasowa z odrzucaniem dokumentów negatywnych `inny` -> `{}`, ekstrakcja przez zdalne API LLM oraz walidacja/normalizacja schematu UZP) i zapisuje wynikowe pliki `.json` z zachowaniem struktury katalogów w `/opt/solution/output`. Po zakończeniu wszystkich plików kontener kończy działanie kodem `0`.

## Wykorzystywane usługi zewnętrzne i konfiguracja (`.env`)

- **Zewnętrzne API LLM (EU/EOG, model Open Source `<= 8B`):** konfigurowane w pliku `/opt/solution/.env` poprzez `LLM_API_KEY`, `LLM_BASE_URL` oraz `LLM_MODEL` (np. OVHcloud AI Endpoints `Mistral-7B-Instruct-v0.3` lub Mistral AI `ministral-8b-latest` / `ministral-3b-latest`).
- **OCR i parsowanie PDF:** działa w 100% lokalnie na CPU wewnątrz obrazu Dockera (`tesseract-ocr` + `tesseract-ocr-pol`, `pypdf`, `pypdfium2`).
- **Opcjonalny podgląd webowy (GUI + API):** działa w tle jako usługa `app` (`restart: unless-stopped`) na portach `80` i `8000`.
