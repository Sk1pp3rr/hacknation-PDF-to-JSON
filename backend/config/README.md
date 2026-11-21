# Moduł Konfiguracji (Config-Driven Architecture)

Katalog ten stanowi serce architektury **Config-Driven**, realizującej wymóg elastyczności (30% oceny w etapie finałowym konkursu: dodawanie nowych klas dokumentów i pól bez modyfikacji kodu źródłowego).

---

## Docelowa zawartość katalogu

```text
backend/config/
├── README.md               # Niniejsza dokumentacja
├── settings.py             # Zmienne środowiskowe aplikacji (BaseSettings)
├── loader.py               # Dynamiczny czytnik i walidator plików YAML
└── documents/              # Samodzielne definicje klas dokumentów (YAML)
    ├── odwolanie.yaml      # Profil dla klasy: odwołanie do KIO
    ├── pelnomocnictwo.yaml # Profil dla klasy: pełnomocnictwo
    └── przelew.yaml        # Profil dla klasy: potwierdzenie wpłaty / przelew
```

---

## 1. `settings.py` (Konfiguracja środowiska i API)
Klasa oparta na `pydantic-settings` zarządzająca zmiennymi z pliku `.env` oraz środowiska systemowego:
* **LLM Provider:** `LLM_API_KEY`, `LLM_BASE_URL` (np. endpointy modeli otwartoźródłowych Mistral/vLLM/Scaleway), `LLM_MODEL`, `LLM_TEMPERATURE`, `LLM_TIMEOUT`.
* **Aplikacja:** `APP_NAME`, `DEBUG`, `LOG_LEVEL`.
* **Ścieżki:** `DOCUMENTS_CONFIG_DIR` (domyślnie wskazująca na `backend/config/documents`), `MODELS_DIR`.

---

## 2. `loader.py` (Automatyczny rejestrator klas dokumentów)
Moduł w Pythonie odpowiedzialny za dynamiczne skanowanie podkatalogu `documents/`:
* Automatycznie wykrywa wszystkie pliki `*.yaml` w `documents/`.
* Waliduje ich strukturę za pomocą modelu Pydantic (`DocumentConfig`).
* Udostępnia metody pobierania konfiguracji:
  * `get_all_document_configs() -> dict[str, DocumentConfig]`
  * `get_document_config(doc_type: str) -> DocumentConfig`
* **Kluczowy atut:** Dodanie nowego pliku YAML (np. `wybor_oferty.yaml`) sprawia, że cały system natychmiast potrafi klasyfikować i ekstrahować nowy typ dokumentu bez zmian w kodzie Pythona.

---

## 3. Podkatalog `documents/` (Profile klas dokumentów)
Każdy plik YAML definiuje kompletną wiedzę domenową o danym typie dokumentu dla wszystkich 4 etapów pipeline'u:

### Przykład struktury pliku YAML (`documents/<klasa>.yaml`):
```yaml
id: "odwolanie"
nazwa: "Treść odwołania do KIO"
opis: "Pismo odwoławcze kierowane do Prezesa Krajowej Izby Odwoławczej"

# Krok 2: Konfiguracja klasyfikatora
klasyfikacja:
  filtry_nazw_plikow:
    - "odwolanie"
    - "odwolanie.pdf"
    - "odwolanie.json"
  frazy_kluczowe:
    - "odwołanie"
    - "krajowa izba odwoławcza"
    - "czynność zamawiającego"
    - "odwołujący"
  prog_pewnosci: 0.60

# Krok 3: Konfiguracja promptu LLM
ekstrakcja:
  schema_name: "OdwolanieSchema"
  system_prompt: >
    Jesteś ekspertem prawa zamówień publicznych. Dokonaj ekstrakcji danych z odwołania do KIO.
  wskazowki:
    - "Data odwołania znajduje się w nagłówku pisma na 1. stronie."
    - "Przypisz czynnosc_powod_id w przedziale 1-12 zgodnie ze słownikiem."

# Krok 4: Reguły normalizacji pól
normalizacja:
  wymagane_pola:
    - "OdwolanieKIO.czynnosc_powod_id"
    - "Odwolujacy"
    - "Zamawiajacy"
  formaty:
    nip: "cyfry_10"
    krs: "cyfry_10"
    daty: "YYYY-MM-DD"
```
