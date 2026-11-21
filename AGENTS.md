# AGENTS.md – Konfiguracja i zasady pracy nad projektem "hacknation-PDF-to-JSON”

Ten plik stanowi jedyne źródło prawdy (Single Source of Truth) dla wszystkich agentów AI i członków zespołu. Przestrzegaj poniższych zasad w każdej sesji.

---

## 1. Cel projektu i kryteria oceny
Budujemy aplikację przekształcającą wielodokumentowe sprawy odwoławcze KIO (pliki PDF) w ustrukturyzowany, kanoniczny model danych JSON.

Priorytety wynikające z oceny konkursowej:
1. **Etap 1 (Preselekcja – 100% oceny):** Jakość ekstrakcji danych (**F1 Score**) na ukrytym, **realnym** zbiorze testowym.
2. **Etap 2 (Finał):**
   - **50%** – Jakość ekstrakcji (F1 Score).
   - **30%** – Elastyczność i rozbudowa (dodawanie nowych klas dokumentów i pól **bez modyfikacji kodu źródłowego**).
   - **20%** – Architektura, modularność, przejrzystość i dokumentacja.

---

## 2. Twarde ograniczenia (Guardrails)
Łamanie poniższych zasad dyskwalifikuje rozwiązanie:
- **Wyłącznie Open Source:** Wszystkie użyte biblioteki, narzędzia OCR/PDF oraz modele językowe muszą być otwartoźródłowe.
- **On-premise bez lokalnego GPU:** Aplikacja musi działać lokalnie na samym CPU. Modele LLM mogą być wywoływane wyłącznie przez zewnętrzne/zdalne API.
- **Architektura Config-Driven:** Zakaz hardcodowania typów dokumentów, list pól czy promptów głęboko w logice biznesowej. Definicje klas dokumentów i schematy ekstrakcji muszą być sterowane konfiguracją.
- **Elastyczność technologiczna:** Nie przywiązuj logiki biznesowej do konkretnego dostawcy API, modelu LLM ani silnika OCR/PDF. Stosuj cienkie warstwy abstrakcji (interfejsy/adaptery), aby zespół mógł na bieżąco podmieniać technologie bez przepisywania pipeline'u.
- **Transparentność:** Projektuj pipeline tak, aby umożliwiał śledzenie procesu ekstrakcji (np. metadane o klasie, pewności, numerze strony lub fragmencie źródłowym – oddzielone od czystego wynikowego JSON-a zgodnego ze schematem UZP).

---

## 3. Charakterystyka danych i pułapki domenowe
- **Zbiór danych (`dane/`):** 60 syntetycznych spraw (`przyklad_01`–`przyklad_60`) z referencyjnymi plikami JSON oraz katalog `dane/negatywne/`. **Nigdy nie modyfikuj plików w katalogu `dane/`.**
- **Klasy dokumentów (zestaw bazowy):**
  1. Treść odwołania (`odwolanie`)
  2. Pełnomocnictwo (`pelnomocnictwo`)
  3. Potwierdzenie przelewu (`przelew`)
  4. **Dokumenty negatywne / spoza zakresu:** W danych występują dokumenty niezwiązane z powyższymi klasami (zarówno w `dane/negatywne/`, jak i domieszane w wybranych sprawach, np. `przyklad_04`, `10`, `24`, `37`). System musi je rozpoznawać i bezpiecznie odrzucać zamiast halucynować dane.
- **Otwartość na nowe klasy dokumentów (wymaganie projektowe):** Powyższa lista to wyłącznie klasy startowe. System musi być zaprojektowany jako uniwersalne narzędzie, w którym dodanie nowej klasy dokumentu (oraz zdefiniowanie jej schematu ekstrakcji) odbywa się **poprzez konfigurację, bez modyfikacji kodu źródłowego** (kluczowe kryterium oceny – 30% w finale).
- **Odporność na realne dokumenty (Anti-Overfitting):** Zbiór treningowy jest syntetyczny, ale testowy zawiera **prawdziwe** dokumenty (miks PDF-ów cyfrowych i skanów). Kluczowe informacje zwykle znajdują się na pierwszych 3 stronach, ale w bardzo różnym układzie. Zakaz pisania sztywnych reguł opartych wyłącznie na układzie wizualnym danych syntetycznych.

---

## 4. Standard pracy agenta (Workflow & Definition of Done)
1. **Przed dodaniem nowej biblioteki lub modelu:** Zweryfikuj i potwierdź w odpowiedzi, że komponent jest **open source** oraz **nie wymaga lokalnego GPU**.
2. **Małe, atomowe zmiany:** Nie refaktoryzuj plików spoza bieżącego zadania bez wyraźnej prośby.
3. **Weryfikacja po każdej zmianie:** Po modyfikacji logiki uruchom testy z katalogu `tests/`. Jeśli zmiana dotyczy jakości ekstrakcji lub klasyfikacji, raportuj wpływ na lokalny wynik (docelowo F1 Score na zbiorze treningowym).
4. **Spójność ze schematem:** Wynikowy JSON dla każdej klasy musi być w 100% zgodny z oczekiwanym formatem UZP (żadnych nadmiarowych kluczy w głównym obiekcie wynikowym).
5. **Uzasadnianie decyzji:** Przed każdą zmianą wyjaśnij swoje rozumowanie i uzasadnij decyzję w odpowiedzi.
6. **Zrozumienie problemu:** Zanim zaczniesz pisać kod, upewnij się że w pełni rozumiesz problem, specyfikacje i ograniczenia. Zadawaj pytania jeśli czegoś nie rozumiesz.
7. **Unikanie halucynacji:** Nigdy nie wprowadzaj w kodzie nowych klas, modeli, funkcji, zmiennych czy modułów bez wyraźnej, szczegółowej instrukcji oraz uzasadnienia.
