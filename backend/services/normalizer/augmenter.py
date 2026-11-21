"""Deterministic pre- and post-extraction augmentation (0 LLM tokens)."""

from copy import deepcopy
import re
from typing import Any

from backend.core.models import DocumentType
from backend.services.normalizer.validator import extract_identifier_hints, normalize_date_str

_HEADER_CITY_DATE_RE = re.compile(
    r"^(?P<city>[A-ZĄĆĘŁŃÓŚŹŻ][A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż\s\-]{1,40}?)"
    r"(?:,\s*(?:(?i:dnia)\s+)?|\s+(?:(?i:dnia)\s+)?)"
    r"(?P<date>(?:\d{1,2}\.\d{1,2}\.\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}\s+"
    r"(?i:stycznia|lutego|marca|kwietnia|maja|czerwca|lipca|sierpnia|"
    r"wrze[sś]nia|pa[zź]dziernika|listopada|grudnia)\s+\d{4})"
    r"(?:\s*(?i:r\.?|roku))?)$"
)

_NON_CITY_PREFIX_RE = re.compile(r"(?i)\b(?:ul|al|pl|art|ust|nr)\.")

_KIO_ADDRESSEE_LINE_RE = re.compile(
    r"(?im)^(?:\s*Prezes\s+Krajowej\s+Izby\s+Odwo[łl]awczej\s*$|"
    r"\s*ul\.\s*Post[ęe]pu\s+17\s*[Aa]?\s*$|"
    r"\s*02-676\s+Warszawa\s*$)"
)

_LEADING_TITLE_RE = re.compile(
    r"(?i)^(?:r\.\s*pr\.|radca\s+prawny|adw\.|adwokat|mec\.|mgr|dr|prof\.)\s+"
)
_TRAILING_TITLE_RE = re.compile(
    r"(?i),?\s*(?:r\.\s*pr\.|radca\s+prawny|adw\.|adwokat|mec\.)$"
)


def extract_header_city_and_date(text: str) -> tuple[str | None, str | None]:
    """Deterministically parse the top-of-page '<City>, [dnia] <Date>' line from an appeal."""
    lines = [line.strip() for line in text.splitlines() if line.strip()][:45]
    for line in lines:
        m = _HEADER_CITY_DATE_RE.match(line)
        if not m:
            continue
        city = m.group("city").strip(" ,")
        if any(ch.isdigit() for ch in city) or _NON_CITY_PREFIX_RE.search(city):
            continue
        iso_date = normalize_date_str(m.group("date"))
        if iso_date and re.match(r"^\d{4}-\d{2}-\d{2}$", iso_date):
            return city, iso_date
    return None, None


def strip_kio_addressee(text: str) -> str:
    """Remove boilerplate KIO addressee lines so small models do not confuse Warsaw with miejscowosc."""
    kept = [line for line in text.splitlines() if not _KIO_ADDRESSEE_LINE_RE.match(line)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def annotate_odwolanie_header(header_slice: str) -> str:
    """Strip KIO boilerplate, insert clear section markers, and prepend deterministic header summary."""
    cleaned = strip_kio_addressee(header_slice)

    for pattern, marker in (
        (r"(?im)^(\s*Odwo[łl]uj[ąa]cy\s*(?:\([^)]*\))?\s*:)", "\n<<< SEKCJA: ODWOŁUJĄCY >>>\n\\1"),
        (r"(?im)^(\s*Zamawiaj[ąa]cy\s*(?:\([^)]*\))?\s*:)", "\n<<< SEKCJA: ZAMAWIAJĄCY >>>\n\\1"),
        (r"(?im)^(\s*O\s*D\s*W\s*O\s*[ŁL]\s*A\s*N\s*I\s*E\s*$)", "\n<<< SEKCJA: DANE ZAMÓWIENIA >>>\n\\1"),
        (r"(?im)^(\s*dotyczy\s*:)", "\n<<< SEKCJA: DANE ZAMÓWIENIA >>>\n\\1"),
    ):
        cleaned = re.sub(pattern, marker, cleaned, count=1)

    city, iso_date = extract_header_city_and_date(header_slice)
    if city and iso_date:
        banner = f"[NAGŁÓWEK: miejscowosc={city}, data={iso_date}]"
    else:
        banner = "[BRAK DATY I MIEJSCOWOŚCI W NAGŁÓWKU -> data=null, miejscowosc=null]"

    return f"{banner}\n{cleaned}".strip()


def strip_person_titles(name: str | None) -> str | None:
    """Strip Polish legal/academic titles from a person's full name."""
    if not name or not isinstance(name, str):
        return name
    cleaned = name.strip()
    while True:
        updated = _LEADING_TITLE_RE.sub("", cleaned).strip()
        updated = _TRAILING_TITLE_RE.sub("", updated).strip(" ,")
        if updated == cleaned:
            break
        cleaned = updated
    return cleaned or None


def _clean_single_representative(name: str | None) -> str | None:
    cleaned = strip_person_titles(name)
    if cleaned and "," in cleaned:
        first = cleaned.split(",", 1)[0].strip()
        return strip_person_titles(first)
    return cleaned


def _entity_keys(entity: dict[str, Any]) -> tuple[set[str], set[str], set[str]]:
    nips = {re.sub(r"\D", "", str(entity["nip"]))} if entity.get("nip") else set()
    krss = {re.sub(r"\D", "", str(entity["krs"]))} if entity.get("krs") else set()
    names = {str(entity["nazwa"]).strip().lower()} if entity.get("nazwa") else set()
    return nips, krss, names


def _matches_any_key(entity: dict[str, Any], nips: set[str], krss: set[str], names: set[str]) -> bool:
    e_nip = re.sub(r"\D", "", str(entity["nip"])) if entity.get("nip") else ""
    e_krs = re.sub(r"\D", "", str(entity["krs"])) if entity.get("krs") else ""
    e_name = str(entity["nazwa"]).strip().lower() if entity.get("nazwa") else ""
    return bool((e_nip and e_nip in nips) or (e_krs and e_krs in krss) or (e_name and e_name in names))


def _dedupe_entities(odwolujacy: list[Any], zamawiajacy: list[Any]) -> tuple[list[Any], list[Any]]:
    if len(odwolujacy) > 1 and zamawiajacy:
        zam_nips: set[str] = set()
        zam_krss: set[str] = set()
        zam_names: set[str] = set()
        for item in zamawiajacy:
            if isinstance(item, dict):
                n, k, nm = _entity_keys(item)
                zam_nips |= n
                zam_krss |= k
                zam_names |= nm
        filtered = [
            item
            for item in odwolujacy
            if not (isinstance(item, dict) and _matches_any_key(item, zam_nips, zam_krss, zam_names))
        ]
        if filtered:
            odwolujacy = filtered

    if len(zamawiajacy) > 1 and odwolujacy and isinstance(odwolujacy[0], dict):
        o_nips, o_krss, o_names = _entity_keys(odwolujacy[0])
        filtered_zam = [
            item
            for item in zamawiajacy
            if not (isinstance(item, dict) and _matches_any_key(item, o_nips, o_krss, o_names))
        ]
        if filtered_zam:
            zamawiajacy = filtered_zam

    return odwolujacy, zamawiajacy


def _augment_odwolanie(data: dict[str, Any], source_text: str) -> None:
    odwolujacy = data.get("Odwolujacy")
    if isinstance(odwolujacy, list):
        for item in odwolujacy:
            if not isinstance(item, dict):
                continue
            rep = item.get("pelnomocnik")
            if isinstance(rep, dict) and isinstance(rep.get("imie_nazwisko"), str):
                rep["imie_nazwisko"] = _clean_single_representative(rep["imie_nazwisko"])

    zamawiajacy = data.get("Zamawiajacy")
    if isinstance(odwolujacy, list) and isinstance(zamawiajacy, list):
        odwolujacy, zamawiajacy = _dedupe_entities(odwolujacy, zamawiajacy)
        data["Odwolujacy"] = odwolujacy
        data["Zamawiajacy"] = zamawiajacy

    if isinstance(zamawiajacy, list) and len(zamawiajacy) == 1 and isinstance(zamawiajacy[0], dict):
        zamawiajacy[0]["prowadzacy_postepowanie"] = None

    kio = data.get("OdwolanieKIO")
    if isinstance(kio, dict):
        city, iso_date = extract_header_city_and_date(source_text)
        if city and iso_date:
            kio["miejscowosc"] = city
            kio["data"] = iso_date
        else:
            if kio.get("data") is not None and kio.get("data") == kio.get("czynnosc_dzien"):
                kio["data"] = None
            if kio.get("miejscowosc") == "Warszawa" and "02-676" in source_text:
                kio["miejscowosc"] = None

    zp = data.get("ZamowieniePubliczne")
    if isinstance(zp, dict):
        hints = extract_identifier_hints(source_text)
        if zp.get("numer_bzp") is None and len(hints.bzp) == 1:
            zp["numer_bzp"] = hints.bzp[0]
        if zp.get("numer_ted") is None and len(hints.ted) == 1:
            zp["numer_ted"] = hints.ted[0]


def _augment_pelnomocnictwo(data: dict[str, Any], source_text: str) -> None:
    mocodawcy = data.get("mocodawca")
    if isinstance(mocodawcy, list):
        for entry in mocodawcy:
            if not isinstance(entry, dict):
                continue
            for signer in entry.get("podpisujacy") or ():
                if isinstance(signer, dict) and isinstance(signer.get("imie_nazwisko"), str):
                    signer["imie_nazwisko"] = strip_person_titles(signer["imie_nazwisko"]) or signer["imie_nazwisko"]

    hints = extract_identifier_hints(source_text)
    pelnomocnicy = data.get("pelnomocnicy")
    if isinstance(pelnomocnicy, list):
        for person in pelnomocnicy:
            if not isinstance(person, dict):
                continue
            if isinstance(person.get("imie_nazwisko"), str):
                person["imie_nazwisko"] = strip_person_titles(person["imie_nazwisko"]) or person["imie_nazwisko"]
            pesel = person.get("pesel")
            if pesel is not None:
                digits = re.sub(r"\D", "", str(pesel))
                if len(digits) != 11 and not hints.pesel:
                    person["pesel"] = None


def augment_post_extraction(doc_type: DocumentType | str, data: dict[str, Any], source_text: str) -> dict[str, Any]:
    """Apply deterministic, zero-token domain augmentation after LLM extraction."""
    if not isinstance(data, dict):
        return {}
    augmented = deepcopy(data)

    if str(doc_type) == DocumentType.ODWOLANIE.value:
        _augment_odwolanie(augmented, source_text)
    elif str(doc_type) == DocumentType.PELNOMOCNICTWO.value:
        _augment_pelnomocnictwo(augmented, source_text)

    return augmented
