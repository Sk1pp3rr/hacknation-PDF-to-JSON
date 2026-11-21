"""Deterministic hint extraction, payload normalization, and grounding validation."""

from copy import deepcopy
from datetime import date
import re
from typing import Any, Iterable

from pydantic import BaseModel, Field, ValidationError

from backend.schemas import CZYNNOSC_OPIS_BY_ID, SCHEMA_REGISTRY, DocumentType

_NULL_PLACEHOLDERS = {"", "null", "none", "brak", "n/a", "-"}

_POLISH_MONTHS: dict[str, int] = {
    "stycznia": 1,
    "lutego": 2,
    "marca": 3,
    "kwietnia": 4,
    "maja": 5,
    "czerwca": 6,
    "lipca": 7,
    "sierpnia": 8,
    "września": 9,
    "wrzesnia": 9,
    "października": 10,
    "pazdziernika": 10,
    "listopada": 11,
    "grudnia": 12,
}

_NIP_RE = re.compile(r"(?i)\bNIP\b[:\s#-]*([0-9][0-9\s-]{8,14}[0-9])")
_KRS_RE = re.compile(r"(?i)\bKRS\b[:\s#-]*([0-9][0-9\s-]{8,14}[0-9])")
_REGON_RE = re.compile(r"(?i)\bREGON\b[:\s#-]*([0-9][0-9\s-]{7,16}[0-9])")
_NRB_RE = re.compile(r"(?i)(?<![0-9A-Za-z])(?:PL\s*)?(\d(?:[\s-]?\d){25})(?!\d)")
_BZP_RE = re.compile(r"(?i)\b(\d{4}\s*/\s*BZP\s*\d{8}(?:\s*/\s*\d{2})?)\b")
_TED_SHORT_RE = re.compile(r"\b(\d{6}-\d{4})\b")
_TED_OJ_RE = re.compile(r"(?i)\b(\d{4})/S\s*\d{1,3}-(\d{6})\b")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_DATE_ISO_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_DATE_DMY_RE = re.compile(r"\b(\d{1,2}\.\d{1,2}\.\d{4})\b")
_DATE_PL_RE = re.compile(
    r"(?i)\b(\d{1,2}\s+(?:stycznia|lutego|marca|kwietnia|maja|czerwca|lipca|sierpnia|wrze[sś]nia|pa[zź]dziernika|listopada|grudnia)\s+\d{4}(?:\s*(?:r\.?|roku))?)\b"
)


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _digits_only(value: Any) -> str | None:
    if value is None:
        return None
    digits = re.sub(r"\D", "", str(value))
    return digits or None


def _strip_account(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"[\s-]+", "", str(value)).upper()
    if cleaned.startswith("PL"):
        cleaned = cleaned[2:]
    digits = re.sub(r"\D", "", cleaned)
    return digits or None


def _parse_amount(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text.lower() in _NULL_PLACEHOLDERS:
        return None
    text = re.sub(r"(?i)\bPLN\b|zł", "", text)
    text = text.replace("\xa0", "").replace(" ", "").replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    return float(match.group(0))


def normalize_date_str(value: str | None) -> str | None:
    """Normalize common ISO, dot-separated, and Polish textual dates to YYYY-MM-DD."""
    if value is None:
        return None
    raw = str(value).strip()
    if raw.lower() in _NULL_PLACEHOLDERS:
        return None

    cleaned = re.sub(r"(?i)\s*(?:roku|r\.?)$", "", raw).strip()

    for pattern, order in (
        (r"^(\d{4})[-./](\d{1,2})[-./](\d{1,2})$", "ymd"),
        (r"^(\d{1,2})[-./](\d{1,2})[-./](\d{4})$", "dmy"),
    ):
        m = re.match(pattern, cleaned)
        if m:
            g1, g2, g3 = int(m.group(1)), int(m.group(2)), int(m.group(3))
            year, month, day = (g1, g2, g3) if order == "ymd" else (g3, g2, g1)
            try:
                return date(year, month, day).isoformat()
            except ValueError:
                return raw

    m_pl = re.match(r"(?i)^(\d{1,2})\s+([a-ząćęłńóśźż]+)\s+(\d{4})$", cleaned)
    if m_pl:
        day = int(m_pl.group(1))
        month = _POLISH_MONTHS.get(m_pl.group(2).lower())
        year = int(m_pl.group(3))
        if month is not None:
            try:
                return date(year, month, day).isoformat()
            except ValueError:
                return raw

    return raw


class IdentifierHints(BaseModel):
    """Deterministic regex candidates extracted from document text."""

    nip: list[str] = Field(default_factory=list)
    krs: list[str] = Field(default_factory=list)
    regon: list[str] = Field(default_factory=list)
    nrb: list[str] = Field(default_factory=list)
    bzp: list[str] = Field(default_factory=list)
    ted: list[str] = Field(default_factory=list)
    emails: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return not any(
            (self.nip, self.krs, self.regon, self.nrb, self.bzp, self.ted, self.emails, self.dates)
        )

    def format_compact(self, max_per_kind: int = 4) -> str:
        """Format non-empty candidates as compact hint lines for LLM grounding."""
        lines: list[str] = []
        for label, values in (
            ("NIP", self.nip),
            ("KRS", self.krs),
            ("REGON", self.regon),
            ("NRB", self.nrb),
            ("BZP", self.bzp),
            ("TED", self.ted),
            ("EMAIL", self.emails),
            ("DATY", self.dates),
        ):
            if values:
                lines.append(f"- {label}: {', '.join(values[:max_per_kind])}")
        return "\n".join(lines)


def extract_identifier_hints(text: str) -> IdentifierHints:
    """Extract normalized identifier and date candidates from raw document text."""
    if not text:
        return IdentifierHints()

    nips = [d for m in _NIP_RE.findall(text) if (d := _digits_only(m)) and len(d) == 10]
    krss = [d for m in _KRS_RE.findall(text) if (d := _digits_only(m)) and len(d) == 10]
    regons = [d for m in _REGON_RE.findall(text) if (d := _digits_only(m)) and len(d) in (9, 14)]
    nrbs = [d for m in _NRB_RE.findall(text) if (d := _digits_only(m)) and len(d) == 26]
    bzps = [re.sub(r"\s+", "", m).upper() for m in _BZP_RE.findall(text)]

    teds: list[str] = list(_TED_SHORT_RE.findall(text))
    for year, num in _TED_OJ_RE.findall(text):
        teds.append(f"{num}-{year}")

    emails = _EMAIL_RE.findall(text)

    raw_dates = [*_DATE_ISO_RE.findall(text), *_DATE_DMY_RE.findall(text), *_DATE_PL_RE.findall(text)]
    iso_dates = [
        norm
        for raw_d in raw_dates
        if (norm := normalize_date_str(raw_d)) and re.match(r"^\d{4}-\d{2}-\d{2}$", norm)
    ]

    return IdentifierHints(
        nip=_dedupe(nips),
        krs=_dedupe(krss),
        regon=_dedupe(regons),
        nrb=_dedupe(nrbs),
        bzp=_dedupe(bzps),
        ted=_dedupe(teds),
        emails=_dedupe(emails),
        dates=_dedupe(iso_dates),
    )


def _clean_placeholders(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _clean_placeholders(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_clean_placeholders(item) for item in node]
    if isinstance(node, str):
        stripped = node.strip()
        return None if stripped.lower() in _NULL_PLACEHOLDERS else stripped
    return node


def _normalize_bzp(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return re.sub(r"\s+", "", value).upper()


def _normalize_ted(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = re.sub(r"\s+", " ", value).strip()
    m_oj = re.match(r"(?i)^(\d{4})/S\s*\d{1,3}-(\d{6})$", cleaned)
    if m_oj:
        return f"{m_oj.group(2)}-{m_oj.group(1)}"
    return re.sub(r"\s+", "", cleaned)


def _normalize_przelew(data: dict[str, Any]) -> None:
    pb = data.get("przelew_bankowy")
    if not isinstance(pb, dict):
        return
    pb["data_operacji"] = normalize_date_str(pb.get("data_operacji"))
    pb["na_rachunek"] = _strip_account(pb.get("na_rachunek"))
    pb["z_rachunku"] = _strip_account(pb.get("z_rachunku"))
    if "kwota" in pb:
        parsed_amount = _parse_amount(pb.get("kwota"))
        if parsed_amount is not None or pb.get("kwota") is None:
            pb["kwota"] = parsed_amount


def _normalize_odwolanie(data: dict[str, Any]) -> None:
    kio = data.get("OdwolanieKIO")
    if isinstance(kio, dict):
        kio["data"] = normalize_date_str(kio.get("data"))
        kio["czynnosc_dzien"] = normalize_date_str(kio.get("czynnosc_dzien"))
        powod_id = kio.get("czynnosc_powod_id")
        if isinstance(powod_id, str) and powod_id.isdigit():
            powod_id = int(powod_id)
            kio["czynnosc_powod_id"] = powod_id
        if isinstance(powod_id, int) and powod_id in CZYNNOSC_OPIS_BY_ID:
            kio["czynnosc_opis"] = CZYNNOSC_OPIS_BY_ID[powod_id]

    for key in ("Odwolujacy", "Zamawiajacy"):
        items = data.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            for id_field in ("nip", "krs", "regon"):
                if id_field in item:
                    item[id_field] = _digits_only(item.get(id_field))

    odwolujacy = data.get("Odwolujacy")
    if isinstance(odwolujacy, list):
        for item in odwolujacy:
            if not isinstance(item, dict):
                continue
            if "przez-pelnomocnika" not in item and "przez_pelnomocnika" in item:
                item["przez-pelnomocnika"] = item.pop("przez_pelnomocnika")
            rep = item.get("pelnomocnik")
            if isinstance(rep, dict) and all(v is None for v in rep.values()):
                item["pelnomocnik"] = None
                rep = None
            has_rep = isinstance(rep, dict)
            if item.get("przez-pelnomocnika") is None:
                item["przez-pelnomocnika"] = 1 if has_rep else 0
            elif has_rep and item.get("przez-pelnomocnika") == 0:
                item["przez-pelnomocnika"] = 1

    zp = data.get("ZamowieniePubliczne")
    if isinstance(zp, dict):
        if zp.get("numer_bzp") is not None:
            zp["numer_bzp"] = _normalize_bzp(zp["numer_bzp"])
        if zp.get("numer_ted") is not None:
            zp["numer_ted"] = _normalize_ted(zp["numer_ted"])


def _normalize_pelnomocnictwo(data: dict[str, Any]) -> None:
    mocodawcy = data.get("mocodawca")
    if isinstance(mocodawcy, list):
        for entry in mocodawcy:
            if not isinstance(entry, dict):
                continue
            entity = entry.get("data")
            if isinstance(entity, dict):
                entity["nip"] = _digits_only(entity.get("nip"))
                entity["krs"] = _digits_only(entity.get("krs"))
                if entity.get("typ") == "osoba_prawna":
                    entity["imie_nazwisko"] = None

    pelnomocnicy = data.get("pelnomocnicy")
    if isinstance(pelnomocnicy, list):
        for person in pelnomocnicy:
            if isinstance(person, dict) and person.get("pesel") is not None:
                person["pesel"] = _digits_only(person["pesel"])


def normalize_payload(doc_type: DocumentType, raw: dict[str, Any]) -> dict[str, Any]:
    """Deterministically clean placeholders, identifiers, dates, and derived fields."""
    if not isinstance(raw, dict):
        return {}
    data = _clean_placeholders(deepcopy(raw))
    if not isinstance(data, dict):
        return {}

    if doc_type == DocumentType.PRZELEW:
        _normalize_przelew(data)
    elif doc_type == DocumentType.ODWOLANIE:
        _normalize_odwolanie(data)
    elif doc_type == DocumentType.PELNOMOCNICTWO:
        _normalize_pelnomocnictwo(data)

    return data


class ValidationOutcome(BaseModel):
    """Result of schema validation plus source-text grounding checks."""

    valid: bool
    data: dict[str, Any] | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _check_iso_date(field_path: str, value: str | None, errors: list[str]) -> None:
    if value is None:
        return
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", value):
        errors.append(f"{field_path}='{value}' nie ma formatu YYYY-MM-DD")
        return
    try:
        date.fromisoformat(value)
    except ValueError:
        errors.append(f"{field_path}='{value}' nie jest poprawną datą kalendarzową")


def _check_grounded_digits(
    field_path: str,
    value: Any,
    compact_source: str,
    candidates: list[str],
    errors: list[str],
) -> None:
    if value is None:
        return
    digits = _digits_only(value)
    if not digits:
        return
    if digits not in compact_source:
        hint = f"; wykryte w dokumencie: {candidates}" if candidates else ""
        errors.append(f"{field_path}='{value}' nie występuje w tekście źródłowym{hint}")


def _run_grounding_checks(
    doc_type: DocumentType,
    data: dict[str, Any],
    source_text: str,
    hints: IdentifierHints,
) -> list[str]:
    errors: list[str] = []
    compact_source = re.sub(r"[\s-]+", "", source_text).upper()

    if doc_type == DocumentType.PRZELEW:
        pb = data.get("przelew_bankowy") or {}
        _check_iso_date("przelew_bankowy.data_operacji", pb.get("data_operacji"), errors)
        _check_grounded_digits("przelew_bankowy.z_rachunku", pb.get("z_rachunku"), compact_source, hints.nrb, errors)
        _check_grounded_digits("przelew_bankowy.na_rachunek", pb.get("na_rachunek"), compact_source, hints.nrb, errors)

    elif doc_type == DocumentType.ODWOLANIE:
        kio = data.get("OdwolanieKIO") or {}
        _check_iso_date("OdwolanieKIO.data", kio.get("data"), errors)
        _check_iso_date("OdwolanieKIO.czynnosc_dzien", kio.get("czynnosc_dzien"), errors)

        for section_name in ("Odwolujacy", "Zamawiajacy"):
            for idx, entity in enumerate(data.get(section_name) or []):
                _check_grounded_digits(f"{section_name}[{idx}].nip", entity.get("nip"), compact_source, hints.nip, errors)
                _check_grounded_digits(f"{section_name}[{idx}].krs", entity.get("krs"), compact_source, hints.krs, errors)
                _check_grounded_digits(
                    f"{section_name}[{idx}].regon", entity.get("regon"), compact_source, hints.regon, errors
                )

        zp = data.get("ZamowieniePubliczne") or {}
        bzp = zp.get("numer_bzp")
        if bzp and re.sub(r"\s+", "", bzp).upper() not in compact_source:
            hint = f"; wykryte w dokumencie: {hints.bzp}" if hints.bzp else ""
            errors.append(f"ZamowieniePubliczne.numer_bzp='{bzp}' nie występuje w tekście źródłowym{hint}")

        ted = zp.get("numer_ted")
        if ted:
            m_ted = re.match(r"^(\d{6})-(\d{4})$", ted)
            grounded = (
                m_ted.group(1) in compact_source and m_ted.group(2) in compact_source
                if m_ted
                else re.sub(r"[\s-]+", "", ted).upper() in compact_source
            )
            if not grounded:
                hint = f"; wykryte w dokumencie: {hints.ted}" if hints.ted else ""
                errors.append(f"ZamowieniePubliczne.numer_ted='{ted}' nie występuje w tekście źródłowym{hint}")

    elif doc_type == DocumentType.PELNOMOCNICTWO:
        for idx, entry in enumerate(data.get("mocodawca") or []):
            entity = entry.get("data") or {}
            _check_grounded_digits(
                f"mocodawca[{idx}].data.nip", entity.get("nip"), compact_source, hints.nip, errors
            )
            _check_grounded_digits(
                f"mocodawca[{idx}].data.krs", entity.get("krs"), compact_source, hints.krs, errors
            )
        for idx, person in enumerate(data.get("pelnomocnicy") or []):
            _check_grounded_digits(f"pelnomocnicy[{idx}].pesel", person.get("pesel"), compact_source, [], errors)

    return errors


def validate_extraction(doc_type: DocumentType, raw: dict[str, Any], source_text: str) -> ValidationOutcome:
    """Normalize LLM output, validate against Pydantic UZP schema, and verify grounding."""
    normalized = normalize_payload(doc_type, raw)
    schema_cls = SCHEMA_REGISTRY.get(doc_type)
    if schema_cls is None:
        return ValidationOutcome(valid=True, data=normalized)

    try:
        model = schema_cls.model_validate(normalized)
    except ValidationError as exc:
        errors = [
            f"{'.'.join(str(part) for part in err['loc']) or 'root'}: {err['msg']}"
            for err in exc.errors()
        ]
        return ValidationOutcome(valid=False, data=normalized, errors=errors)

    dumped = model.model_dump(by_alias=True)
    hints = extract_identifier_hints(source_text)
    grounding_errors = _run_grounding_checks(doc_type, dumped, source_text, hints)
    return ValidationOutcome(
        valid=len(grounding_errors) == 0,
        data=dumped,
        errors=grounding_errors,
    )
