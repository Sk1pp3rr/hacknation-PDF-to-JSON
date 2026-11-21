"""Deterministic hint extraction, payload normalization, and grounding validation."""

from copy import deepcopy
from datetime import date
import re
from typing import Any, Callable, Iterable

from pydantic import BaseModel, Field, ValidationError

from backend.config import settings
from backend.core.models import DocumentType
from backend.schemas.uzp import CZYNNOSC_OPIS_BY_ID, SCHEMA_REGISTRY

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
_PESEL_LABELED_RE = re.compile(r"(?i)\bPESEL\b[:\s#-]*([0-9][0-9\s-]{9,15}[0-9])")
_PESEL_STANDALONE_RE = re.compile(r"(?<!\d)(\d{11})(?!\d)")
_NRB_RE = re.compile(r"(?i)(?<![0-9A-Za-z])(?:PL\s*)?(\d(?:[\s-]?\d){25})(?!\d)")
_IBAN_TOKEN_RE = re.compile(
    r"(?<![0-9A-Za-z])(?:(?:PL\s*)?\d(?:[\s-]?\d){25}|[A-Z]{2}\d{2}(?:[\s-]?[0-9A-Z]){11,30})(?![0-9A-Za-z])"
)
_BZP_RE = re.compile(r"(?i)\b(\d{4}\s*/\s*BZP\s*\d{8}(?:\s*/\s*\d{2})?)\b")
_TED_SHORT_RE = re.compile(r"\b(\d{6}-\d{4})\b")
_TED_OJ_RE = re.compile(r"(?i)\b(\d{4})/S\s*\d{1,3}-(\d{6})\b")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_DATE_ISO_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_DATE_DMY_RE = re.compile(r"\b(\d{1,2}[.-]\d{1,2}[.-]\d{4})\b")
_DATE_PL_RE = re.compile(
    r"(?i)\b(\d{1,2}\s+(?:stycznia|lutego|marca|kwietnia|maja|czerwca|lipca|sierpnia|wrze[sś]nia|pa[zź]dziernika|listopada|grudnia)\s+\d{4}(?:\s*(?:r\.?|roku))?)\b"
)


def _coerce_doc_type(doc_type: DocumentType | str) -> DocumentType | None:
    if isinstance(doc_type, DocumentType):
        return doc_type
    try:
        return DocumentType(str(doc_type))
    except ValueError:
        return None


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
    if not cleaned or cleaned.lower() in _NULL_PLACEHOLDERS:
        return None
    if re.match(r"^[A-Z]{2}\d{2}[0-9A-Z]{11,30}$", cleaned):
        return cleaned
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


def is_valid_nip(value: str) -> bool:
    """Validate 10-digit Polish NIP check digit."""
    s = re.sub(r"\D", "", str(value))
    if len(s) != 10:
        return False
    weights = (6, 5, 7, 2, 3, 4, 5, 6, 7)
    checksum = sum(int(s[i]) * weights[i] for i in range(9)) % 11
    return checksum == int(s[9])


def is_valid_regon(value: str) -> bool:
    """Validate 9- or 14-digit Polish REGON check digit."""
    s = re.sub(r"\D", "", str(value))
    if len(s) == 9:
        weights = (8, 9, 2, 3, 4, 5, 6, 7)
    elif len(s) == 14:
        weights = (2, 4, 8, 5, 0, 9, 7, 3, 6, 1, 2, 4, 8)
    else:
        return False
    checksum = sum(int(s[i]) * weights[i] for i in range(len(weights))) % 11
    return (0 if checksum == 10 else checksum) == int(s[-1])


def is_valid_pesel(value: str) -> bool:
    """Validate 11-digit Polish PESEL check digit."""
    s = re.sub(r"\D", "", str(value))
    if len(s) != 11:
        return False
    weights = (1, 3, 7, 9, 1, 3, 7, 9, 1, 3)
    checksum = (10 - (sum(int(s[i]) * weights[i] for i in range(10)) % 10)) % 10
    return checksum == int(s[10])


def is_valid_nrb(value: str) -> bool:
    """Validate 26-digit Polish NRB or international ISO 13616 IBAN check digits (mod 97 == 1)."""
    s = _strip_account(value) or ""
    if s.isdigit():
        if len(s) != 26:
            return False
        rearranged = s[2:] + "2521" + s[:2]
        return int(rearranged) % 97 == 1
    if re.match(r"^[A-Z]{2}\d{2}[0-9A-Z]{11,30}$", s):
        moved = s[4:] + s[:4]
        numeric = "".join(str(ord(ch) - 55) if ch.isalpha() else ch for ch in moved)
        return int(numeric) % 97 == 1
    return False


def extract_accounts_in_order(text: str) -> list[str]:
    """Extract Polish NRBs/IBANs (preserving PL when present) and valid foreign IBANs in document order."""
    accounts: list[str] = []
    for m in _IBAN_TOKEN_RE.finditer(text or ""):
        norm = _strip_account(m.group(0))
        if not norm:
            continue
        is_pl_with_prefix = norm.startswith("PL") and len(norm) == 28 and norm[2:].isdigit()
        is_pl_digits = norm.isdigit() and len(norm) == 26
        if is_pl_with_prefix or is_pl_digits or (not norm.isdigit() and is_valid_nrb(norm)):
            accounts.append(norm)
    return _dedupe(accounts)


def is_valid_krs(value: str) -> bool:
    """Validate 10-digit KRS format."""
    s = re.sub(r"\D", "", str(value))
    return len(s) == 10


class IdentifierHints(BaseModel):
    """Deterministic regex candidates extracted from document text."""

    nip: list[str] = Field(default_factory=list)
    krs: list[str] = Field(default_factory=list)
    regon: list[str] = Field(default_factory=list)
    pesel: list[str] = Field(default_factory=list)
    nrb: list[str] = Field(default_factory=list)
    bzp: list[str] = Field(default_factory=list)
    ted: list[str] = Field(default_factory=list)
    emails: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return not any(
            (self.nip, self.krs, self.regon, self.pesel, self.nrb, self.bzp, self.ted, self.emails, self.dates)
        )

    def format_compact(self, max_per_kind: int = 4) -> str:
        """Format non-empty candidates as compact hint lines for LLM grounding."""
        lines: list[str] = []
        for label, values in (
            ("NIP", self.nip),
            ("KRS", self.krs),
            ("REGON", self.regon),
            ("PESEL", self.pesel),
            ("NRB", self.nrb),
            ("BZP", self.bzp),
            ("TED", self.ted),
            ("EMAIL", self.emails),
            ("DATY", self.dates),
        ):
            if values:
                lines.append(f"- {label}: {', '.join(values[:max_per_kind])}")
        return "\n".join(lines)


def _prioritize_checksum_valid(items: list[str], check_fn: Callable[[str], bool]) -> list[str]:
    deduped = _dedupe(items)
    return sorted(deduped, key=lambda item: not check_fn(item))


def extract_identifier_hints(text: str) -> IdentifierHints:
    """Extract normalized identifier and date candidates from raw document text."""
    if not text:
        return IdentifierHints()

    nips = [d for m in _NIP_RE.findall(text) if (d := _digits_only(m)) and len(d) == 10]
    krss = [d for m in _KRS_RE.findall(text) if (d := _digits_only(m)) and len(d) == 10]
    regons = [d for m in _REGON_RE.findall(text) if (d := _digits_only(m)) and len(d) in (9, 14)]
    pesels_labeled = [d for m in _PESEL_LABELED_RE.findall(text) if (d := _digits_only(m)) and len(d) == 11]
    pesels_standalone = [d for d in _PESEL_STANDALONE_RE.findall(text) if is_valid_pesel(d)]
    nrbs = extract_accounts_in_order(text)
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
        nip=_prioritize_checksum_valid(nips, is_valid_nip),
        krs=_dedupe(krss),
        regon=_prioritize_checksum_valid(regons, is_valid_regon),
        pesel=_prioritize_checksum_valid([*pesels_labeled, *pesels_standalone], is_valid_pesel),
        nrb=_prioritize_checksum_valid(nrbs, is_valid_nrb),
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


def _is_pattern_placeholder(value: str) -> bool:
    upper = value.upper()
    return "YYYY" in upper or "####" in upper or "NNNNNN" in upper


def _normalize_bzp(value: Any) -> str | None:
    if not isinstance(value, str) or _is_pattern_placeholder(value):
        return None
    return re.sub(r"\s+", "", value).upper()


def _normalize_ted(value: Any) -> str | None:
    if not isinstance(value, str) or _is_pattern_placeholder(value):
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

    zamawiajacy = data.get("Zamawiajacy")
    if isinstance(zamawiajacy, list) and len(zamawiajacy) == 1 and isinstance(zamawiajacy[0], dict):
        zamawiajacy[0]["prowadzacy_postepowanie"] = None


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


def normalize_payload(doc_type: DocumentType | str, raw: dict[str, Any]) -> dict[str, Any]:
    """Deterministically clean placeholders, identifiers, dates, and derived fields."""
    if not isinstance(raw, dict):
        return {}
    data = _clean_placeholders(deepcopy(raw))
    if not isinstance(data, dict):
        return {}

    dt = _coerce_doc_type(doc_type)
    if dt == DocumentType.PRZELEW:
        _normalize_przelew(data)
    elif dt == DocumentType.ODWOLANIE:
        _normalize_odwolanie(data)
    elif dt == DocumentType.PELNOMOCNICTWO:
        _normalize_pelnomocnictwo(data)

    return data


class ValidationOutcome(BaseModel):
    """Result of schema validation plus source-text grounding checks."""

    valid: bool
    data: dict[str, Any] | None = None
    errors: list[str] = Field(default_factory=list)
    failed_fields: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _format_loc(loc: tuple[Any, ...]) -> str:
    if not loc:
        return "root"
    parts: list[str] = []
    for part in loc:
        if isinstance(part, int):
            if parts:
                parts[-1] = f"{parts[-1]}[{part}]"
            else:
                parts.append(f"[{part}]")
        else:
            parts.append(str(part))
    return ".".join(parts)


def _check_iso_date(field_path: str, value: str | None, findings: list[tuple[str, str]]) -> None:
    if value is None:
        return
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", value):
        findings.append((field_path, f"{field_path}='{value}' nie ma formatu YYYY-MM-DD"))
        return
    try:
        date.fromisoformat(value)
    except ValueError:
        findings.append((field_path, f"{field_path}='{value}' nie jest poprawną datą kalendarzową"))


def _check_identifier_field(
    field_path: str,
    value: Any,
    compact_source: str,
    candidates: list[str],
    expected_lengths: tuple[int, ...],
    checksum_fn: Callable[[str], bool] | None,
    strict_checksums: bool,
    findings: list[tuple[str, str]],
    warnings: list[str],
) -> None:
    if value is None:
        return
    digits = _digits_only(value) or ""
    hint = f"; wykryte w dokumencie: {candidates}" if candidates else ""

    if len(digits) not in expected_lengths:
        lengths_label = "/".join(str(n) for n in expected_lengths)
        findings.append(
            (
                field_path,
                f"{field_path}='{value}' ma niepoprawną długość {len(digits)} (wymagane: {lengths_label} cyfr){hint}",
            )
        )
        return

    if digits not in compact_source:
        findings.append((field_path, f"{field_path}='{value}' nie występuje w tekście źródłowym{hint}"))
        return

    if checksum_fn is not None and not checksum_fn(digits):
        msg = f"{field_path}='{value}' nie przechodzi weryfikacji cyfry kontrolnej"
        if strict_checksums:
            findings.append((field_path, f"{msg}{hint}"))
        else:
            warnings.append(f"{msg} (dane syntetyczne/zanonimizowane lub błąd OCR)")


def _check_account_field(
    field_path: str,
    value: Any,
    compact_source: str,
    candidates: list[str],
    strict_checksums: bool,
    findings: list[tuple[str, str]],
    warnings: list[str],
) -> None:
    if value is None:
        return
    norm = _strip_account(value) or ""
    hint = f"; wykryte w dokumencie: {candidates}" if candidates else ""
    is_pl_nrb = norm.isdigit() and len(norm) == 26
    is_foreign_iban = bool(re.match(r"^[A-Z]{2}\d{2}[0-9A-Z]{11,30}$", norm))

    if not (is_pl_nrb or is_foreign_iban):
        findings.append(
            (
                field_path,
                f"{field_path}='{value}' ma niepoprawny format rachunku (wymagane 26 cyfr NRB lub międzynarodowy IBAN){hint}",
            )
        )
        return

    if norm not in compact_source:
        findings.append((field_path, f"{field_path}='{value}' nie występuje w tekście źródłowym{hint}"))
        return

    if not is_valid_nrb(norm):
        msg = f"{field_path}='{value}' nie przechodzi weryfikacji cyfry kontrolnej"
        if strict_checksums:
            findings.append((field_path, f"{msg}{hint}"))
        else:
            warnings.append(f"{msg} (dane syntetyczne/zanonimizowane lub błąd OCR)")


def _run_grounding_checks(
    doc_type: DocumentType,
    data: dict[str, Any],
    source_text: str,
    hints: IdentifierHints,
    *,
    strict_checksums: bool,
) -> tuple[list[tuple[str, str]], list[str]]:
    findings: list[tuple[str, str]] = []
    warnings: list[str] = []
    compact_source = re.sub(r"[\s-]+", "", source_text).upper()

    if doc_type == DocumentType.PRZELEW:
        pb = data.get("przelew_bankowy") or {}
        _check_iso_date("przelew_bankowy.data_operacji", pb.get("data_operacji"), findings)
        _check_account_field(
            "przelew_bankowy.z_rachunku",
            pb.get("z_rachunku"),
            compact_source,
            hints.nrb,
            strict_checksums,
            findings,
            warnings,
        )
        _check_account_field(
            "przelew_bankowy.na_rachunek",
            pb.get("na_rachunek"),
            compact_source,
            hints.nrb,
            strict_checksums,
            findings,
            warnings,
        )

    elif doc_type == DocumentType.ODWOLANIE:
        kio = data.get("OdwolanieKIO") or {}
        _check_iso_date("OdwolanieKIO.data", kio.get("data"), findings)
        _check_iso_date("OdwolanieKIO.czynnosc_dzien", kio.get("czynnosc_dzien"), findings)

        if not (data.get("Zamawiajacy") or []):
            findings.append(
                (
                    "Zamawiajacy",
                    "Zamawiajacy nie może być pustą listą — każde odwołanie do KIO ma co najmniej jednego Zamawiającego",
                )
            )

        for section_name in ("Odwolujacy", "Zamawiajacy"):
            for idx, entity in enumerate(data.get(section_name) or []):
                _check_identifier_field(
                    f"{section_name}[{idx}].nip",
                    entity.get("nip"),
                    compact_source,
                    hints.nip,
                    (10,),
                    is_valid_nip,
                    strict_checksums,
                    findings,
                    warnings,
                )
                _check_identifier_field(
                    f"{section_name}[{idx}].krs",
                    entity.get("krs"),
                    compact_source,
                    hints.krs,
                    (10,),
                    None,
                    strict_checksums,
                    findings,
                    warnings,
                )
                _check_identifier_field(
                    f"{section_name}[{idx}].regon",
                    entity.get("regon"),
                    compact_source,
                    hints.regon,
                    (9, 14),
                    is_valid_regon,
                    strict_checksums,
                    findings,
                    warnings,
                )

        zp = data.get("ZamowieniePubliczne") or {}
        bzp = zp.get("numer_bzp")
        if bzp and re.sub(r"\s+", "", bzp).upper() not in compact_source:
            hint = f"; wykryte w dokumencie: {hints.bzp}" if hints.bzp else ""
            findings.append(
                ("ZamowieniePubliczne.numer_bzp", f"ZamowieniePubliczne.numer_bzp='{bzp}' nie występuje w tekście źródłowym{hint}")
            )

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
                findings.append(
                    ("ZamowieniePubliczne.numer_ted", f"ZamowieniePubliczne.numer_ted='{ted}' nie występuje w tekście źródłowym{hint}")
                )

    elif doc_type == DocumentType.PELNOMOCNICTWO:
        for idx, entry in enumerate(data.get("mocodawca") or []):
            entity = entry.get("data") or {}
            _check_identifier_field(
                f"mocodawca[{idx}].data.nip",
                entity.get("nip"),
                compact_source,
                hints.nip,
                (10,),
                is_valid_nip,
                strict_checksums,
                findings,
                warnings,
            )
            _check_identifier_field(
                f"mocodawca[{idx}].data.krs",
                entity.get("krs"),
                compact_source,
                hints.krs,
                (10,),
                None,
                strict_checksums,
                findings,
                warnings,
            )
        for idx, person in enumerate(data.get("pelnomocnicy") or []):
            _check_identifier_field(
                f"pelnomocnicy[{idx}].pesel",
                person.get("pesel"),
                compact_source,
                hints.pesel,
                (11,),
                is_valid_pesel,
                strict_checksums,
                findings,
                warnings,
            )

    return findings, warnings


def validate_extraction(
    doc_type: DocumentType | str,
    raw: dict[str, Any],
    source_text: str,
    *,
    strict_checksums: bool | None = None,
) -> ValidationOutcome:
    """Normalize LLM output, validate against Pydantic UZP schema, and verify grounding."""
    normalized = normalize_payload(doc_type, raw)
    dt = _coerce_doc_type(doc_type)
    schema_cls = SCHEMA_REGISTRY.get(dt) if dt is not None else None
    if dt is None or schema_cls is None:
        return ValidationOutcome(valid=True, data=normalized)

    try:
        model = schema_cls.model_validate(normalized)
    except ValidationError as exc:
        findings: list[tuple[str, str]] = []
        for err in exc.errors():
            path = _format_loc(tuple(err["loc"]))
            findings.append((path, f"{path}: {err['msg']}"))
        failed_fields = _dedupe(path for path, _ in findings if path != "root")
        return ValidationOutcome(
            valid=False,
            data=normalized,
            errors=[msg for _, msg in findings],
            failed_fields=failed_fields,
        )

    dumped = model.model_dump(by_alias=True)
    hints = extract_identifier_hints(source_text)
    use_strict = settings.llm_strict_checksums if strict_checksums is None else strict_checksums
    grounding_findings, warnings = _run_grounding_checks(
        dt,
        dumped,
        source_text,
        hints,
        strict_checksums=use_strict,
    )
    return ValidationOutcome(
        valid=len(grounding_findings) == 0,
        data=dumped,
        errors=[msg for _, msg in grounding_findings],
        failed_fields=_dedupe(path for path, _ in grounding_findings),
        warnings=warnings,
    )
