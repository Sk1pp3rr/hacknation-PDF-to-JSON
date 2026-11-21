"""Deterministic pre- and post-extraction augmentation (0 LLM tokens)."""

from copy import deepcopy
import re
from typing import Any

from pydantic import BaseModel

from backend.core.models import DocumentType
from backend.services.normalizer.validator import (
    _BZP_RE,
    _DATE_DMY_RE,
    _DATE_ISO_RE,
    _DATE_PL_RE,
    _EMAIL_RE,
    _KRS_RE,
    _NIP_RE,
    _NRB_RE,
    _PESEL_STANDALONE_RE,
    _REGON_RE,
    _digits_only,
    extract_identifier_hints,
    is_valid_pesel,
    normalize_date_str,
)

_ENTITY_WINDOW_STOP_RE = re.compile(
    r"(?i)(?:\b\d\)\s*(?:Partner|Cz[łl]onek|Wykonawca)|\bZamawiaj[ąa]cy\b|\bOdwo[łl]uj[ąa]cy\b|"
    r"\bWykonawca\s+wybrany|\bdotyczy\s*:|\breprezentowan[ay]\s+przez|\badres\s+do\s+korespondencji|"
    r"\bPe[łl]nomocni(?:k|cy)\b|\bniniejszym\s+upowa[żz]niam|\bustanawiam\s+pe[łl]nomocnikiem)"
)
_HEADER_BODY_BOUNDARY_RE = re.compile(
    r"(?im)^(?:\s*Zarzut\s+nr\s*1\b|\s*Wskazanie\s+narusze[ńn]\b|\s*Wnosz[ęe]\s+o\b|\s*UZASADNIENIE\b)"
)
_REP_ANCHOR_RE = re.compile(
    r"(?i)(?:reprezentowan[ay]\s+przez|adres\s+do\s+korespondencji|pe[łl]nomocnik)"
)
_REP_STOP_RE = re.compile(r"(?i)\b(?:Zamawiaj[ąa]cy|Wykonawca\s+wybrany|dotyczy)\b")

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


def _header_slice_only(source_text: str) -> str:
    m = _HEADER_BODY_BOUNDARY_RE.search(source_text)
    if m and m.start() >= 100:
        return source_text[: m.start()]
    return source_text[:3500]


def _bind_entity_identifiers(text: str, nazwa: str | None) -> tuple[str | None, str | None, str | None]:
    """Deterministically find (krs, nip, regon) right after `nazwa` in `text`."""
    if not nazwa or not isinstance(nazwa, str):
        return None, None, None
    norm_text = re.sub(r"\s+", " ", text)
    norm_name = re.sub(r"\s+", " ", nazwa).strip()
    pos = norm_text.lower().find(norm_name.lower())
    if pos < 0:
        short = " ".join(norm_name.split()[:2]).lower()
        pos = norm_text.lower().find(short) if short else -1
    if pos < 0:
        return None, None, None

    window = norm_text[pos : pos + 350]
    stop = _ENTITY_WINDOW_STOP_RE.search(window[len(norm_name) :])
    entity_win = window[: len(norm_name) + stop.start()] if stop else window[:260]

    krs_m = _KRS_RE.search(entity_win)
    nip_m = _NIP_RE.search(entity_win)
    reg_m = _REGON_RE.search(entity_win)
    krs = _digits_only(krs_m.group(1)) if krs_m else None
    nip = _digits_only(nip_m.group(1)) if nip_m else None
    reg = _digits_only(reg_m.group(1)) if reg_m else None
    return (
        krs if krs and len(krs) == 10 else None,
        nip if nip and len(nip) == 10 else None,
        reg if reg and len(reg) in (9, 14) else None,
    )


def _extract_representative_mail(header_text: str) -> str | None:
    norm_hdr = re.sub(r"\s+", " ", header_text)
    m_rep = _REP_ANCHOR_RE.search(norm_hdr)
    if not m_rep:
        return None
    sub = norm_hdr[m_rep.start() : m_rep.start() + 350]
    stop = _REP_STOP_RE.search(sub)
    if stop:
        sub = sub[: stop.start()]
    em = _EMAIL_RE.search(sub)
    return em.group(0) if em else None


def _fill_missing_entity_ids(entity: dict[str, Any], text: str, *, include_regon: bool = True) -> None:
    entity.setdefault("krs", None)
    entity.setdefault("nip", None)
    if include_regon:
        entity.setdefault("regon", None)
    krs, nip, regon = _bind_entity_identifiers(text, entity.get("nazwa"))
    if entity.get("krs") is None and krs is not None:
        entity["krs"] = krs
    if entity.get("nip") is None and nip is not None:
        entity["nip"] = nip
    if include_regon and entity.get("regon") is None and regon is not None:
        entity["regon"] = regon


def _augment_odwolanie(data: dict[str, Any], source_text: str) -> None:
    header_text = _header_slice_only(source_text)
    rep_mail = _extract_representative_mail(header_text)

    odwolujacy = data.get("Odwolujacy")
    if isinstance(odwolujacy, list):
        for item in odwolujacy:
            if not isinstance(item, dict):
                continue
            _fill_missing_entity_ids(item, header_text, include_regon=True)
            rep = item.get("pelnomocnik")
            if isinstance(rep, dict):
                rep.setdefault("mail", None)
                if isinstance(rep.get("imie_nazwisko"), str):
                    rep["imie_nazwisko"] = _clean_single_representative(rep["imie_nazwisko"])
                raw_mail = rep.get("mail")
                if raw_mail is None and rep_mail is not None:
                    rep["mail"] = rep_mail
                elif isinstance(raw_mail, str) and ("," in raw_mail or ";" in raw_mail):
                    first_mail = _EMAIL_RE.search(raw_mail)
                    rep["mail"] = first_mail.group(0) if first_mail else rep_mail

    zamawiajacy = data.get("Zamawiajacy")
    if isinstance(zamawiajacy, list):
        for item in zamawiajacy:
            if isinstance(item, dict):
                _fill_missing_entity_ids(item, header_text, include_regon=True)

    if isinstance(odwolujacy, list) and isinstance(zamawiajacy, list):
        odwolujacy, zamawiajacy = _dedupe_entities(odwolujacy, zamawiajacy)
        data["Odwolujacy"] = odwolujacy
        data["Zamawiajacy"] = zamawiajacy

    if isinstance(zamawiajacy, list) and len(zamawiajacy) == 1 and isinstance(zamawiajacy[0], dict):
        zamawiajacy[0]["prowadzacy_postepowanie"] = None

    kio = data.get("OdwolanieKIO")
    if isinstance(kio, dict):
        kio.setdefault("data", None)
        kio.setdefault("miejscowosc", None)
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
        zp.setdefault("numer_bzp", None)
        zp.setdefault("numer_ted", None)
        header_hints = extract_identifier_hints(header_text)
        raw_bzp = zp.get("numer_bzp")
        if isinstance(raw_bzp, str) and raw_bzp.strip() and not _BZP_RE.search(raw_bzp):
            # Model confused numer_postepowania with numer_bzp (e.g. 'ZP.431.2026').
            cur_post = zp.get("numer_postepowania")
            if not cur_post or (isinstance(cur_post, str) and cur_post in raw_bzp):
                zp["numer_postepowania"] = raw_bzp.strip()
            zp["numer_bzp"] = None
        if zp.get("numer_bzp") is None and len(header_hints.bzp) == 1:
            zp["numer_bzp"] = header_hints.bzp[0]
        if zp.get("numer_ted") is None and len(header_hints.ted) == 1:
            zp["numer_ted"] = header_hints.ted[0]


def _augment_pelnomocnictwo(data: dict[str, Any], source_text: str) -> None:
    hints = extract_identifier_hints(source_text)
    mocodawcy = data.get("mocodawca")
    if isinstance(mocodawcy, list):
        for entry in mocodawcy:
            if not isinstance(entry, dict):
                continue
            m_data = entry.get("data")
            if isinstance(m_data, dict):
                _fill_missing_entity_ids(m_data, source_text, include_regon=False)
                if len(mocodawcy) == 1:
                    if m_data.get("krs") is None and len(hints.krs) == 1:
                        m_data["krs"] = hints.krs[0]
                    if m_data.get("nip") is None and len(hints.nip) == 1:
                        m_data["nip"] = hints.nip[0]
            for signer in entry.get("podpisujacy") or ():
                if isinstance(signer, dict) and isinstance(signer.get("imie_nazwisko"), str):
                    signer["imie_nazwisko"] = strip_person_titles(signer["imie_nazwisko"]) or signer["imie_nazwisko"]

    norm_text = re.sub(r"\s+", " ", source_text)
    pelnomocnicy = data.get("pelnomocnicy")
    if isinstance(pelnomocnicy, list):
        for person in pelnomocnicy:
            if not isinstance(person, dict):
                continue
            person.setdefault("pesel", None)
            if isinstance(person.get("imie_nazwisko"), str):
                person["imie_nazwisko"] = strip_person_titles(person["imie_nazwisko"]) or person["imie_nazwisko"]
            pesel = person.get("pesel")
            if pesel is not None:
                digits = re.sub(r"\D", "", str(pesel))
                if len(digits) != 11 and not hints.pesel:
                    person["pesel"] = None
            elif hints.pesel:
                name = person.get("imie_nazwisko")
                if isinstance(name, str) and name.strip():
                    pos = norm_text.lower().find(name.strip().lower())
                    if pos >= 0:
                        win = norm_text[pos : pos + 160]
                        for cand in _PESEL_STANDALONE_RE.findall(win):
                            if is_valid_pesel(cand):
                                person["pesel"] = cand
                                break
                if person.get("pesel") is None and len(pelnomocnicy) == 1 and len(hints.pesel) == 1:
                    person["pesel"] = hints.pesel[0]


def _augment_przelew(data: dict[str, Any], source_text: str) -> None:
    pb = data.get("przelew_bankowy")
    if not isinstance(pb, dict):
        return
    pb.setdefault("z_rachunku", None)
    pb.setdefault("na_rachunek", None)
    if pb.get("z_rachunku") is not None and pb.get("na_rachunek") is not None:
        return
    nrbs = [d for m in _NRB_RE.findall(source_text) if (d := _digits_only(m)) and len(d) == 26]
    ordered = list(dict.fromkeys(nrbs))
    if len(ordered) == 2:
        if pb.get("z_rachunku") is None:
            pb["z_rachunku"] = ordered[0]
        if pb.get("na_rachunek") is None:
            pb["na_rachunek"] = ordered[1]


def augment_post_extraction(doc_type: DocumentType | str, data: dict[str, Any], source_text: str) -> dict[str, Any]:
    """Apply deterministic, zero-token domain augmentation after LLM extraction."""
    if not isinstance(data, dict):
        return {}
    augmented = deepcopy(data)

    if str(doc_type) == DocumentType.ODWOLANIE.value:
        _augment_odwolanie(augmented, source_text)
    elif str(doc_type) == DocumentType.PELNOMOCNICTWO.value:
        _augment_pelnomocnictwo(augmented, source_text)
    elif str(doc_type) == DocumentType.PRZELEW.value:
        _augment_przelew(augmented, source_text)

    return augmented


class FieldEvidence(BaseModel):
    """Zero-token source citation (1-based page number and verbatim quote) for an extracted leaf field."""

    field: str
    value: Any
    page: int | None = None
    quote: str | None = None


def _iter_non_null_leaves(node: Any, prefix: str = "") -> list[tuple[str, Any]]:
    leaves: list[tuple[str, Any]] = []
    if isinstance(node, dict):
        for k, v in node.items():
            path = f"{prefix}.{k}" if prefix else k
            leaves.extend(_iter_non_null_leaves(v, path))
    elif isinstance(node, list):
        for idx, item in enumerate(node):
            leaves.extend(_iter_non_null_leaves(item, f"{prefix}[{idx}]"))
    elif node is not None:
        leaves.append((prefix, node))
    return leaves


def _snippet_around(text: str, start: int, end: int, max_len: int = 160) -> str:
    line_start = text.rfind("\n", 0, start)
    line_start = 0 if line_start < 0 else line_start + 1
    line_end = text.find("\n", end)
    line_end = len(text) if line_end < 0 else line_end
    snippet = re.sub(r"\s+", " ", text[line_start:line_end]).strip()
    if len(snippet) > max_len:
        rel = max(0, start - line_start - 30)
        snippet = re.sub(r"\s+", " ", text[line_start + rel : line_start + rel + max_len]).strip()
    return snippet


def _match_on_page(field_path: str, value: Any, page_text: str) -> str | None:
    if not page_text:
        return None
    leaf_name = field_path.rsplit(".", 1)[-1]
    s_val = str(value).strip()
    if not s_val:
        return None

    # 1. Derived flags / action codes (check before generic substring matching of '0', '1', '9')
    if leaf_name == "przez-pelnomocnika":
        m_rep = _REP_ANCHOR_RE.search(page_text)
        return _snippet_around(page_text, m_rep.start(), m_rep.end()) if m_rep else None
    if leaf_name in {"czynnosc_powod_id", "czynnosc_opis"}:
        m_act = re.search(
            r"(?im)^.*(?:odrzuc|wyb[óo]r|wyklucz|specyfikacj|SWZ|uniewa[żz]ni|OPZ|postanowie[ńn]\s+umowy).*$",
            page_text,
        )
        return re.sub(r"\s+", " ", m_act.group(0)).strip()[:160] if m_act else None
    if leaf_name == "prowadzacy_postepowanie":
        m_zam = re.search(r"(?im)^.*Zamawiaj[ąa]cy.*$", page_text)
        return re.sub(r"\s+", " ", m_zam.group(0)).strip()[:160] if m_zam else None

    # 2. Exact or whitespace-flexible match
    words = [re.escape(w) for w in s_val.split() if w]
    if words:
        flex_re = re.compile(r"\s+".join(words), re.IGNORECASE)
        m = flex_re.search(page_text)
        if m:
            return _snippet_around(page_text, m.start(), m.end())

    # 2. ISO date (YYYY-MM-DD) -> match textual / DMY / ISO date on page
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s_val):
        for pat in (_DATE_ISO_RE, _DATE_DMY_RE, _DATE_PL_RE):
            for m in pat.finditer(page_text):
                if normalize_date_str(m.group(1)) == s_val:
                    return _snippet_around(page_text, m.start(), m.end())

    # 3. Digits with optional separators (NIP, KRS, REGON, PESEL, NRB)
    digits = re.sub(r"\D", "", s_val)
    if len(digits) in (9, 10, 11, 14, 26) and digits == s_val:
        sep_pat = r"[\s-]*".join(re.escape(ch) for ch in digits)
        m = re.search(sep_pat, page_text)
        if m:
            return _snippet_around(page_text, m.start(), m.end())

    # 4. BZP / TED
    if leaf_name == "numer_bzp":
        m_bzp = _BZP_RE.search(page_text)
        if m_bzp and re.sub(r"\s+", "", m_bzp.group(1)).upper() == s_val.upper():
            return _snippet_around(page_text, m_bzp.start(), m_bzp.end())
    if leaf_name == "numer_ted" and "-" in s_val:
        num, year = s_val.split("-", 1)
        m_ted = re.search(rf"(?i)\b{re.escape(year)}/S\s*\d{{1,3}}-{re.escape(num)}\b", page_text)
        if m_ted:
            return _snippet_around(page_text, m_ted.start(), m_ted.end())

    # 5. Amount (float -> Polish comma/space formatted amount)
    if leaf_name == "kwota" and isinstance(value, (int, float)):
        int_part = int(round(float(value)))
        m_amt = re.search(rf"\b{int_part}(?:[\s\xa0]*[,.]\d{{2}})?\s*(?:PLN|z[łl])\b", page_text, re.IGNORECASE)
        if m_amt:
            return _snippet_around(page_text, m_amt.start(), m_amt.end())

    # 6. Derived flags / action codes
    if leaf_name == "przez-pelnomocnika" and value == 1:
        m_rep = _REP_ANCHOR_RE.search(page_text)
        if m_rep:
            return _snippet_around(page_text, m_rep.start(), m_rep.end())
    if leaf_name in {"czynnosc_powod_id", "czynnosc_opis"}:
        m_act = re.search(
            r"(?im)^.*(?:odrzuc|wyb[óo]r|wyklucz|specyfikacj|SWZ|uniewa[żz]ni|OPZ|postanowie[ńn]\s+umowy).*$",
            page_text,
        )
        if m_act:
            return re.sub(r"\s+", " ", m_act.group(0)).strip()[:160]

    # 7. Prefix-stem fallback for inflected Polish names / titles
    stem_tokens = [re.escape(w[: max(4, len(w) - 2)]) + r"\w*" for w in s_val.split() if len(w) >= 3]
    if len(stem_tokens) >= 2:
        m_stem = re.search(r"\s+".join(stem_tokens[:4]), page_text, re.IGNORECASE)
        if m_stem:
            return _snippet_around(page_text, m_stem.start(), m_stem.end())

    return None


def locate_field_evidence(
    data: dict[str, Any],
    source_text: str,
    pages: list[str] | None = None,
) -> list[FieldEvidence]:
    """Deterministically locate 1-based page number and verbatim source quote for each extracted field (0 tokens)."""
    if not isinstance(data, dict) or not data:
        return []
    page_list = pages if (pages and any(pages)) else ([source_text] if source_text else [])
    evidence: list[FieldEvidence] = []

    for field_path, value in _iter_non_null_leaves(data):
        found_page: int | None = None
        found_quote: str | None = None
        for idx, page_text in enumerate(page_list):
            quote = _match_on_page(field_path, value, page_text)
            if quote:
                found_page = idx + 1
                found_quote = quote
                break
        evidence.append(
            FieldEvidence(
                field=field_path,
                value=value,
                page=found_page,
                quote=found_quote,
            )
        )
    return evidence
