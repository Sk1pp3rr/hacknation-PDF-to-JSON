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
    _PESEL_STANDALONE_RE,
    _REGON_RE,
    _digits_only,
    extract_accounts_in_order,
    extract_identifier_hints,
    is_valid_pesel,
    normalize_city_name,
    normalize_date_str,
    normalize_person_name,
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
_LAWYER_NAME_RE = re.compile(
    r"(?i)(?:r\.\s*pr\.|radc[aąęy]\s+prawn(?:ego|emu|ym|y)|adw\.|adwokat(?:a|owi|em)?|mec\.|mecenas(?:a|owi|em)?)\s+"
    r"([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+\s+[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż\-]+)"
)
_DOTS_LINE_RE = re.compile(r"^[\.\s_…]{5,}$")
_COMPANY_NAME_RE = re.compile(
    r"([A-ZĄĆĘŁŃÓŚŹŻ][A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż0-9\-\s\.]{2,60}?"
    r"(?:S\.A\.|sp\.\s*z\s*o\.o\.(?:\s*sp\.\s*k\.)?|sp\.\s*[kj]\.))"
)

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
    r"(?i)^(?:r\.\s*pr\.|radc[aąęy]\s+prawn(?:ego|emu|ym|y)|adw\.|adwokat(?:a|owi|em)?|"
    r"mec\.|mecenas(?:a|owi|em)?|dr\s+hab\.|mgr\s+in[żz]\.|mgr|dr|prof\.)\s+"
)
_TRAILING_TITLE_RE = re.compile(
    r"(?i),?\s*(?:r\.\s*pr\.|radc[aąęy]\s+prawn(?:ego|emu|ym|y)|adw\.|adwokat(?:a|owi|em)?|mec\.|mecenas(?:a|owi|em)?)$"
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
            return normalize_city_name(city) or city, iso_date
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
    return normalize_person_name(cleaned) or None


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


def _extract_signers_from_lines(lines: list[str]) -> list[dict[str, str]]:
    signers: list[dict[str, str]] = []
    idx = 0
    while idx < len(lines):
        if _DOTS_LINE_RE.match(lines[idx]):
            while idx < len(lines) and (_DOTS_LINE_RE.match(lines[idx]) or lines[idx].lower().startswith("za ")):
                idx += 1
            if idx < len(lines):
                cleaned = strip_person_titles(lines[idx]) or lines[idx]
                signers.append({"imie_nazwisko": cleaned})
        idx += 1
    return signers


def _extract_first_company_name(text: str) -> str | None:
    t_top = re.split(r"(?i)prowadzonym\s+przez|Zamawiaj[ąa]cy", text)[0]
    m = _COMPANY_NAME_RE.search(t_top)
    if not m:
        return None
    found = re.sub(r"\s+", " ", m.group(1)).strip()
    found = re.sub(
        r"(?i)^(?:.*?:\s*|.*?\b(?:PE[ŁL]NOMOCNICTWO(?:\s+PROCESOWE)?|P\s*E\s*[ŁL]\s*N\s*O\s*M\s*O\s*C\s*N\s*I\s*C\s*T\s*W\s*O|"
        r"Mocodawc[ąa]\s+jest|Mocodawca|w\s+imieniu|reprezentacji)\s+)",
        "",
        found,
    ).strip()
    return found or None


def _expand_odwolanie_representatives(raw_rep: Any, rep_mail: str | None) -> list[dict[str, Any]]:
    if raw_rep is None:
        raw_reps: list[dict[str, Any]] = []
    elif isinstance(raw_rep, dict):
        raw_reps = [dict(raw_rep)]
    elif isinstance(raw_rep, list):
        raw_reps = [dict(r) for r in raw_rep if isinstance(r, dict)]
    else:
        raw_reps = []

    expanded: list[dict[str, Any]] = []
    for rep in raw_reps:
        rep.setdefault("mail", None)
        raw_mail = rep.get("mail")
        if raw_mail is None and rep_mail is not None:
            rep["mail"] = rep_mail
        elif isinstance(raw_mail, str) and ("," in raw_mail or ";" in raw_mail):
            first_mail = _EMAIL_RE.search(raw_mail)
            rep["mail"] = first_mail.group(0) if first_mail else rep_mail

        raw_name = rep.get("imie_nazwisko")
        if isinstance(raw_name, str) and "," in raw_name:
            names = [
                cleaned
                for part in raw_name.split(",")
                if (cleaned := strip_person_titles(part)) is not None
            ]
            if names:
                for idx, name in enumerate(names):
                    entry = dict(rep) if idx == 0 else {"nazwa": rep.get("nazwa"), "mail": rep.get("mail")}
                    entry["imie_nazwisko"] = name
                    expanded.append(entry)
                continue

        if isinstance(raw_name, str):
            rep["imie_nazwisko"] = strip_person_titles(raw_name)
        expanded.append(rep)

    return expanded


def _augment_odwolanie(data: dict[str, Any], source_text: str) -> None:
    header_text = _header_slice_only(source_text)
    rep_mail = _extract_representative_mail(header_text)
    header_lawyers = list(
        dict.fromkeys(
            norm
            for m in _LAWYER_NAME_RE.finditer(header_text)
            if (norm := normalize_person_name(re.sub(r"\s+", " ", m.group(1)).strip()))
        )
    )

    odwolujacy = data.get("Odwolujacy")
    if isinstance(odwolujacy, list):
        for item in odwolujacy:
            if not isinstance(item, dict):
                continue
            _fill_missing_entity_ids(item, header_text, include_regon=True)
            reps = _expand_odwolanie_representatives(item.get("pelnomocnik"), rep_mail)
            firm_name = next((r.get("nazwa") for r in reps if isinstance(r.get("nazwa"), str) and r["nazwa"]), None)
            mail_val = next((r.get("mail") for r in reps if isinstance(r.get("mail"), str) and r["mail"]), rep_mail)

            if header_lawyers and (item.get("przez-pelnomocnika") == 1 or reps):
                by_name = {
                    r["imie_nazwisko"].strip().lower(): r
                    for r in reps
                    if isinstance(r.get("imie_nazwisko"), str) and r["imie_nazwisko"].strip()
                }
                merged_reps: list[dict[str, Any]] = []
                for lawyer_name in header_lawyers:
                    existing = by_name.get(lawyer_name.lower())
                    if existing is not None:
                        entry = dict(existing)
                        entry["imie_nazwisko"] = lawyer_name
                        if not entry.get("nazwa"):
                            entry["nazwa"] = firm_name
                        if not entry.get("mail"):
                            entry["mail"] = mail_val
                    else:
                        entry = {"imie_nazwisko": lawyer_name, "nazwa": firm_name, "mail": mail_val}
                    merged_reps.append(entry)
                item["pelnomocnik"] = merged_reps
            else:
                item["pelnomocnik"] = [r for r in reps if any(v is not None for v in r.values())]

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
    t_main = re.split(r"(?im)^\s*Pe[łl]nomocnictwo\s+przyjmuj[ęe]", source_text)[0]
    found_lawyers = list(
        dict.fromkeys(re.sub(r"\s+", " ", m.group(1)).strip() for m in _LAWYER_NAME_RE.finditer(t_main))
    )
    if found_lawyers:
        cur_peln = [
            strip_person_titles(p.get("imie_nazwisko"))
            for p in (data.get("pelnomocnicy") or [])
            if isinstance(p, dict) and isinstance(p.get("imie_nazwisko"), str)
        ]
        if cur_peln != found_lawyers:
            data["pelnomocnicy"] = [{"imie_nazwisko": name, "pesel": None} for name in found_lawyers]

    mocodawcy = data.get("mocodawca")
    if isinstance(mocodawcy, list):
        if len(mocodawcy) == 1 and isinstance(mocodawcy[0], dict):
            lines = [ln.strip() for ln in t_main.splitlines() if ln.strip()]
            signers = _extract_signers_from_lines(lines)
            if signers:
                mocodawcy[0]["podpisujacy"] = signers
        elif len(mocodawcy) > 1:
            chunks = re.split(r"(?im)^\s*W\s+imieniu\s+", t_main)[1:]
            if len(chunks) == len(mocodawcy):
                for ch, entry in zip(chunks, mocodawcy):
                    if isinstance(entry, dict):
                        c_lines = [ln.strip() for ln in ch.splitlines() if ln.strip()]
                        signers = _extract_signers_from_lines(c_lines)
                        if signers:
                            entry["podpisujacy"] = signers

        for entry in mocodawcy:
            if not isinstance(entry, dict):
                continue
            m_data = entry.get("data")
            if isinstance(m_data, dict):
                cur_nazwa = str(m_data.get("nazwa") or "").strip()
                if len(mocodawcy) == 1 and (
                    not cur_nazwa
                    or re.search(r"(?i)^(?:KRS|NIP|REGON|ul\.)|\b\d{2}-\d{3}\b", cur_nazwa)
                    or cur_nazwa not in source_text
                    or (m_data.get("typ") == "osoba_prawna" and not _COMPANY_NAME_RE.search(cur_nazwa))
                ):
                    repaired_name = _extract_first_company_name(t_main)
                    if repaired_name:
                        m_data["nazwa"] = repaired_name
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


_TX_REF_RE = re.compile(
    r"(?im)(?:nr\s*ref\.?|numer\s+referencyjny|numer\s+transakcji|nr\s+transakcji|id\s+transakcji|numer\s+operacji|nr\s+operacji)"
    r"[\s.:·\-]{1,60}([A-Z0-9][A-Z0-9\-/]{5,40})"
)


def _extract_transaction_number(source_text: str) -> str | None:
    match = _TX_REF_RE.search(source_text)
    if match:
        candidate = match.group(1).strip(".,;:")
        if any(ch.isdigit() for ch in candidate):
            return candidate
    return None


_TX_TITLE_LABEL_RE = re.compile(
    r"(?im)^[\s.\-•'\"]*(?:[tq]ytu[łl](?:\s+(?:przelewu|operacji|transakcji))?|opis(?:\s+(?:operacji|transakcji|przelewu))?)\s*(?:[:;]|(?:\s+[y:]))?\s*(.*)$"
)
_TX_TITLE_STOP_RE = re.compile(
    r"(?i)^(?:typ\s+transakcji|kwota\b|nr\s*ref|numer\s+referencyjn|status\b|kana[łl]\b|tryb\b|"
    r"dokument\s+jest\s+wydrukiem|zrealizowano\b|potwierdzenie\s+ma\s+charakter|wp[łl]yn[ęe][łl]o\b|"
    r"bank\s+[\w\s]+s\.a\.|data\s+(?:operacji|waluty|ksi[ęe]gowania|transakcji)|rachunek\b|z\s+rachunku|na\s+rachunek)"
)
_BANK_OP_HEADER_TYTUL_RE = re.compile(
    r"(?i)^[\"']?(?:przelew\s+(?:elixir|krajowy|wewn[ęe]trzny|z\s+rachunku|sorbnet|natychmiastowy)|"
    r"zlecenie\s+przelewu)\b"
)


def _extract_transfer_title(source_text: str) -> str | None:
    lines = [line.strip() for line in source_text.splitlines()]
    for idx, line in enumerate(lines):
        if not line or re.match(r"(?i)^opis\s+kwota$", line):
            continue
        m = _TX_TITLE_LABEL_RE.match(line)
        if not m:
            continue
        first = m.group(1).strip(" .:;-")
        # Ignore standalone single-letter OCR noise like `Tytuł y`
        parts: list[str] = [] if len(first) <= 1 and first.isalpha() else ([first] if first else [])
        for nxt in lines[idx + 1 : idx + 8]:
            if not nxt:
                if parts:
                    continue
                continue
            if _TX_TITLE_STOP_RE.match(nxt):
                break
            cleaned_nxt = re.sub(r"\s+[.;:]$", "", nxt).strip()
            if cleaned_nxt:
                parts.append(cleaned_nxt)
        joined = re.sub(r"\s+", " ", " ".join(parts)).strip(" .;:")
        if len(joined) >= 6 and not _BANK_OP_HEADER_TYTUL_RE.match(joined):
            return joined
    return None


def _is_bank_header_tytul(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    return not stripped or bool(_BANK_OP_HEADER_TYTUL_RE.match(stripped))


def _augment_przelew(data: dict[str, Any], source_text: str) -> None:
    pb = data.get("przelew_bankowy")
    if not isinstance(pb, dict):
        return
    pb.setdefault("z_rachunku", None)
    pb.setdefault("na_rachunek", None)
    if pb.get("numer_transakcji") is None:
        pb["numer_transakcji"] = _extract_transaction_number(source_text)
    if _is_bank_header_tytul(pb.get("tytul")):
        extracted_title = _extract_transfer_title(source_text)
        if extracted_title:
            pb["tytul"] = extracted_title
    if pb.get("z_rachunku") is not None and pb.get("na_rachunek") is not None:
        return
    ordered = extract_accounts_in_order(source_text)
    if len(ordered) >= 2:
        if pb.get("z_rachunku") is None:
            pb["z_rachunku"] = ordered[0]
        if pb.get("na_rachunek") is None:
            pb["na_rachunek"] = ordered[1]
    elif len(ordered) == 1 and pb.get("z_rachunku") is None and pb.get("na_rachunek") is None:
        only_acc = ordered[0]
        digits = re.sub(r"\D", "", only_acc)
        compact = re.sub(r"\s+", "", source_text)
        pos = compact.find(digits)
        ctx = compact[max(0, pos - 60) : pos].lower() if pos >= 0 else ""
        if any(k in ctx for k in ("odbiorc", "narachunek", "beneficjent", "adresat")):
            pb["na_rachunek"] = only_acc
        else:
            pb["z_rachunku"] = only_acc


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
            r"(?im)^.*(?:odrzuc|wyb[óo]r|wyklucz|specyfikacj|SWZ|wyja[śs]nie|modyfikacj|zaniecha|uniewa[żz]ni|OPZ|przedmiotu\s+zam[óo]wienia|postanowie[ńn]\s+umowy).*$",
            page_text,
        )
        return re.sub(r"\s+", " ", m_act.group(0)).strip()[:160] if m_act else None
    if leaf_name == "prowadzacy_postepowanie":
        m_zam = re.search(r"(?im)^.*Zamawiaj[ąa]cy.*$", page_text)
        return re.sub(r"\s+", " ", m_zam.group(0)).strip()[:160] if m_zam else None
    if leaf_name == "typ":
        m_typ = re.search(
            r"(?im)^.*(?:Mocodawc|KRS|NIP|CEIDG|sp\.\s*z\s*o\.o\.|S\.A\.|sp\.\s*[kj]\.).*$",
            page_text,
        )
        return re.sub(r"\s+", " ", m_typ.group(0)).strip()[:160] if m_typ else None

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

    # 3. Digits or international IBAN with optional separators (NIP, KRS, REGON, PESEL, NRB, IBAN)
    digits = re.sub(r"\D", "", s_val)
    if (len(digits) in (9, 10, 11, 14, 26) and digits == s_val) or re.match(r"^[A-Z]{2}\d{2}[0-9A-Z]{11,30}$", s_val):
        sep_pat = r"[\s-]*".join(re.escape(ch) for ch in s_val)
        m = re.search(sep_pat, page_text, re.IGNORECASE)
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

    # 5. Amount (float -> Polish comma/dot formatted amount with or without currency suffix)
    if leaf_name == "kwota" and isinstance(value, (int, float)):
        int_part = int(round(float(value)))
        m_amt = re.search(
            rf"\b{int_part}(?:[\s\xa0]*[,.]\d{{2}}\b|\s*(?:PLN|z[łl])\b)",
            page_text,
            re.IGNORECASE,
        )
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
