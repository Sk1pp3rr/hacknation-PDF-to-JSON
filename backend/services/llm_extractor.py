"""Token-efficient EU LLM extraction engine with strict schemas, validation, and targeted retry."""

import asyncio
from dataclasses import dataclass, field
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.config import settings
from backend.schemas import (
    CZYNNOSC_OPIS_BY_ID,
    DocumentSegment,
    DocumentType,
    OdwolanieSchema,
    OdwolujacyItem,
    PelnomocnictwoSchema,
    PrzelewSchema,
    ZamawiajacyItem,
    ZamowieniePubliczne,
)
from backend.services.llm_client import LLMClient, TokenUsage
from backend.services.schema_utils import to_strict_json_schema
from backend.services.validator import extract_identifier_hints, validate_extraction

_CZYN_DICT_PROMPT = "; ".join(f"{k}={v}" for k, v in CZYNNOSC_OPIS_BY_ID.items())


@dataclass(frozen=True)
class ExtractionSpec:
    """Per-document-class extraction settings and token budget."""

    doc_type: DocumentType
    schema_cls: type[BaseModel]
    system_prompt: str
    max_tokens: int
    max_chars: int
    exclude_llm_fields: frozenset[str] = field(default_factory=frozenset)
    force_string_fields: frozenset[str] = field(default_factory=frozenset)


EXTRACTION_SPECS: dict[DocumentType, ExtractionSpec] = {
    DocumentType.ODWOLANIE: ExtractionSpec(
        doc_type=DocumentType.ODWOLANIE,
        schema_cls=OdwolanieSchema,
        system_prompt=(
            "Ekstrahujesz dane z polskiego odwołania do KIO do ścisłego JSON.\n"
            "Zasady:\n"
            "- Zwracaj null jeśli brak informacji w tekście (nie zgaduj).\n"
            "- Daty podawaj w formacie YYYY-MM-DD.\n"
            "- data i miejscowosc bierz z nagłówka pierwszej strony.\n"
            "- czynnosc_dzien i czynnosc_powod_id ustalaj WYŁĄCZNIE dla pierwszego zarzutu/zaskarżonej czynności.\n"
            f"- czynnosc_powod_id (1-12): {_CZYN_DICT_PROMPT}.\n"
            "- Odwołujący: jeśli jest konsorcjum, podaj Lidera; przez-pelnomocnika=1 gdy działa pełnomocnik, inaczej 0.\n"
            "- numer_bzp format 'YYYY/BZP########' bez spacji; numer_ted format 'NNNNNN-YYYY'."
        ),
        max_tokens=850,
        max_chars=10000,
        exclude_llm_fields=frozenset({"czynnosc_opis"}),
    ),
    DocumentType.PELNOMOCNICTWO: ExtractionSpec(
        doc_type=DocumentType.PELNOMOCNICTWO,
        schema_cls=PelnomocnictwoSchema,
        system_prompt=(
            "Ekstrahujesz dane z polskiego pełnomocnictwa do ścisłego JSON.\n"
            "Zasady:\n"
            "- Zwracaj null jeśli brak pola (nie zgaduj).\n"
            "- mocodawca[].data.typ to 'osoba_prawna' lub 'osoba_fizyczna'; dla osoby prawnej imie_nazwisko=null.\n"
            "- podpisujacy to lista osób podpisujących w imieniu mocodawcy.\n"
            "- pelnomocnicy[].imie_nazwisko podawaj w mianowniku, bez tytułów (np. bez 'r. pr.').\n"
            "- pesel zwracaj jako napis cyfr lub null."
        ),
        max_tokens=500,
        max_chars=6000,
        force_string_fields=frozenset({"pesel"}),
    ),
    DocumentType.PRZELEW: ExtractionSpec(
        doc_type=DocumentType.PRZELEW,
        schema_cls=PrzelewSchema,
        system_prompt=(
            "Ekstrahujesz dane z potwierdzenia przelewu bankowego do ścisłego JSON.\n"
            "Zasady:\n"
            "- Zwracaj null jeśli brak informacji.\n"
            "- data_operacji w formacie YYYY-MM-DD.\n"
            "- kwota jako liczba (float), bez waluty.\n"
            "- z_rachunku i na_rachunek jako same 26 cyfr bez spacji i bez prefiksu 'PL'."
        ),
        max_tokens=350,
        max_chars=4000,
    ),
}


class _OdwolanieHeaderPayload(BaseModel):
    data: str | None = None
    miejscowosc: str | None = None
    Odwolujacy: list[OdwolujacyItem]
    Zamawiajacy: list[ZamawiajacyItem]
    ZamowieniePubliczne: ZamowieniePubliczne


class _OdwolanieActionPayload(BaseModel):
    czynnosc_dzien: str | None = None
    czynnosc_powod_id: int = Field(ge=1, le=12)


_HEADER_SYSTEM_PROMPT = (
    "Ekstrahujesz WYŁĄCZNIE nagłówek, strony (Odwołujący, Zamawiający) i dane zamówienia publicznego do JSON.\n"
    "Zwracaj null jeśli brak informacji. Daty w formacie YYYY-MM-DD. "
    "numer_bzp bez spacji ('YYYY/BZP########'), numer_ted jako 'NNNNNN-YYYY'."
)

_ACTION_SYSTEM_PROMPT = (
    "Ekstrahujesz WYŁĄCZNIE datę czynności (czynnosc_dzien, YYYY-MM-DD lub null) "
    "oraz czynnosc_powod_id (liczba 1-12) wyłącznie dla pierwszego zarzutu/zaskarżonej czynności.\n"
    f"Kody czynnosc_powod_id: {_CZYN_DICT_PROMPT}."
)


class DocumentExtractionOutcome(BaseModel):
    """Full extraction result including validation status and token telemetry."""

    doc_type: DocumentType
    valid: bool
    data: dict[str, Any] = Field(default_factory=dict)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    attempts: int = 0
    validation_errors: list[str] = Field(default_factory=list)
    mode: str = "single"


def _clean_text(text: str, max_chars: int) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"(?im)^\s*Strona\s+\d+\s+z\s+\d+\s*$", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip()
    return cleaned


def split_odwolanie_sections(text: str) -> tuple[str, str]:
    """Deterministically slice an appeal into (header_text, first_claim_text) to shrink 3B prompts."""
    cleaned = _clean_text(text, max_chars=12000)

    header_end_match = re.search(
        r"(?im)^(?:\s*Zarzut\s+nr\s*1\b|\s*Wskazanie\s+narusze[ńn]\b|\s*Wnosz[ęe]\s+o\b|\s*UZASADNIENIE\b)",
        cleaned,
    )
    header_slice = cleaned[: header_end_match.start()].strip() if header_end_match else cleaned[:3500]
    if len(header_slice) < 100:
        header_slice = cleaned[:3500]

    action_start_match = re.search(r"(?im)^(?:\s*Dzia[łl]aj[ąa]c\b|\s*O\s*D\s*W\s*O\s*[ŁL]\s*A\s*N\s*I\s*E\b)", cleaned)
    action_start = action_start_match.start() if action_start_match else min(500, len(cleaned) // 4)

    action_tail = cleaned[action_start:]
    action_end_match = re.search(
        r"(?im)^(?:\s*Zarzut\s+nr\s*2\b|\s*UZASADNIENIE\b|\s*Za[łl][ąa]czniki\s*:)",
        action_tail,
    )
    action_slice = action_tail[: action_end_match.start()].strip() if action_end_match else action_tail[:4500]
    if len(action_slice) < 80:
        action_slice = cleaned[min(500, len(cleaned) // 4) : 5500]

    return header_slice, action_slice


def _parse_json_content(content: str) -> tuple[dict[str, Any], list[str]]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return {}, [f"Niepoprawny JSON od modelu: {exc.msg}"]
    if not isinstance(parsed, dict):
        return {}, ["Odpowiedź JSON modelu musi być obiektem"]
    return parsed, []


def _build_user_message(text: str, *, include_hints: bool) -> str:
    if not include_hints:
        return f"TEKST DOKUMENTU:\n{text}"
    hints = extract_identifier_hints(text)
    if hints.is_empty():
        return f"TEKST DOKUMENTU:\n{text}"
    return f"WYKRYTE KANDYDATY REGEX (użyj do weryfikacji):\n{hints.format_compact()}\n\nTEKST DOKUMENTU:\n{text}"


def _build_repair_messages(
    spec: ExtractionSpec,
    pruned_text: str,
    previous_payload: dict[str, Any],
    errors: list[str],
    *,
    include_hints: bool,
) -> list[dict[str, str]]:
    hints_block = ""
    if include_hints:
        hints = extract_identifier_hints(pruned_text)
        if not hints.is_empty():
            hints_block = f"\nWykryte kandydaty w dokumencie:\n{hints.format_compact()}\n"

    error_lines = "\n".join(f"- {err}" for err in errors[:8])
    return [
        {"role": "system", "content": spec.system_prompt},
        {
            "role": "user",
            "content": (
                "Poprzedni JSON zawiera błędy walidacji względem schematu lub tekstu źródłowego:\n"
                f"{error_lines}\n"
                f"{hints_block}\n"
                f"POPRZEDNI JSON:\n{json.dumps(previous_payload, ensure_ascii=False)}\n\n"
                f"FRAGMENT DOKUMENTU:\n{pruned_text[:6000]}\n\n"
                "Zwróć wyłącznie poprawiony kompletny JSON."
            ),
        },
    ]


async def _run_single_shot(
    spec: ExtractionSpec,
    pruned_text: str,
    strict_schema: dict[str, Any],
    client: LLMClient,
    *,
    include_hints: bool,
) -> tuple[dict[str, Any], list[str], TokenUsage, int]:
    messages = [
        {"role": "system", "content": spec.system_prompt},
        {"role": "user", "content": _build_user_message(pruned_text, include_hints=include_hints)},
    ]
    res = await client.complete_json(
        messages=messages,
        schema_name=f"{spec.doc_type.value}_extraction",
        json_schema=strict_schema,
        max_tokens=spec.max_tokens,
    )
    parsed, parse_errors = _parse_json_content(res.content)
    return parsed, parse_errors, res.usage, 1


async def _run_odwolanie_section_swarm(
    pruned_text: str,
    client: LLMClient,
    *,
    include_hints: bool,
) -> tuple[dict[str, Any], list[str], TokenUsage, int]:
    header_slice, action_slice = split_odwolanie_sections(pruned_text)
    header_schema = to_strict_json_schema(_OdwolanieHeaderPayload)
    action_schema = to_strict_json_schema(_OdwolanieActionPayload)

    header_msgs = [
        {"role": "system", "content": _HEADER_SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_message(header_slice, include_hints=include_hints)},
    ]
    action_msgs = [
        {"role": "system", "content": _ACTION_SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_message(action_slice, include_hints=False)},
    ]

    header_res, action_res = await asyncio.gather(
        client.complete_json(
            messages=header_msgs,
            schema_name="odwolanie_header",
            json_schema=header_schema,
            max_tokens=650,
        ),
        client.complete_json(
            messages=action_msgs,
            schema_name="odwolanie_action",
            json_schema=action_schema,
            max_tokens=160,
        ),
    )

    header_dict, header_errors = _parse_json_content(header_res.content)
    action_dict, action_errors = _parse_json_content(action_res.content)
    total_usage = header_res.usage + action_res.usage

    merged: dict[str, Any] = {
        "OdwolanieKIO": {
            "data": header_dict.get("data"),
            "czynnosc_dzien": action_dict.get("czynnosc_dzien"),
            "czynnosc_powod_id": action_dict.get("czynnosc_powod_id"),
            "miejscowosc": header_dict.get("miejscowosc"),
        },
        "Odwolujacy": header_dict.get("Odwolujacy", []),
        "Zamawiajacy": header_dict.get("Zamawiajacy", []),
        "ZamowieniePubliczne": header_dict.get("ZamowieniePubliczne", {}),
    }
    return merged, [*header_errors, *action_errors], total_usage, 2


async def extract_document(
    doc_type: DocumentType,
    text: str,
    *,
    client: LLMClient | None = None,
    mode: Literal["single", "section_swarm"] | None = None,
    max_retries: int | None = None,
    include_hints: bool = True,
) -> DocumentExtractionOutcome:
    """Extract canonical UZP JSON for one classified document with external validation."""
    if doc_type == DocumentType.INNY or not text.strip() or doc_type not in EXTRACTION_SPECS:
        return DocumentExtractionOutcome(
            doc_type=doc_type,
            valid=True,
            data={},
            usage=TokenUsage(),
            attempts=0,
            mode="skipped",
        )

    spec = EXTRACTION_SPECS[doc_type]
    active_client = client or LLMClient()
    active_mode = mode or settings.llm_extraction_mode
    if doc_type != DocumentType.ODWOLANIE:
        active_mode = "single"
    retries_allowed = settings.llm_max_retries if max_retries is None else max_retries

    pruned_text = _clean_text(text, max_chars=spec.max_chars)
    strict_schema = to_strict_json_schema(
        spec.schema_cls,
        exclude_fields=set(spec.exclude_llm_fields),
        force_string_fields=set(spec.force_string_fields),
    )

    if active_mode == "section_swarm":
        raw_payload, parse_errors, total_usage, attempts = await _run_odwolanie_section_swarm(
            pruned_text,
            active_client,
            include_hints=include_hints,
        )
    else:
        raw_payload, parse_errors, total_usage, attempts = await _run_single_shot(
            spec,
            pruned_text,
            strict_schema,
            active_client,
            include_hints=include_hints,
        )

    if parse_errors:
        errors = parse_errors
        outcome_data = raw_payload
        is_valid = False
    else:
        validation = validate_extraction(doc_type, raw_payload, pruned_text)
        outcome_data = validation.data or {}
        errors = validation.errors
        is_valid = validation.valid

    for _ in range(max(0, retries_allowed)):
        if is_valid:
            break
        repair_msgs = _build_repair_messages(
            spec,
            pruned_text,
            outcome_data,
            errors,
            include_hints=include_hints,
        )
        repair_res = await active_client.complete_json(
            messages=repair_msgs,
            schema_name=f"{doc_type.value}_repair",
            json_schema=strict_schema,
            max_tokens=spec.max_tokens,
        )
        total_usage = total_usage + repair_res.usage
        attempts += 1

        repaired_raw, repair_parse_errors = _parse_json_content(repair_res.content)
        if repair_parse_errors:
            errors = repair_parse_errors
            is_valid = False
            continue

        validation = validate_extraction(doc_type, repaired_raw, pruned_text)
        outcome_data = validation.data or {}
        errors = validation.errors
        is_valid = validation.valid

    return DocumentExtractionOutcome(
        doc_type=doc_type,
        valid=is_valid,
        data=outcome_data,
        usage=total_usage,
        attempts=attempts,
        validation_errors=errors,
        mode=active_mode,
    )


async def extract_structured_data(
    segments: list[DocumentSegment],
    *,
    client: LLMClient | None = None,
) -> dict[str, Any]:
    """Extract structured JSON from classified document segments."""
    active_segments = [seg for seg in segments if seg.doc_type != DocumentType.INNY and seg.text.strip()]
    if not active_segments:
        return {}

    outcomes = await asyncio.gather(
        *(extract_document(seg.doc_type, seg.text, client=client) for seg in active_segments)
    )
    if len(outcomes) == 1:
        return outcomes[0].data
    return {"documents": [outcome.data for outcome in outcomes]}
