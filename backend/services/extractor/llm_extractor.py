"""Config-driven EU LLM extraction engine with strict schemas, validation, and surgical retry."""

import asyncio
from copy import deepcopy
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.config import (
    DocumentConfig,
    RepairPromptsConfig,
    get_document_config,
    get_repair_prompts,
    settings,
)
from backend.core.models import DocumentSegment, DocumentType
from backend.schemas.uzp import (
    CZYNNOSC_OPIS_BY_ID,
    OdwolujacyItem,
    SCHEMA_REGISTRY,
    ZamawiajacyItem,
    ZamowieniePubliczne,
)
from backend.services.extractor.llm_client import LLMClient, TokenUsage
from backend.services.extractor.schema_utils import to_strict_json_schema
from backend.services.normalizer import (
    ValidationOutcome,
    annotate_odwolanie_header,
    augment_post_extraction,
    extract_identifier_hints,
    validate_extraction,
)

_CZYN_DICT_PROMPT = "; ".join(f"{k}={v}" for k, v in CZYNNOSC_OPIS_BY_ID.items())
_PROMPT_VARIABLES = {"czynnosc_dict": _CZYN_DICT_PROMPT}
_SCALAR_JSON_TYPES = {"string", "integer", "number", "boolean"}
_SCHEMA_BY_NAME: dict[str, type[BaseModel]] = {cls.__name__: cls for cls in SCHEMA_REGISTRY.values()}


class _OdwolanieHeaderPayload(BaseModel):
    data: str | None = None
    miejscowosc: str | None = None
    Odwolujacy: list[OdwolujacyItem]
    Zamawiajacy: list[ZamawiajacyItem]
    ZamowieniePubliczne: ZamowieniePubliczne


class _OdwolanieActionPayload(BaseModel):
    czynnosc_dzien: str | None = None
    czynnosc_powod_id: int = Field(ge=1, le=12)


class DocumentExtractionOutcome(BaseModel):
    """Full extraction result including validation status and token telemetry."""

    doc_type: DocumentType | str
    valid: bool
    data: dict[str, Any] = Field(default_factory=dict)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    attempts: int = 0
    validation_errors: list[str] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)
    mode: str = "single"


def _render_template(template: str, variables: dict[str, str]) -> str:
    rendered = template.strip()
    for key, value in variables.items():
        rendered = rendered.replace(f"{{{key}}}", value)
    return rendered


def _resolve_strict_schema(cfg: DocumentConfig) -> dict[str, Any] | None:
    schema_cls = _SCHEMA_BY_NAME.get(cfg.ekstrakcja.schema_name)
    if schema_cls is None:
        try:
            schema_cls = SCHEMA_REGISTRY.get(DocumentType(cfg.id))
        except ValueError:
            schema_cls = None

    if schema_cls is not None:
        return to_strict_json_schema(
            schema_cls,
            exclude_fields=set(cfg.ekstrakcja.exclude_llm_fields),
            force_string_fields=set(cfg.ekstrakcja.force_string_fields),
        )
    if cfg.ekstrakcja.json_schema is not None:
        return deepcopy(cfg.ekstrakcja.json_schema)
    return None


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

    action_start_match = re.search(
        r"(?im)^(?:\s*Dzia[łl]aj[ąa]c\b|\s*O\s*D\s*W\s*O\s*[ŁL]\s*A\s*N\s*I\s*E\b)",
        cleaned,
    )
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
    system_prompt: str,
    repair_prompts: RepairPromptsConfig,
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
    user_content = repair_prompts.full_repair_user_template.format(
        error_lines=error_lines,
        hints_block=hints_block,
        previous_json=json.dumps(previous_payload, ensure_ascii=False),
        text_slice=pruned_text[:6000],
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


# --- Surgical field-only repair helpers ---


def _parse_field_path(path: str) -> list[str | int]:
    tokens: list[str | int] = []
    for part in path.split("."):
        m = re.match(r"^([A-Za-z0-9_-]+)\[(\d+)\]$", part)
        if m:
            tokens.append(m.group(1))
            tokens.append(int(m.group(2)))
        else:
            tokens.append(part)
    return tokens


def _unwrap_object_node(node: dict[str, Any]) -> dict[str, Any] | None:
    if node.get("type") == "object" and isinstance(node.get("properties"), dict):
        return node
    for branch in node.get("anyOf") or ():
        if isinstance(branch, dict) and branch.get("type") == "object" and isinstance(branch.get("properties"), dict):
            return branch
    return None


def _is_scalar_schema(node: Any) -> bool:
    if not isinstance(node, dict):
        return False
    if node.get("type") in _SCALAR_JSON_TYPES:
        return True
    any_of = node.get("anyOf")
    if isinstance(any_of, list) and any_of:
        allowed = _SCALAR_JSON_TYPES | {"null"}
        return all(isinstance(opt, dict) and opt.get("type") in allowed for opt in any_of) and any(
            opt.get("type") in _SCALAR_JSON_TYPES for opt in any_of
        )
    return False


def _lookup_leaf_schema(strict_schema: dict[str, Any], field_path: str) -> dict[str, Any] | None:
    curr: Any = strict_schema
    for token in _parse_field_path(field_path):
        if isinstance(token, int):
            if isinstance(curr, dict) and curr.get("type") == "array" and isinstance(curr.get("items"), dict):
                curr = curr["items"]
            else:
                return None
        else:
            obj = _unwrap_object_node(curr) if isinstance(curr, dict) else None
            if obj is None or token not in obj["properties"]:
                return None
            curr = obj["properties"][token]
    return deepcopy(curr) if _is_scalar_schema(curr) else None


def _resolve_parent(payload: dict[str, Any], tokens: list[str | int]) -> tuple[Any, str | int] | None:
    if not tokens:
        return None
    curr: Any = payload
    for token in tokens[:-1]:
        if isinstance(token, int):
            if not isinstance(curr, list) or not (0 <= token < len(curr)):
                return None
            curr = curr[token]
        else:
            if not isinstance(curr, dict) or token not in curr:
                return None
            curr = curr[token]
    last = tokens[-1]
    if isinstance(last, int):
        if not isinstance(curr, list) or not (0 <= last < len(curr)):
            return None
    elif not isinstance(curr, dict):
        return None
    return curr, last


def _get_by_path(payload: dict[str, Any], field_path: str) -> tuple[bool, Any]:
    resolved = _resolve_parent(payload, _parse_field_path(field_path))
    if resolved is None:
        return False, None
    parent, last = resolved
    if isinstance(last, int):
        return True, parent[last]
    return True, parent.get(last)


def _set_by_path(payload: dict[str, Any], field_path: str, value: Any) -> bool:
    resolved = _resolve_parent(payload, _parse_field_path(field_path))
    if resolved is None:
        return False
    parent, last = resolved
    parent[last] = value
    return True


def _make_safe_key(field_path: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", field_path.replace("[", "_").replace("]", "_"))


def _build_surgical_patch_plan(
    doc_type: DocumentType | str,
    repair_prompts: RepairPromptsConfig,
    strict_schema: dict[str, Any],
    outcome_data: dict[str, Any],
    failed_fields: list[str],
    pruned_text: str,
    errors: list[str],
    *,
    include_hints: bool,
) -> tuple[dict[str, Any], dict[str, str], list[dict[str, str]]] | None:
    if not (1 <= len(failed_fields) <= 5) or not isinstance(outcome_data, dict):
        return None

    patch_props: dict[str, Any] = {}
    key_to_path: dict[str, str] = {}
    current_values: dict[str, Any] = {}

    for idx, field_path in enumerate(failed_fields):
        leaf_schema = _lookup_leaf_schema(strict_schema, field_path)
        exists, current_val = _get_by_path(outcome_data, field_path)
        if leaf_schema is None or not exists:
            return None
        safe_key = _make_safe_key(field_path)
        if safe_key in key_to_path:
            safe_key = f"{safe_key}_{idx}"
        patch_props[safe_key] = leaf_schema
        key_to_path[safe_key] = field_path
        current_values[safe_key] = current_val

    patch_schema: dict[str, Any] = {
        "type": "object",
        "properties": patch_props,
        "required": list(patch_props.keys()),
        "additionalProperties": False,
    }

    if str(doc_type) == DocumentType.ODWOLANIE.value and all(
        p.startswith("OdwolanieKIO.czynnosc_") for p in failed_fields
    ):
        text_slice = split_odwolanie_sections(pruned_text)[1]
    else:
        text_slice = pruned_text[:4500]

    hints_block = ""
    if include_hints:
        hints = extract_identifier_hints(text_slice)
        if any("dzien" in p or "data" in p for p in failed_fields):
            hints.dates = [d for d in hints.dates if d != "2019-09-11"]
        if not hints.is_empty():
            hints_block = f"\nWykryte kandydaty w dokumencie:\n{hints.format_compact()}\n"

    key_map_lines = "\n".join(f"- {k} -> {p}" for k, p in key_to_path.items())
    error_lines = "\n".join(f"- {err}" for err in errors[:6])
    user_content = repair_prompts.surgical_repair_user_template.format(
        error_lines=error_lines,
        key_map_lines=key_map_lines,
        current_values=json.dumps(current_values, ensure_ascii=False),
        hints_block=hints_block,
        text_slice=text_slice,
    )
    messages = [
        {"role": "system", "content": repair_prompts.surgical_repair_system.strip()},
        {"role": "user", "content": user_content},
    ]
    return patch_schema, key_to_path, messages


async def _run_single_shot(
    doc_id: str,
    system_prompt: str,
    max_tokens: int,
    pruned_text: str,
    strict_schema: dict[str, Any],
    client: LLMClient,
    *,
    include_hints: bool,
) -> tuple[dict[str, Any], list[str], TokenUsage, int]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _build_user_message(pruned_text, include_hints=include_hints)},
    ]
    res = await client.complete_json(
        messages=messages,
        schema_name=f"{doc_id}_extraction",
        json_schema=strict_schema,
        max_tokens=max_tokens,
    )
    parsed, parse_errors = _parse_json_content(res.content)
    return parsed, parse_errors, res.usage, 1


async def _run_odwolanie_section_swarm(
    cfg: DocumentConfig,
    pruned_text: str,
    client: LLMClient,
    *,
    include_hints: bool,
) -> tuple[dict[str, Any], list[str], TokenUsage, int]:
    swarm_cfg = cfg.ekstrakcja.section_swarm
    assert swarm_cfg is not None

    header_slice, action_slice = split_odwolanie_sections(pruned_text)
    header_schema = to_strict_json_schema(_OdwolanieHeaderPayload)
    action_schema = to_strict_json_schema(_OdwolanieActionPayload)

    header_system_prompt = _render_template(swarm_cfg.header_system_prompt, _PROMPT_VARIABLES)
    action_system_prompt = _render_template(swarm_cfg.action_system_prompt, _PROMPT_VARIABLES)

    annotated_header = annotate_odwolanie_header(header_slice)
    header_msgs = [
        {"role": "system", "content": header_system_prompt},
        {"role": "user", "content": f"TEKST DOKUMENTU:\n{annotated_header}"},
    ]

    action_user_content = f"TEKST DOKUMENTU:\n{action_slice}"
    if include_hints:
        action_hints = extract_identifier_hints(action_slice)
        action_dates = [d for d in action_hints.dates if d != "2019-09-11"]
        if action_dates:
            action_user_content = (
                f"- DATY (kandydaci na czynnosc_dzien): {', '.join(action_dates[:4])}\n\n"
                f"{action_user_content}"
            )
    action_msgs = [
        {"role": "system", "content": action_system_prompt},
        {"role": "user", "content": action_user_content},
    ]

    header_res, action_res = await asyncio.gather(
        client.complete_json(
            messages=header_msgs,
            schema_name="odwolanie_header",
            json_schema=header_schema,
            max_tokens=swarm_cfg.header_max_tokens,
        ),
        client.complete_json(
            messages=action_msgs,
            schema_name="odwolanie_action",
            json_schema=action_schema,
            max_tokens=swarm_cfg.action_max_tokens,
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


def _augment_and_validate(
    doc_type: DocumentType | str,
    raw_payload: dict[str, Any],
    pruned_text: str,
) -> ValidationOutcome:
    augmented = augment_post_extraction(doc_type, raw_payload, pruned_text)
    return validate_extraction(doc_type, augmented, pruned_text)


async def extract_document(
    doc_type: DocumentType | str,
    text: str,
    *,
    client: LLMClient | None = None,
    mode: Literal["single", "section_swarm"] | None = None,
    max_retries: int | None = None,
    include_hints: bool = True,
    config: DocumentConfig | None = None,
) -> DocumentExtractionOutcome:
    """Extract canonical JSON for one classified document using YAML configuration."""
    if str(doc_type) == DocumentType.INNY.value or not text.strip():
        return DocumentExtractionOutcome(
            doc_type=doc_type,
            valid=True,
            data={},
            usage=TokenUsage(),
            attempts=0,
            mode="skipped",
        )

    if config is not None:
        cfg = config
    else:
        try:
            cfg = get_document_config(doc_type)
        except KeyError:
            return DocumentExtractionOutcome(
                doc_type=doc_type,
                valid=True,
                data={},
                usage=TokenUsage(),
                attempts=0,
                mode="skipped",
            )

    strict_schema = _resolve_strict_schema(cfg)
    if strict_schema is None:
        return DocumentExtractionOutcome(
            doc_type=doc_type,
            valid=True,
            data={},
            usage=TokenUsage(),
            attempts=0,
            mode="skipped",
        )

    active_client = client or LLMClient()
    requested_mode = mode or settings.llm_extraction_mode
    active_mode = (
        "section_swarm"
        if requested_mode == "section_swarm" and cfg.ekstrakcja.section_swarm is not None
        else "single"
    )
    retries_allowed = settings.llm_max_retries if max_retries is None else max_retries

    system_prompt = cfg.ekstrakcja.render_system_prompt(_PROMPT_VARIABLES)
    pruned_text = _clean_text(text, max_chars=cfg.ekstrakcja.max_chars)

    if active_mode == "section_swarm":
        raw_payload, parse_errors, total_usage, attempts = await _run_odwolanie_section_swarm(
            cfg,
            pruned_text,
            active_client,
            include_hints=include_hints,
        )
    else:
        raw_payload, parse_errors, total_usage, attempts = await _run_single_shot(
            cfg.id,
            system_prompt,
            cfg.ekstrakcja.max_tokens,
            pruned_text,
            strict_schema,
            active_client,
            include_hints=include_hints,
        )

    if parse_errors:
        errors = parse_errors
        warnings: list[str] = []
        failed_fields: list[str] = []
        outcome_data = raw_payload
        is_valid = False
    else:
        validation = _augment_and_validate(doc_type, raw_payload, pruned_text)
        outcome_data = validation.data or {}
        errors = validation.errors
        warnings = validation.warnings
        failed_fields = validation.failed_fields
        is_valid = validation.valid

    repair_prompts: RepairPromptsConfig | None = None
    for _ in range(max(0, retries_allowed)):
        if is_valid:
            break
        if repair_prompts is None:
            repair_prompts = get_repair_prompts()

        patch_plan = (
            None
            if parse_errors
            else _build_surgical_patch_plan(
                doc_type,
                repair_prompts,
                strict_schema,
                outcome_data,
                failed_fields,
                pruned_text,
                errors,
                include_hints=include_hints,
            )
        )

        if patch_plan is not None:
            patch_schema, key_to_path, repair_msgs = patch_plan
            repair_res = await active_client.complete_json(
                messages=repair_msgs,
                schema_name=f"{cfg.id}_patch",
                json_schema=patch_schema,
                max_tokens=min(240, 80 * len(failed_fields) + 60),
            )
            total_usage = total_usage + repair_res.usage
            attempts += 1

            patch_raw, repair_parse_errors = _parse_json_content(repair_res.content)
            if repair_parse_errors:
                parse_errors = repair_parse_errors
                errors = repair_parse_errors
                failed_fields = []
                is_valid = False
                continue

            candidate = deepcopy(outcome_data)
            for safe_key, field_path in key_to_path.items():
                if safe_key in patch_raw:
                    _set_by_path(candidate, field_path, patch_raw[safe_key])
            validation = _augment_and_validate(doc_type, candidate, pruned_text)
            outcome_data = validation.data or {}
            errors = validation.errors
            warnings = validation.warnings
            failed_fields = validation.failed_fields
            is_valid = validation.valid
        else:
            repair_msgs = _build_repair_messages(
                system_prompt,
                repair_prompts,
                pruned_text,
                outcome_data,
                errors,
                include_hints=include_hints,
            )
            repair_res = await active_client.complete_json(
                messages=repair_msgs,
                schema_name=f"{cfg.id}_repair",
                json_schema=strict_schema,
                max_tokens=cfg.ekstrakcja.max_tokens,
            )
            total_usage = total_usage + repair_res.usage
            attempts += 1

            repaired_raw, repair_parse_errors = _parse_json_content(repair_res.content)
            if repair_parse_errors:
                parse_errors = repair_parse_errors
                errors = repair_parse_errors
                failed_fields = []
                is_valid = False
                continue

            parse_errors = []
            validation = _augment_and_validate(doc_type, repaired_raw, pruned_text)
            outcome_data = validation.data or {}
            errors = validation.errors
            warnings = validation.warnings
            failed_fields = validation.failed_fields
            is_valid = validation.valid

    return DocumentExtractionOutcome(
        doc_type=doc_type,
        valid=is_valid,
        data=outcome_data,
        usage=total_usage,
        attempts=attempts,
        validation_errors=errors,
        validation_warnings=warnings,
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
