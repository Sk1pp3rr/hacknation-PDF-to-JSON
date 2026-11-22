"""Offline unit and integration tests for the EU LLM API interface and validator."""

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from backend.config import get_all_document_configs, get_repair_prompts
from backend.schemas import (
    CZYNNOSC_OPIS_BY_ID,
    DocumentSegment,
    DocumentType,
    OdwolanieSchema,
    PelnomocnictwoSchema,
    PrzelewSchema,
)
from backend.services.extractor import LLMClient, LLMClientError, extract_document, extract_structured_data
from backend.services.extractor.schema_utils import to_strict_json_schema
from backend.services.normalizer import (
    annotate_odwolanie_header,
    augment_post_extraction,
    extract_header_city_and_date,
    extract_identifier_hints,
    format_uzp_json,
    is_valid_nip,
    is_valid_nrb,
    is_valid_pesel,
    is_valid_regon,
    normalize_payload,
    strip_kio_addressee,
    strip_person_titles,
    validate_extraction,
)
from backend.services.parser import extract_text_from_pdf_file


def _collect_keys(node: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(node, dict):
        keys.update(node.keys())
        for val in node.values():
            keys.update(_collect_keys(val))
    elif isinstance(node, list):
        for item in node:
            keys.update(_collect_keys(item))
    return keys


def _assert_all_objects_strict(node: Any) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node:
            assert node.get("additionalProperties") is False
            assert set(node.get("required") or []) == set(node["properties"].keys())
        for val in node.values():
            _assert_all_objects_strict(val)
    elif isinstance(node, list):
        for item in node:
            _assert_all_objects_strict(item)


def test_to_strict_json_schema_compliance() -> None:
    odw_schema = to_strict_json_schema(OdwolanieSchema, exclude_fields={"czynnosc_opis"})
    keys = _collect_keys(odw_schema)
    assert "$defs" not in keys
    assert "$ref" not in keys
    assert "title" not in keys
    assert "default" not in keys
    _assert_all_objects_strict(odw_schema)

    kio_props = odw_schema["properties"]["OdwolanieKIO"]["properties"]
    assert "czynnosc_opis" not in kio_props
    assert "czynnosc_opis" not in odw_schema["properties"]["OdwolanieKIO"]["required"]

    odwolujacy_props = odw_schema["properties"]["Odwolujacy"]["items"]["properties"]
    assert "przez-pelnomocnika" in odwolujacy_props
    assert "przez_pelnomocnika" not in odwolujacy_props
    assert odwolujacy_props["pelnomocnik"]["type"] == "array"

    pel_schema = to_strict_json_schema(PelnomocnictwoSchema, force_string_fields={"pesel"})
    _assert_all_objects_strict(pel_schema)
    pesel_schema = pel_schema["properties"]["pelnomocnicy"]["items"]["properties"]["pesel"]
    assert pesel_schema == {"anyOf": [{"type": "string"}, {"type": "null"}]}

    prz_schema = to_strict_json_schema(PrzelewSchema)
    _assert_all_objects_strict(prz_schema)


def test_hints_and_normalization_polish_formats() -> None:
    text = (
        "Warszawa, dnia 16 czerwca 2025 r.\n"
        "NIP: 999-928-40-65, KRS: 9999405188, REGON: 999642785\n"
        "Ogłoszenie nr 2026/BZP 00999778, TED: 2025/S 188-641009\n"
        "Rachunek: PL 05 4706 8028 7581 5397 2145 5024\n"
        "Kontakt: biuro@poczta.example\n"
    )
    hints = extract_identifier_hints(text)
    assert hints.nip == ["9999284065"]
    assert hints.krs == ["9999405188"]
    assert hints.regon == ["999642785"]
    assert hints.nrb == ["PL05470680287581539721455024"]
    assert hints.bzp == ["2026/BZP00999778"]
    assert hints.ted == ["641009-2025"]
    assert hints.emails == ["biuro@poczta.example"]
    assert hints.dates == ["2025-06-16"]

    norm_przelew = normalize_payload(
        DocumentType.PRZELEW,
        {
            "przelew_bankowy": {
                "data_operacji": "20 marca 2026 roku",
                "kwota": "200,00 PLN",
                "na_rachunek": "PL68 4382 2304 9992 2952 6283 4736",
                "numer_transakcji": "brak",
                "tytul": "opłata skarbowa",
                "z_rachunku": "PL05 4706 8028 7581 5397 2145 5024",
            }
        },
    )
    assert norm_przelew["przelew_bankowy"] == {
        "data_operacji": "2026-03-20",
        "kwota": 200.0,
        "na_rachunek": "PL68438223049992295262834736",
        "numer_transakcji": None,
        "tytul": "opłata skarbowa",
        "z_rachunku": "PL05470680287581539721455024",
    }

    norm_odw = normalize_payload(
        DocumentType.ODWOLANIE,
        {
            "OdwolanieKIO": {
                "data": "16 czerwca 2025 r.",
                "czynnosc_dzien": "10.06.2025",
                "czynnosc_powod_id": 9,
                "miejscowosc": "Warszawa",
            },
            "Odwolujacy": [
                {
                    "nazwa": "Belmoria S.A.",
                    "nip": "9999284065",
                    "krs": "9999405188",
                    "regon": "999642785",
                    "przez-pelnomocnika": 1,
                    "pelnomocnik": {"imie_nazwisko": "", "nazwa": "null", "mail": "-"},
                }
            ],
            "Zamawiajacy": [{"nazwa": "Aurelink S.A."}],
            "ZamowieniePubliczne": {
                "numer_bzp": "2026/BZP 00999778",
                "numer_ted": "2025/S 188-641009",
                "numer_postepowania": "ZS.791.2026",
                "tytul": "Zakup",
            },
        },
    )
    assert norm_odw["OdwolanieKIO"]["data"] == "2025-06-16"
    assert norm_odw["OdwolanieKIO"]["czynnosc_dzien"] == "2025-06-10"
    assert norm_odw["OdwolanieKIO"]["czynnosc_opis"] == CZYNNOSC_OPIS_BY_ID[9]
    assert norm_odw["Odwolujacy"][0]["pelnomocnik"] == []
    assert norm_odw["ZamowieniePubliczne"]["numer_bzp"] == "2026/BZP00999778"
    assert norm_odw["ZamowieniePubliczne"]["numer_ted"] == "641009-2025"

    norm_pel = normalize_payload(
        DocumentType.PELNOMOCNICTWO,
        {
            "mocodawca": [
                {
                    "data": {
                        "imie_nazwisko": "Jan Kowalski",
                        "krs": "9999405188",
                        "nazwa": "Belmoria S.A.",
                        "nip": "9999284065",
                        "typ": "osoba_prawna",
                    },
                    "podpisujacy": [{"imie_nazwisko": "Oliwia Kaczmarek"}],
                }
            ],
            "pelnomocnicy": [{"imie_nazwisko": "Aleksandra Marciniak", "pesel": 85120133547}],
        },
    )
    assert norm_pel["mocodawca"][0]["data"]["imie_nazwisko"] is None
    assert norm_pel["pelnomocnicy"][0]["pesel"] == "85120133547"


def test_validate_extraction_grounding_accepts_synthetic_and_catches_hallucinations() -> None:
    source_text = (
        "Odwołujący: Belmoria S.A., KRS: 9999405188, NIP: 9999284065, REGON: 999642785\n"
        "Zamawiający: Aurelink S.A., NIP: 9997518659\n"
        "Zamawiający w dniu 20 marca 2026 roku unieważnił postępowanie."
    )
    valid_payload = {
        "OdwolanieKIO": {
            "data": None,
            "czynnosc_dzien": "2026-03-20",
            "czynnosc_powod_id": 9,
            "miejscowosc": None,
        },
        "Odwolujacy": [
            {
                "krs": "9999405188",
                "nazwa": "Belmoria S.A.",
                "nip": "9999284065",
                "regon": "999642785",
                "przez-pelnomocnika": 0,
                "pelnomocnik": None,
            }
        ],
        "Zamawiajacy": [{"nazwa": "Aurelink S.A.", "nip": "9997518659"}],
        "ZamowieniePubliczne": {},
    }
    ok = validate_extraction(DocumentType.ODWOLANIE, valid_payload, source_text)
    assert ok.valid is True
    assert ok.errors == []

    bad_payload = json.loads(json.dumps(valid_payload))
    bad_payload["Odwolujacy"][0]["nip"] = "9999284066"
    bad = validate_extraction(DocumentType.ODWOLANIE, bad_payload, source_text)
    assert bad.valid is False
    assert any("Odwolujacy[0].nip='9999284066'" in err and "9999284065" in err for err in bad.errors)


def test_llm_client_strict_request_and_missing_key() -> None:
    live_client = LLMClient(api_key="")
    with pytest.raises(LLMClientError, match="LLM_API_KEY"):
        asyncio.run(
            live_client.complete_json(
                messages=[{"role": "user", "content": "hi"}],
                schema_name="test",
                json_schema={"type": "object", "properties": {}, "additionalProperties": False, "required": []},
            )
        )

    seen_requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "model": "ministral-3b-latest",
                "choices": [{"message": {"content": '{"ok": true}'}}],
                "usage": {"prompt_tokens": 120, "completion_tokens": 15, "total_tokens": 135},
            },
        )

    client = LLMClient(api_key="test-key", model="ministral-3b-latest", transport=httpx.MockTransport(handler))
    res = asyncio.run(
        client.complete_json(
            messages=[{"role": "user", "content": "extract"}],
            schema_name="przelew_extraction",
            json_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
            max_tokens=200,
        )
    )
    assert res.content == '{"ok": true}'
    assert res.usage.total_tokens == 135
    assert res.used_schema_fallback is False
    assert seen_requests[0]["temperature"] == 0.0
    assert seen_requests[0]["response_format"]["type"] == "json_schema"
    assert seen_requests[0]["response_format"]["json_schema"]["strict"] is True


def test_llm_client_schema_fallback_on_http_400() -> None:
    response_formats: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        response_formats.append(body["response_format"])
        if len(response_formats) == 1:
            return httpx.Response(400, text="json_schema unsupported")
        return httpx.Response(
            200,
            json={
                "model": "open-mistral-7b",
                "choices": [{"message": {"content": '{"ok": 1}'}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
            },
        )

    client = LLMClient(api_key="test-key", model="open-mistral-7b", transport=httpx.MockTransport(handler))
    res = asyncio.run(
        client.complete_json(
            messages=[{"role": "user", "content": "extract"}],
            schema_name="fallback_test",
            json_schema={"type": "object", "properties": {}, "required": []},
        )
    )
    assert res.used_schema_fallback is True
    assert response_formats[0]["type"] == "json_schema"
    assert response_formats[1] == {"type": "json_object"}


def test_extract_document_single_shot_przelew_and_odwolanie() -> None:
    przelew_text = (
        "POLECENIE PRZELEWU\n"
        "Rachunek obciążany: PL05470680287581539721455024\n"
        "Rachunek odbiorcy: PL68438223049992295262834736\n"
        "Kwota operacji: 200,00 PLN\n"
        "Data transakcji: 2026-03-21\n"
        "Nr ref.: CEN2604016344389\n"
        "Opis: opłata skarbowa\n"
    )
    odwolanie_text = (
        "Odwołujący: Belmoria Rozwiązania S.A.\n"
        "KRS: 9999405188, NIP: 9999284065, REGON: 999642785\n"
        "reprezentowana przez: r. pr. Aleksandra Marciniak, Kancelaria Adwokacka, biuro@poczta.example\n"
        "Zamawiający: Aurelink Automatyka S.A.\n"
        "KRS: 9999716254, NIP: 9997518659, REGON: 999184816\n"
        "ODWOŁANIE\n"
        "dotyczy: Zakup sprzętu, nr ZS.791.2026, 2026/BZP 00999778, TED: 999137-2026\n"
        "Zamawiający w dniu 20 marca 2026 roku unieważnił postępowanie.\n"
    )

    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        req_body = json.loads(request.content.decode("utf-8"))
        schema_name = req_body["response_format"]["json_schema"]["name"]
        if schema_name.startswith("przelew"):
            payload = {
                "przelew_bankowy": {
                    "data_operacji": "2026-03-21",
                    "kwota": 200.0,
                    "na_rachunek": "PL68438223049992295262834736",
                    "numer_transakcji": "CEN2604016344389",
                    "tytul": "opłata skarbowa",
                    "z_rachunku": "PL05470680287581539721455024",
                }
            }
        else:
            payload = {
                "OdwolanieKIO": {
                    "data": None,
                    "czynnosc_dzien": "2026-03-20",
                    "czynnosc_powod_id": 9,
                    "miejscowosc": None,
                },
                "Odwolujacy": [
                    {
                        "krs": "9999405188",
                        "nazwa": "Belmoria Rozwiązania S.A.",
                        "nip": "9999284065",
                        "regon": "999642785",
                        "przez-pelnomocnika": 1,
                        "pelnomocnik": {
                            "imie_nazwisko": "Aleksandra Marciniak",
                            "nazwa": "Kancelaria Adwokacka",
                            "mail": "biuro@poczta.example",
                        },
                    }
                ],
                "Zamawiajacy": [
                    {
                        "krs": "9999716254",
                        "nazwa": "Aurelink Automatyka S.A.",
                        "nip": "9997518659",
                        "regon": "999184816",
                        "prowadzacy_postepowanie": None,
                    }
                ],
                "ZamowieniePubliczne": {
                    "numer_bzp": "2026/BZP 00999778",
                    "numer_postepowania": "ZS.791.2026",
                    "numer_ted": "999137-2026",
                    "tytul": "Zakup sprzętu",
                },
            }
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps(payload, ensure_ascii=False)}}],
                "usage": {"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280},
            },
        )

    client = LLMClient(api_key="test-key", transport=httpx.MockTransport(handler))

    res_prz = asyncio.run(extract_document(DocumentType.PRZELEW, przelew_text, client=client, mode="single"))
    assert res_prz.valid is True
    assert res_prz.attempts == 1
    assert PrzelewSchema.model_validate(res_prz.data).przelew_bankowy.z_rachunku == "PL05470680287581539721455024"

    res_odw = asyncio.run(extract_document(DocumentType.ODWOLANIE, odwolanie_text, client=client, mode="single"))
    assert res_odw.valid is True
    assert calls == 2
    assert res_odw.attempts == 1
    parsed_odw = OdwolanieSchema.model_validate(res_odw.data)
    assert parsed_odw.OdwolanieKIO.czynnosc_opis == CZYNNOSC_OPIS_BY_ID[9]
    assert parsed_odw.ZamowieniePubliczne.numer_bzp == "2026/BZP00999778"


def test_extract_document_targeted_retry_fixes_hallucinated_nip() -> None:
    source_text = (
        "Odwołujący: Belmoria S.A., NIP: 9999284065\n"
        "Zamawiający: Aurelink S.A., NIP: 9997518659\n"
        "Zamawiający w dniu 2026-03-20 unieważnił postępowanie."
    )
    prompts_seen: list[str] = []
    schemas_seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        prompts_seen.append(body["messages"][-1]["content"])
        schemas_seen.append(body["response_format"]["json_schema"])
        if len(prompts_seen) == 1:
            payload: dict[str, Any] = {
                "OdwolanieKIO": {
                    "data": None,
                    "czynnosc_dzien": "2026-03-20",
                    "czynnosc_powod_id": 9,
                    "miejscowosc": None,
                },
                "Odwolujacy": [
                    {
                        "krs": None,
                        "nazwa": "Belmoria S.A.",
                        "nip": "9999284066",
                        "regon": None,
                        "przez-pelnomocnika": 0,
                        "pelnomocnik": None,
                    }
                ],
                "Zamawiajacy": [
                    {
                        "krs": None,
                        "nazwa": "Aurelink S.A.",
                        "nip": "9997518659",
                        "regon": None,
                        "prowadzacy_postepowanie": None,
                    }
                ],
                "ZamowieniePubliczne": {
                    "numer_bzp": None,
                    "numer_postepowania": None,
                    "numer_ted": None,
                    "tytul": None,
                },
            }
            usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        else:
            payload = {"Odwolujacy_0__nip": "9999284065"}
            usage = {"prompt_tokens": 60, "completion_tokens": 15, "total_tokens": 75}

        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps(payload)}}],
                "usage": usage,
            },
        )

    client = LLMClient(api_key="test-key", transport=httpx.MockTransport(handler))
    res = asyncio.run(
        extract_document(DocumentType.ODWOLANIE, source_text, client=client, mode="single", max_retries=1)
    )
    assert res.valid is True
    assert res.attempts == 2
    assert res.usage.total_tokens == 225
    assert schemas_seen[1]["name"] == "odwolanie_patch"
    assert list(schemas_seen[1]["schema"]["properties"].keys()) == ["Odwolujacy_0__nip"]
    assert res.data["Odwolujacy"][0]["nip"] == "9999284065"
    assert res.data["Zamawiajacy"][0]["nip"] == "9997518659"
    assert "9999284066" in prompts_seen[1]
    assert "9999284065" in prompts_seen[1]


def test_single_vs_multi_zamawiajacy_normalization() -> None:
    base = {
        "OdwolanieKIO": {"data": None, "czynnosc_dzien": None, "czynnosc_powod_id": 4, "miejscowosc": None},
        "Odwolujacy": [{"nazwa": "Wykonawca", "przez-pelnomocnika": 0, "pelnomocnik": None}],
        "ZamowieniePubliczne": {},
    }
    single = normalize_payload(
        DocumentType.ODWOLANIE,
        {**base, "Zamawiajacy": [{"nazwa": "Zam1", "prowadzacy_postepowanie": 1}]},
    )
    assert single["Zamawiajacy"][0]["prowadzacy_postepowanie"] is None

    multi = normalize_payload(
        DocumentType.ODWOLANIE,
        {
            **base,
            "Zamawiajacy": [
                {"nazwa": "Zam1", "prowadzacy_postepowanie": 1},
                {"nazwa": "Zam2", "prowadzacy_postepowanie": 0},
            ],
        },
    )
    assert [z["prowadzacy_postepowanie"] for z in multi["Zamawiajacy"]] == [1, 0]


def test_full_repair_fallback_on_invalid_json() -> None:
    source_text = (
        "POLECENIE PRZELEWU\nKwota: 200,00 PLN\n"
        "Z: PL05470680287581539721455024\nNa: PL68438223049992295262834736\n"
    )
    schema_names: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        schema_names.append(body["response_format"]["json_schema"]["name"])
        if len(schema_names) == 1:
            content = "{not valid json"
        else:
            content = json.dumps(
                {
                    "przelew_bankowy": {
                        "data_operacji": None,
                        "kwota": 200.0,
                        "na_rachunek": "68438223049992295262834736",
                        "numer_transakcji": None,
                        "tytul": None,
                        "z_rachunku": "05470680287581539721455024",
                    }
                }
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 50, "completion_tokens": 20}},
        )

    client = LLMClient(api_key="test-key", transport=httpx.MockTransport(handler))
    res = asyncio.run(extract_document(DocumentType.PRZELEW, source_text, client=client, max_retries=1))
    assert res.valid is True
    assert res.attempts == 2
    assert schema_names == ["przelew_extraction", "przelew_repair"]


def test_extract_document_section_swarm_for_odwolanie() -> None:
    odwolanie_text = (
        "Warszawa, dnia 16 czerwca 2025 r.\n"
        "Odwołujący: Belmoria S.A., KRS: 9999405188, NIP: 9999284065\n"
        "Zamawiający: Aurelink S.A., NIP: 9997518659\n"
        "ODWOŁANIE\n"
        "Dotyczy: Dostawa sprzętu, nr RZP.27.29.2025, 2025/BZP 00270698\n"
        "Działając w imieniu Odwołującego...\n"
        "Zarzut nr 1: Zamawiający w dniu 10 czerwca 2025 r. dokonał publikacji specyfikacji warunków zamówienia (SWZ).\n"
        "UZASADNIENIE\n"
        "W wyroku KIO z dnia 4 marca 2024 r. wskazano...\n"
    )
    schema_names: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        schema_name = body["response_format"]["json_schema"]["name"]
        schema_names.append(schema_name)
        if schema_name == "odwolanie_header":
            payload = {
                "data": "2025-06-16",
                "miejscowosc": "Warszawa",
                "Odwolujacy": [
                    {
                        "krs": "9999405188",
                        "nazwa": "Belmoria S.A.",
                        "nip": "9999284065",
                        "regon": None,
                        "przez-pelnomocnika": 0,
                        "pelnomocnik": None,
                    }
                ],
                "Zamawiajacy": [
                    {
                        "krs": None,
                        "nazwa": "Aurelink S.A.",
                        "nip": "9997518659",
                        "regon": None,
                        "prowadzacy_postepowanie": None,
                    }
                ],
                "ZamowieniePubliczne": {
                    "numer_bzp": "2025/BZP 00270698",
                    "numer_postepowania": "RZP.27.29.2025",
                    "numer_ted": None,
                    "tytul": "Dostawa sprzętu",
                },
            }
            usage = {"prompt_tokens": 110, "completion_tokens": 70, "total_tokens": 180}
        else:
            payload = {"czynnosc_dzien": "2025-06-10", "czynnosc_powod_id": 4}
            usage = {"prompt_tokens": 60, "completion_tokens": 20, "total_tokens": 80}

        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(payload)}}], "usage": usage},
        )

    client = LLMClient(api_key="test-key", transport=httpx.MockTransport(handler))
    res = asyncio.run(
        extract_document(DocumentType.ODWOLANIE, odwolanie_text, client=client, mode="section_swarm")
    )
    assert res.valid is True
    assert res.mode == "section_swarm"
    assert res.attempts == 2
    assert set(schema_names) == {"odwolanie_header", "odwolanie_action"}
    assert res.usage.total_tokens == 260
    parsed = OdwolanieSchema.model_validate(res.data)
    assert parsed.OdwolanieKIO.czynnosc_opis == CZYNNOSC_OPIS_BY_ID[4]
    assert parsed.ZamowieniePubliczne.numer_bzp == "2025/BZP00270698"


def test_negative_and_empty_documents_use_zero_tokens() -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(500)

    client = LLMClient(api_key="test-key", transport=httpx.MockTransport(handler))

    res_inny = asyncio.run(extract_document(DocumentType.INNY, "Decyzja Ministra Sportu", client=client))
    assert res_inny.valid is True
    assert res_inny.data == {}
    assert res_inny.attempts == 0
    assert res_inny.usage.total_tokens == 0

    res_empty = asyncio.run(extract_document(DocumentType.PRZELEW, "   \n  ", client=client))
    assert res_empty.data == {}
    assert res_empty.attempts == 0

    res_seg = asyncio.run(
        extract_structured_data([DocumentSegment(doc_type=DocumentType.INNY, text="cokolwiek")], client=client)
    )
    assert res_seg == {}
    assert called is False


def test_polish_checksum_validators() -> None:
    assert is_valid_nip("5262239325") is True
    assert is_valid_nip("9999284065") is True
    assert is_valid_nip("5262239326") is False
    assert is_valid_nip("1234567890") is False  # mod 11 == 10 is invalid

    assert is_valid_regon("000279077") is True
    assert is_valid_regon("000279078") is False
    assert is_valid_regon("12345678512347") is True
    assert is_valid_regon("12345678512348") is False

    assert is_valid_pesel("44051401359") is True
    assert is_valid_pesel("44051401358") is False
    assert is_valid_pesel("93198") is False

    assert is_valid_nrb("60101010100081362231000000") is True
    assert is_valid_nrb("60101010100081362231000001") is False
    assert is_valid_nrb("DE89 3704 0044 0532 0130 00") is True
    assert is_valid_nrb("DE89 3704 0044 0532 0130 01") is False
    assert is_valid_nrb("GB82 WEST 1234 5698 7654 32") is True

    intl_text = (
        "Z rachunku: DE89 3704 0044 0532 0130 00\n"
        "Na rachunek: PL60 1010 1010 0081 3622 3100 0000\n"
        "Kwota: 15000,00 PLN\nData: 2026-05-10\n"
    )
    intl_aug = augment_post_extraction(
        DocumentType.PRZELEW,
        {"przelew_bankowy": {"data_operacji": "2026-05-10", "kwota": 15000.0, "numer_transakcji": "TX1", "tytul": "wpis"}},
        intl_text,
    )
    intl_val = validate_extraction(DocumentType.PRZELEW, intl_aug, intl_text, strict_checksums=True)
    assert intl_val.valid is True
    assert intl_val.data is not None
    assert intl_val.data["przelew_bankowy"]["z_rachunku"] == "DE89370400440532013000"
    assert intl_val.data["przelew_bankowy"]["na_rachunek"] == "PL60101010100081362231000000"


def test_identifier_length_and_strict_checksum_modes() -> None:
    przelew_text = (
        "Rachunek z: PL05470680287581539721455024\n"
        "Rachunek na: PL68438223049992295262834736\n"
    )
    bad_len_przelew = {
        "przelew_bankowy": {
            "data_operacji": None,
            "kwota": 200.0,
            "na_rachunek": "68438223049992295262",
            "numer_transakcji": None,
            "tytul": None,
            "z_rachunku": "05470680287581539721455024",
        }
    }
    res_bad_len = validate_extraction(DocumentType.PRZELEW, bad_len_przelew, przelew_text, strict_checksums=False)
    assert res_bad_len.valid is False
    assert "przelew_bankowy.na_rachunek" in res_bad_len.failed_fields

    synthetic_przelew = {
        "przelew_bankowy": {
            "data_operacji": None,
            "kwota": 200.0,
            "na_rachunek": "68438223049992295262834736",
            "numer_transakcji": None,
            "tytul": None,
            "z_rachunku": "05470680287581539721455024",
        }
    }
    res_lenient = validate_extraction(DocumentType.PRZELEW, synthetic_przelew, przelew_text, strict_checksums=False)
    assert res_lenient.valid is True
    assert len(res_lenient.warnings) == 2

    res_strict = validate_extraction(DocumentType.PRZELEW, synthetic_przelew, przelew_text, strict_checksums=True)
    assert res_strict.valid is False
    assert set(res_strict.failed_fields) == {"przelew_bankowy.z_rachunku", "przelew_bankowy.na_rachunek"}

    pel_text = "Mocodawca: Firma X, NIP: 5262239325\nPełnomocnik: Jan Nowak, PESEL: 44051401359"
    pel_bad_pesel = {
        "mocodawca": [{"data": {"nazwa": "Firma X", "nip": "5262239325", "typ": "osoba_prawna"}, "podpisujacy": []}],
        "pelnomocnicy": [{"imie_nazwisko": "Jan Nowak", "pesel": "93198"}],
    }
    res_pel = validate_extraction(DocumentType.PELNOMOCNICTWO, pel_bad_pesel, pel_text, strict_checksums=False)
    assert res_pel.valid is False
    assert "pelnomocnicy[0].pesel" in res_pel.failed_fields


def test_augmenter_helpers_and_post_extraction() -> None:
    assert strip_person_titles("r. pr. Aleksandra Marciniak") == "Aleksandra Marciniak"
    assert strip_person_titles("Aleksandra Marciniak, r. pr.") == "Aleksandra Marciniak"
    assert strip_person_titles("adw. Zbigniew Górnicki") == "Zbigniew Górnicki"

    raw_header = (
        "Prezes Krajowej Izby Odwoławczej\n"
        "ul. Postępu 17A\n"
        "02-676 Warszawa\n"
        "Odwołujący:\nBelmoria S.A.\n"
        "Zamawiający:\nAurelink S.A.\n"
        "ODWOŁANIE\ndotyczy: Zakup sprzętu"
    )
    stripped = strip_kio_addressee(raw_header)
    assert "Prezes Krajowej Izby Odwoławczej" not in stripped
    assert "02-676 Warszawa" not in stripped

    annotated = annotate_odwolanie_header(raw_header)
    assert "[BRAK DATY I MIEJSCOWOŚCI W NAGŁÓWKU -> data=null, miejscowosc=null]" in annotated
    assert "<<< SEKCJA: ODWOŁUJĄCY >>>" in annotated
    assert "<<< SEKCJA: ZAMAWIAJĄCY >>>" in annotated

    aug_odw = augment_post_extraction(
        DocumentType.ODWOLANIE,
        {
            "OdwolanieKIO": {"data": "2026-03-20", "czynnosc_dzien": "2026-03-20", "czynnosc_powod_id": 9, "miejscowosc": "Warszawa"},
            "Odwolujacy": [
                {"nazwa": "Belmoria S.A.", "nip": "9999284065", "przez-pelnomocnika": 1, "pelnomocnik": {"imie_nazwisko": "Aleksandra Marciniak, r. pr."}},
                {"nazwa": "Aurelink S.A.", "nip": "9997518659", "przez-pelnomocnika": 0, "pelnomocnik": None},
            ],
            "Zamawiajacy": [{"nazwa": "Aurelink S.A.", "nip": "9997518659", "prowadzacy_postepowanie": 1}],
            "ZamowieniePubliczne": {"numer_bzp": None, "numer_ted": None},
        },
        raw_header + "\n2026/BZP 00999778\nTED: 999137-2026",
    )
    assert len(aug_odw["Odwolujacy"]) == 1
    assert aug_odw["Odwolujacy"][0]["nazwa"] == "Belmoria S.A."
    assert aug_odw["Odwolujacy"][0]["pelnomocnik"] == [{"imie_nazwisko": "Aleksandra Marciniak", "mail": None}]
    assert aug_odw["Zamawiajacy"][0]["prowadzacy_postepowanie"] is None
    assert aug_odw["OdwolanieKIO"]["data"] is None
    assert aug_odw["OdwolanieKIO"]["miejscowosc"] is None
    assert aug_odw["ZamowieniePubliczne"]["numer_bzp"] == "2026/BZP00999778"
    assert aug_odw["ZamowieniePubliczne"]["numer_ted"] == "999137-2026"

    aug_pel = augment_post_extraction(
        DocumentType.PELNOMOCNICTWO,
        {
            "mocodawca": [{"data": {"nazwa": "Firma", "typ": "osoba_prawna"}, "podpisujacy": [{"imie_nazwisko": "dr Oliwia Kaczmarek"}]}],
            "pelnomocnicy": [{"imie_nazwisko": "r. pr. Aleksandra Marciniak", "pesel": "93198"}],
        },
        "Pełnomocnik: r. pr. Aleksandra Marciniak (nr wpisu na listę: WA 93198)",
    )
    assert aug_pel["mocodawca"][0]["podpisujacy"][0]["imie_nazwisko"] == "Oliwia Kaczmarek"
    assert aug_pel["pelnomocnicy"][0]["imie_nazwisko"] == "Aleksandra Marciniak"
    assert aug_pel["pelnomocnicy"][0]["pesel"] is None


def test_deterministic_header_extractor_all_60_training_cases() -> None:
    dane_dir = Path("dane")
    if not dane_dir.exists():
        pytest.skip("dane/ directory not available")
    pdf_paths = sorted(dane_dir.glob("przyklad_*/odwolanie.pdf"))
    assert len(pdf_paths) == 60
    for pdf_path in pdf_paths:
        expected = json.loads(pdf_path.with_suffix(".json").read_text(encoding="utf-8"))
        text = extract_text_from_pdf_file(pdf_path)
        city, iso_date = extract_header_city_and_date(text)
        assert city == expected["OdwolanieKIO"]["miejscowosc"], f"City mismatch in {pdf_path}"
        assert iso_date == expected["OdwolanieKIO"]["data"], f"Date mismatch in {pdf_path}"


def test_yaml_document_configs_and_extensibility(tmp_path: Path) -> None:
    configs = get_all_document_configs()
    assert set(configs.keys()) == {"odwolanie", "pelnomocnictwo", "przelew"}
    assert configs["odwolanie"].ekstrakcja.section_swarm is not None
    assert "czynnosc_dict" in configs["odwolanie"].ekstrakcja.section_swarm.action_system_prompt

    repair_prompts = get_repair_prompts()
    assert repair_prompts.surgical_repair_system.strip()
    assert "{error_lines}" in repair_prompts.surgical_repair_user_template
    assert "{previous_json}" in repair_prompts.full_repair_user_template

    custom_yaml = tmp_path / "wybor_oferty.yaml"
    custom_yaml.write_text(
        """id: "wybor_oferty"
nazwa: "Informacja o wyborze oferty"
ekstrakcja:
  schema_name: "WyborOferty"
  max_tokens: 200
  max_chars: 2000
  system_prompt: "Ekstrahuj nazwę wybranego wykonawcy."
  json_schema:
    type: object
    properties:
      wykonawca:
        type: string
    required:
      - wykonawca
    additionalProperties: false
""",
        encoding="utf-8",
    )
    loaded = get_all_document_configs(tmp_path)
    assert list(loaded.keys()) == ["wybor_oferty"]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"wykonawca": "Alfa Sp. z o.o."}'}}]},
        )

    client = LLMClient(api_key="test-key", transport=httpx.MockTransport(handler))
    res = asyncio.run(
        extract_document(
            "wybor_oferty",
            "Wybrano ofertę Alfa Sp. z o.o.",
            client=client,
            config=loaded["wybor_oferty"],
        )
    )
    assert res.valid is True
    assert res.data == {"wykonawca": "Alfa Sp. z o.o."}


def test_bielik_refinements_empty_zamawiajacy_placeholders_and_second_rep() -> None:
    text = (
        "Odwołujący: Firma A, NIP: 9999284065\n"
        "Zamawiający: Urząd X, NIP: 9997518659\n"
        "pełnomocnik: r. pr. Bartosz Szymanski, adw. Cezary Wierzbicki"
    )
    empty_zam = {
        "OdwolanieKIO": {"data": None, "czynnosc_dzien": None, "czynnosc_powod_id": 1, "miejscowosc": None},
        "Odwolujacy": [
            {
                "nazwa": "Firma A",
                "nip": "9999284065",
                "przez-pelnomocnika": 1,
                "pelnomocnik": {"imie_nazwisko": "Bartosz Szymanski, adw. Cezary Wierzbicki", "nazwa": None, "mail": None},
            }
        ],
        "Zamawiajacy": [],
        "ZamowieniePubliczne": {"numer_bzp": "YYYY/BZP########", "numer_ted": "NNNNNN-YYYY"},
    }
    augmented = augment_post_extraction(DocumentType.ODWOLANIE, empty_zam, text)
    assert [r["imie_nazwisko"] for r in augmented["Odwolujacy"][0]["pelnomocnik"]] == [
        "Bartosz Szymanski",
        "Cezary Wierzbicki",
    ]

    outcome = validate_extraction(DocumentType.ODWOLANIE, augmented, text)
    assert outcome.valid is False
    assert "Zamawiajacy" in outcome.failed_fields

    norm_with_placeholder = normalize_payload(DocumentType.ODWOLANIE, augmented)
    assert norm_with_placeholder["ZamowieniePubliczne"]["numer_bzp"] is None
    assert norm_with_placeholder["ZamowieniePubliczne"]["numer_ted"] is None


def test_a1_deterministic_id_binding_across_all_60_cases() -> None:
    dane_dir = Path("dane")
    if not dane_dir.exists():
        pytest.skip("dane/ directory not available")

    for pdf_path in sorted(dane_dir.glob("przyklad_*/odwolanie.pdf")):
        expected = json.loads(pdf_path.with_suffix(".json").read_text(encoding="utf-8"))
        text = extract_text_from_pdf_file(pdf_path)
        minimal_llm = {
            "OdwolanieKIO": {
                "czynnosc_dzien": expected["OdwolanieKIO"]["czynnosc_dzien"],
                "czynnosc_powod_id": expected["OdwolanieKIO"]["czynnosc_powod_id"],
            },
            "Odwolujacy": [
                {
                    "nazwa": o["nazwa"],
                    "przez-pelnomocnika": o["przez-pelnomocnika"],
                    "pelnomocnik": (
                        [
                            {
                                "imie_nazwisko": o["pelnomocnik"]["imie_nazwisko"],
                                "nazwa": o["pelnomocnik"]["nazwa"],
                            }
                        ]
                        if o.get("pelnomocnik")
                        else []
                    ),
                }
                for o in expected["Odwolujacy"]
            ],
            "Zamawiajacy": [
                {
                    "nazwa": z["nazwa"],
                    "prowadzacy_postepowanie": z["prowadzacy_postepowanie"],
                }
                for z in expected["Zamawiajacy"]
            ],
            "ZamowieniePubliczne": {
                "numer_postepowania": expected["ZamowieniePubliczne"]["numer_postepowania"],
                "tytul": expected["ZamowieniePubliczne"]["tytul"],
            },
        }
        aug = normalize_payload(
            DocumentType.ODWOLANIE,
            augment_post_extraction(DocumentType.ODWOLANIE, minimal_llm, text),
        )
        assert aug["OdwolanieKIO"] == expected["OdwolanieKIO"], f"A1 OdwolanieKIO mismatch in {pdf_path}"
        assert aug["Zamawiajacy"] == expected["Zamawiajacy"], f"A1 Zamawiajacy mismatch in {pdf_path}"
        assert aug["ZamowieniePubliczne"] == expected["ZamowieniePubliczne"], f"A1 ZamowieniePubliczne mismatch in {pdf_path}"
        assert len(aug["Odwolujacy"]) == len(expected["Odwolujacy"])
        for aug_o, exp_o in zip(aug["Odwolujacy"], expected["Odwolujacy"]):
            for k in ("krs", "nazwa", "nip", "regon", "przez-pelnomocnika"):
                assert aug_o[k] == exp_o[k], f"A1 {k} mismatch in {pdf_path}"
            exp_rep = exp_o.get("pelnomocnik")
            if exp_rep:
                assert isinstance(aug_o["pelnomocnik"], list) and len(aug_o["pelnomocnik"]) >= 1
                assert aug_o["pelnomocnik"][0] == exp_rep, f"A1 pelnomocnik[0] mismatch in {pdf_path}"
                for rep_entry in aug_o["pelnomocnik"]:
                    assert rep_entry.get("imie_nazwisko")
                    assert rep_entry.get("nazwa") == exp_rep.get("nazwa")
                    assert rep_entry.get("mail") == exp_rep.get("mail")
            else:
                assert aug_o["pelnomocnik"] == []

    for pdf_path in sorted(dane_dir.glob("przyklad_*/pelnomocnictwo.pdf")):
        expected = json.loads(pdf_path.with_suffix(".json").read_text(encoding="utf-8"))
        text = extract_text_from_pdf_file(pdf_path)
        minimal_llm = {
            "mocodawca": [
                {
                    "data": {
                        "imie_nazwisko": m["data"]["imie_nazwisko"],
                        "nazwa": m["data"]["nazwa"],
                        "typ": m["data"]["typ"],
                    },
                    "podpisujacy": m["podpisujacy"],
                }
                for m in expected["mocodawca"]
            ],
            "pelnomocnicy": [{"imie_nazwisko": p["imie_nazwisko"]} for p in expected["pelnomocnicy"]],
        }
        aug = normalize_payload(
            DocumentType.PELNOMOCNICTWO,
            augment_post_extraction(DocumentType.PELNOMOCNICTWO, minimal_llm, text),
        )
        assert aug == expected, f"A1 pelnomocnictwo mismatch in {pdf_path}"

    for pdf_path in sorted(dane_dir.glob("przyklad_*/*przelew*.pdf")):
        ref_path = pdf_path.parent / "przelew.json"
        if not ref_path.is_file():
            ref_path = pdf_path.with_suffix(".json")
        expected = json.loads(ref_path.read_text(encoding="utf-8"))
        text = extract_text_from_pdf_file(pdf_path)
        pb = expected["przelew_bankowy"]
        minimal_llm = {
            "przelew_bankowy": {
                "data_operacji": pb["data_operacji"],
                "kwota": pb["kwota"],
                "numer_transakcji": pb["numer_transakcji"],
                "tytul": pb["tytul"],
            }
        }
        aug = normalize_payload(
            DocumentType.PRZELEW,
            augment_post_extraction(DocumentType.PRZELEW, minimal_llm, text),
        )
        aug_pb = aug["przelew_bankowy"]
        assert aug_pb["z_rachunku"].removeprefix("PL") == pb["z_rachunku"], f"A1 przelew z_rachunku mismatch in {pdf_path}"
        assert aug_pb["na_rachunek"].removeprefix("PL") == pb["na_rachunek"], f"A1 przelew na_rachunek mismatch in {pdf_path}"
        assert aug_pb["data_operacji"] == pb["data_operacji"]
        assert aug_pb["kwota"] == pb["kwota"]
        assert aug_pb["numer_transakcji"] == pb["numer_transakcji"]
        assert aug_pb["tytul"] == pb["tytul"]


def test_field_evidence_quotes_and_parallel_batch() -> None:
    from backend.services.extractor import extract_documents_batch
    from backend.services.normalizer import locate_field_evidence

    pages = [
        "Warszawa, dnia 16 czerwca 2025 r.\nOdwołujący: Belmoria S.A., NIP: 999-928-40-65\nZamawiający: Aurelink S.A., NIP: 9997518659",
        "Zarzut nr 1: Zamawiający w dniu 10 czerwca 2025 roku unieważnił postępowanie.",
    ]
    full_text = "\n\n".join(pages)
    sample_data = {
        "OdwolanieKIO": {
            "data": "2025-06-16",
            "czynnosc_dzien": "2025-06-10",
            "czynnosc_powod_id": 9,
            "czynnosc_opis": "Decyzja o unieważnieniu postępowania",
            "miejscowosc": "Warszawa",
        },
        "Odwolujacy": [{"nazwa": "Belmoria S.A.", "nip": "9999284065", "krs": None, "regon": None, "przez-pelnomocnika": 0, "pelnomocnik": None}],
        "Zamawiajacy": [{"nazwa": "Aurelink S.A.", "nip": "9997518659", "krs": None, "regon": None, "prowadzacy_postepowanie": None}],
        "ZamowieniePubliczne": {"numer_bzp": None, "numer_postepowania": None, "numer_ted": None, "tytul": None},
    }
    ev_list = locate_field_evidence(sample_data, full_text, pages=pages)
    by_field = {ev.field: ev for ev in ev_list}
    assert by_field["Odwolujacy[0].nip"].page == 1
    assert "999-928-40-65" in (by_field["Odwolujacy[0].nip"].quote or "")
    assert by_field["OdwolanieKIO.czynnosc_dzien"].page == 2
    assert "10 czerwca 2025" in (by_field["OdwolanieKIO.czynnosc_dzien"].quote or "")

    def handler(request: httpx.Request) -> httpx.Response:
        payload = {
            "przelew_bankowy": {
                "data_operacji": "2026-03-21",
                "kwota": 200.0,
                "numer_transakcji": "CEN123",
                "tytul": "opłata",
            }
        }
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(payload)}}], "usage": {"prompt_tokens": 30, "completion_tokens": 15}},
        )

    client = LLMClient(api_key="test-key", transport=httpx.MockTransport(handler))
    przelew_txt = "Kwota: 200,00 PLN\nData: 2026-03-21\nNr: CEN123\nTytuł: opłata\nZ: PL05470680287581539721455024\nNa: PL68438223049992295262834736"
    outcomes = asyncio.run(
        extract_documents_batch(
            [
                (DocumentType.PRZELEW, przelew_txt, [przelew_txt]),
                (DocumentType.PRZELEW, przelew_txt, [przelew_txt]),
                (DocumentType.INNY, "niezwiązany dokument", ["niezwiązany dokument"]),
            ],
            client=client,
            max_concurrency=2,
        )
    )
    assert len(outcomes) == 3
    assert outcomes[0].valid is True
    assert outcomes[0].evidence
    assert "evidence" not in outcomes[0].data
    assert PrzelewSchema.model_validate(outcomes[0].data).przelew_bankowy.z_rachunku == "PL05470680287581539721455024"
    assert outcomes[2].mode == "skipped"


def test_format_uzp_json_two_decimal_amount() -> None:
    formatted_200 = format_uzp_json({"przelew_bankowy": {"kwota": 200.0}})
    assert '"kwota": 200.00' in formatted_200
    assert json.loads(formatted_200)["przelew_bankowy"]["kwota"] == 200.0

    formatted_17_5 = format_uzp_json({"przelew_bankowy": {"kwota": 17.5}})
    assert '"kwota": 17.50' in formatted_17_5
    assert json.loads(formatted_17_5)["przelew_bankowy"]["kwota"] == 17.5


def test_invalid_nullable_field_becomes_null_after_retries() -> None:
    bad_payload = {
        "przelew_bankowy": {
            "data_operacji": "2026-03-21",
            "kwota": 200.0,
            "na_rachunek": "99999999999999999999999999",
            "numer_transakcji": "CEN2604016344389",
            "tytul": "opłata skarbowa",
            "z_rachunku": "05470680287581539721455024",
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(bad_payload)}}], "usage": {"prompt_tokens": 10, "completion_tokens": 15}},
        )

    client = LLMClient(api_key="test-key", transport=httpx.MockTransport(handler))
    text = (
        "Potwierdzenie przelewu\n"
        "Data operacji: 2026-03-21\n"
        "Kwota: 200,00 PLN\n"
        "Numer transakcji: CEN2604016344389\n"
        "Tytuł: opłata skarbowa\n"
        "Z rachunku: 05470680287581539721455024\n"
    )
    outcome = asyncio.run(
        extract_document(DocumentType.PRZELEW, text, client=client, mode="single", max_retries=0)
    )
    assert outcome.data["przelew_bankowy"]["na_rachunek"] is None
    assert any("Ustawiono null" in w for w in outcome.validation_warnings)


def test_augment_przelew_extracts_numer_transakcji_from_ref_labels() -> None:
    src_multiline = "Kwota operacji\n200,00 PLN\nNr ref.\nCEN2604016344389\nZREALIZOWANO"
    src_dotted = "NUMER REFERENCYJNY ............................ CEN2603024646444\nSTATUS: Wykonany"

    out1 = augment_post_extraction(DocumentType.PRZELEW, {"przelew_bankowy": {"numer_transakcji": None}}, src_multiline)
    out2 = augment_post_extraction(DocumentType.PRZELEW, {"przelew_bankowy": {"numer_transakcji": None}}, src_dotted)

    assert out1["przelew_bankowy"]["numer_transakcji"] == "CEN2604016344389"
    assert out2["przelew_bankowy"]["numer_transakcji"] == "CEN2603024646444"

    norm_neg_kwota = validate_extraction(
        DocumentType.PRZELEW,
        {"przelew_bankowy": {"kwota": -17.0}},
        "PRZELEW ELIXIR -17,00 PLN",
    )
    assert norm_neg_kwota.data["przelew_bankowy"]["kwota"] == 17.0




