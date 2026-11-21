"""Unit tests for scoring, reference resolution, and filtering in `scripts/benchmark.py`."""

from __future__ import annotations

from pathlib import Path

from scripts.benchmark import (
    _build_summary,
    _compute_prf,
    _find_reference_json,
    _infer_expected_doc_type,
    _score_document,
    collect_pdfs,
)


def test_find_reference_json(tmp_path: Path) -> None:
    odw_pdf = tmp_path / "odwolanie.pdf"
    odw_json = tmp_path / "odwolanie.json"
    odw_pdf.write_bytes(b"%PDF")
    odw_json.write_text("{}", encoding="utf-8")

    potw_pdf = tmp_path / "potwierdzenie_przelewu.pdf"
    przelew_json = tmp_path / "przelew.json"
    potw_pdf.write_bytes(b"%PDF")
    przelew_json.write_text("{}", encoding="utf-8")

    neg_pdf = tmp_path / "2025_BZP_001.pdf"
    neg_pdf.write_bytes(b"%PDF")

    assert _find_reference_json(odw_pdf) == odw_json
    assert _find_reference_json(potw_pdf) == przelew_json
    assert _find_reference_json(neg_pdf) is None


def test_infer_expected_doc_type() -> None:
    assert _infer_expected_doc_type({"OdwolanieKIO": {"miejscowosc": "Warszawa"}}) == "odwolanie"
    assert _infer_expected_doc_type({"mocodawca": [], "pelnomocnicy": []}) == "pelnomocnictwo"
    assert _infer_expected_doc_type({"przelew_bankowy": {"kwota": 200.0}}) == "przelew"
    assert _infer_expected_doc_type({}) == "inny"
    assert _infer_expected_doc_type(None) == "inny"


def test_score_document_positive_and_legacy_pelnomocnik_and_kwota() -> None:
    ref = {
        "OdwolanieKIO": {"data": "2026-03-01", "czynnosc_powod_id": 4},
        "Odwolujacy": [
            {
                "nazwa": "Firma X Sp. z o.o.",
                "przez-pelnomocnika": 1,
                "pelnomocnik": {
                    "imie_nazwisko": "Jan Kowalski",
                    "nazwa": "Kancelaria JK",
                    "mail": "jk@example.com",
                },
            }
        ],
    }
    pred = {
        "OdwolanieKIO": {"data": "2026-03-01", "czynnosc_powod_id": 4},
        "Odwolujacy": [
            {
                "nazwa": "Firma X Sp. z o.o.",
                "przez-pelnomocnika": 1,
                "pelnomocnik": [
                    {
                        "imie_nazwisko": "Jan Kowalski",
                        "nazwa": "Kancelaria JK",
                        "mail": "jk@example.com",
                    }
                ],
            }
        ],
    }
    scored = _score_document(pred, ref)
    assert scored == {
        "tp": 7,
        "fp": 0,
        "fn": 0,
        "fields_ok": 7,
        "fields_total": 7,
        "mismatches": [],
    }

    przelew_ref = {"przelew_bankowy": {"kwota": 200.0, "data_operacji": "2026-03-21"}}
    przelew_pred = {"przelew_bankowy": {"kwota": 200, "data_operacji": " 2026-03-21 "}}
    przelew_scored = _score_document(przelew_pred, przelew_ref)
    assert przelew_scored["tp"] == 2
    assert przelew_scored["fp"] == 0
    assert przelew_scored["fn"] == 0

    bool_vs_int = _score_document({"flag": True}, {"flag": 1})
    assert bool_vs_int["tp"] == 0
    assert bool_vs_int["fp"] == 1
    assert bool_vs_int["fn"] == 1


def test_score_document_negatives_missed_and_wrong_values() -> None:
    rejected = _score_document({}, {})
    assert rejected == {"tp": 0, "fp": 0, "fn": 0, "fields_ok": 0, "fields_total": 0, "mismatches": []}

    hallucinated = _score_document({"przelew_bankowy": {"kwota": 200.0, "tytul": "Wpis"}}, {})
    assert hallucinated["tp"] == 0
    assert hallucinated["fp"] == 2
    assert hallucinated["fn"] == 0

    missed = _score_document({}, {"przelew_bankowy": {"kwota": 200.0, "tytul": "Wpis"}})
    assert missed["tp"] == 0
    assert missed["fp"] == 0
    assert missed["fn"] == 2

    wrong = _score_document({"przelew_bankowy": {"kwota": 500.0}}, {"przelew_bankowy": {"kwota": 200.0}})
    assert wrong["tp"] == 0
    assert wrong["fp"] == 1
    assert wrong["fn"] == 1
    assert len(wrong["mismatches"]) == 1


def test_compute_prf_edge_cases() -> None:
    assert _compute_prf(0, 0, 0) == (0.0, 0.0, 0.0)
    assert _compute_prf(5, 0, 0) == (1.0, 1.0, 1.0)
    p, r, f1 = _compute_prf(2, 2, 0)
    assert p == 0.5
    assert r == 1.0
    assert abs(f1 - (2 / 3)) < 1e-9


def test_collect_pdfs_excludes_substrings(tmp_path: Path) -> None:
    digital = tmp_path / "przyklad_01" / "odwolanie.pdf"
    scanned = tmp_path / "skany" / "odwolania" / "0001" / "0001.pdf"
    digital.parent.mkdir(parents=True)
    scanned.parent.mkdir(parents=True)
    digital.write_bytes(b"%PDF")
    scanned.write_bytes(b"%PDF")

    all_pdfs = collect_pdfs([tmp_path])
    filtered = collect_pdfs([tmp_path], excludes=["skany"])

    assert all_pdfs == [digital, scanned]
    assert filtered == [digital]


def test_build_summary_aggregates_f1_negatives_and_mismatches() -> None:
    rows = [
        {
            "file": "a/przelew.pdf",
            "doc_type": "przelew",
            "expected_doc_type": "przelew",
            "cls_ok": True,
            "tp": 3,
            "fp": 1,
            "fn": 1,
            "fields_ok": 3,
            "fields_total": 4,
            "valid": True,
            "tokens": 100,
            "mismatches": [{"field": "Odwolujacy[0].nip", "pred": "1", "ref": "2"}],
        },
        {
            "file": "a/neg1.pdf",
            "doc_type": "inny",
            "expected_doc_type": "inny",
            "cls_ok": True,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "fields_ok": 0,
            "fields_total": 0,
            "valid": True,
            "tokens": 0,
            "mismatches": [],
        },
        {
            "file": "a/neg2.pdf",
            "doc_type": "odwolanie",
            "expected_doc_type": "inny",
            "cls_ok": False,
            "tp": 0,
            "fp": 1,
            "fn": 0,
            "fields_ok": 0,
            "fields_total": 1,
            "valid": False,
            "tokens": 50,
            "mismatches": [{"field": "OdwolanieKIO.data", "pred": "2026-01-01", "ref": None}],
        },
    ]
    summary = _build_summary(rows, wall=3.0, target=180, budget_minutes=30.0, no_llm=False)

    assert summary["docs"] == 3
    assert summary["cls_ok"] == 2
    assert summary["neg_total"] == 2
    assert summary["neg_ok"] == 1
    assert summary["neg_rejection_rate"] == 0.5
    assert summary["cls_confusions"] == {"inny -> odwolanie": 1}
    assert summary["tp"] == 3
    assert summary["fp"] == 2
    assert summary["fn"] == 1
    assert summary["per_class"]["przelew"]["f1"] == 0.75
    assert summary["top_mismatches"][0]["field"] == "Odwolujacy[*].nip"

