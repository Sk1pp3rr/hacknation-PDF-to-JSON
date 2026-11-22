"""Unit tests for the `/opt/solution/input -> /opt/solution/output` batch processor."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.core.models import DocumentType
from backend.services.extractor import DocumentExtractionOutcome
from backend.services.parser.pdf_parser import ParsedPDF
from scripts import process_batch


def test_process_batch_mirrors_dirs_handles_negatives_and_formats_kwota(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inp = tmp_path / "input"
    out = tmp_path / "output"
    (inp / "sprawa_1").mkdir(parents=True)
    (inp / "sprawa_2").mkdir(parents=True)

    pos_pdf = inp / "sprawa_1" / "przelew.pdf"
    neg_pdf = inp / "sprawa_2" / "ulotka.pdf"
    err_pdf = inp / "sprawa_2" / "uszkodzony.pdf"
    pos_pdf.write_bytes(b"POS")
    neg_pdf.write_bytes(b"NEG")
    err_pdf.write_bytes(b"ERR")

    def fake_parse_pdf(raw: bytes) -> ParsedPDF:
        if raw == b"ERR":
            raise ValueError("broken pdf")
        txt = "Potwierdzenie przelewu kwota 200.00 PLN" if raw == b"POS" else "Ulotka reklamowa"
        return ParsedPDF(text=txt, page_count=1, pages=[txt])

    def fake_classify(text: str) -> tuple[DocumentType, float]:
        if "przelewu" in text:
            return DocumentType.PRZELEW, 0.99
        return DocumentType.INNY, 0.99

    async def fake_extract(doc_type, text, **kwargs):  # type: ignore[no-untyped-def]
        return DocumentExtractionOutcome(
            doc_type=doc_type,
            valid=True,
            data={"przelew_bankowy": {"kwota": 200.0, "tytul": "Wpis"}},
        )

    monkeypatch.setattr(process_batch, "parse_pdf", fake_parse_pdf)
    monkeypatch.setattr(process_batch, "classify_document", fake_classify)
    monkeypatch.setattr(process_batch, "extract_document", fake_extract)

    args = SimpleNamespace(
        input=inp,
        output=out,
        concurrency=2,
        parse_workers=2,
        mode="section_swarm",
        model=None,
    )
    rc = asyncio.run(process_batch.run_batch(args))

    assert rc == 0
    pos_text = (out / "sprawa_1" / "przelew.json").read_text(encoding="utf-8")
    assert '"kwota": 200.00' in pos_text
    assert json.loads(pos_text) == {"przelew_bankowy": {"kwota": 200.0, "tytul": "Wpis"}}
    assert json.loads((out / "sprawa_2" / "ulotka.json").read_text(encoding="utf-8")) == {}
    assert json.loads((out / "sprawa_2" / "uszkodzony.json").read_text(encoding="utf-8")) == {}
