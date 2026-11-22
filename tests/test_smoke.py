"""Basic smoke tests for the backend API."""

from fastapi.testclient import TestClient

from backend.main import app
from backend.schemas import DocumentType
from backend.services.classifier import classify_document, classify_and_segment

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_classifier_fallback_examples() -> None:
    appeal, _ = classify_document("Odwołanie do Krajowej Izby Odwoławczej")
    proxy, _ = classify_document("Pełnomocnictwo dla pełnomocnika i mocodawcy")
    transfer, _ = classify_document("Przelew opłaty na rachunek bankowy")

    assert appeal == DocumentType.ODWOLANIE
    assert proxy == DocumentType.PELNOMOCNICTWO
    assert transfer == DocumentType.DOWOD_WPLATY


def test_segmenter_returns_empty_for_empty_text() -> None:
    assert classify_and_segment("") == []


def test_classifier_rejects_negative_notices_and_weak_long_docs() -> None:
    from backend.services.classifier.classifier import fallback_classify

    notice, _ = classify_document(
        "Dziennik Urzędowy Unii Europejskiej\nOgłoszenie o zamówieniu — odwołanie i pełnomocnictwo"
    )
    ruling, _ = classify_document(
        "Sygn. akt KIO 2617/21\nKrajowa Izba Odwoławcza w składzie:\nPrzewodniczący: Jan Kowalski\nWYROK"
    )
    weak_long, _ = fallback_classify(("Informacja podatkowa o rozliczeniu. " * 20) + " kwota do zapłaty.")

    assert notice == DocumentType.INNY
    assert ruling == DocumentType.INNY
    assert weak_long == DocumentType.INNY


def test_extract_document_endpoint_rejects_negative_pdf() -> None:
    from pathlib import Path

    sample = Path("dane/negatywne/16_komisja_ds._regulacji_wynagrodzeń.pdf")
    response = client.post(
        "/api/extract/document",
        files={"file": (sample.name, sample.read_bytes(), "application/pdf")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["data"] == {}
    assert body["meta"]["doc_type"] == "inny"
    assert len(body["meta"]["page_images"]) >= 1
