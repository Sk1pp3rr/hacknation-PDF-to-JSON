"""CLI runner to test the end-to-end PDF -> classifier -> regex hints -> LLM extraction pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.config import settings  # noqa: E402
from backend.services.classifier import classify_document  # noqa: E402
from backend.services.extractor import LLMClient, extract_document  # noqa: E402
from backend.services.normalizer import extract_identifier_hints  # noqa: E402
from backend.services.parser import extract_text_from_pdf_file  # noqa: E402


def _flatten_leaves(obj: Any, prefix: str = "") -> dict[str, Any]:
    leaves: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, val in obj.items():
            path = f"{prefix}.{key}" if prefix else key
            leaves.update(_flatten_leaves(val, path))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            leaves.update(_flatten_leaves(item, f"{prefix}[{idx}]"))
    else:
        leaves[prefix] = obj
    return leaves


def _print_comparison(predicted: dict[str, Any], reference_path: Path) -> None:
    ref = json.loads(reference_path.read_text(encoding="utf-8"))
    pred_leaves = _flatten_leaves(predicted)
    ref_leaves = _flatten_leaves(ref)
    all_keys = sorted(set(pred_leaves) | set(ref_leaves))
    matched = sum(1 for k in all_keys if pred_leaves.get(k) == ref_leaves.get(k))
    print(f"\nPorównanie z {reference_path}: {matched}/{len(all_keys)} pól identycznych")
    for key in all_keys:
        if pred_leaves.get(key) != ref_leaves.get(key):
            print(f"  - {key}: otrzymano={pred_leaves.get(key)!r} | oczekiwano={ref_leaves.get(key)!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test pipeline: PDF -> klasyfikacja -> ekstrakcja LLM.")
    parser.add_argument(
        "pdf_path",
        nargs="?",
        type=Path,
        default=Path("dane/przyklad_01/odwolanie.pdf"),
        help="Ścieżka do pliku PDF (domyślnie dane/przyklad_01/odwolanie.pdf)",
    )
    parser.add_argument(
        "--mode",
        choices=("single", "section_swarm"),
        default=settings.llm_extraction_mode,
        help="Tryb ekstrakcji odwołania",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Opcjonalne nadpisanie nazwy modelu LLM z .env",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Uruchom tylko PDF -> klasyfikator -> podpowiedzi regex (bez wywołania API LLM)",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        help="Opcjonalna ścieżka do referencyjnego pliku .json do porównania pól",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pdf_path: Path = args.pdf_path
    if not pdf_path.is_file():
        print(f"Błąd: nie znaleziono pliku PDF: {pdf_path}")
        return 1

    text = extract_text_from_pdf_file(pdf_path)
    doc_type, confidence = classify_document(text)
    hints = extract_identifier_hints(text)

    print(f"Plik: {pdf_path}")
    print(f"Długość tekstu: {len(text)} znaków")
    print(f"Klasa dokumentu: {doc_type.value} (pewność: {confidence:.4f})")
    print("Podpowiedzi regex:")
    print(hints.format_compact() or "  (brak)")

    if args.dry_run:
        print("\n[dry-run] Pominięto wywołanie API LLM.")
        return 0

    if not settings.llm_api_key or settings.llm_api_key == "your_api_key_here":
        print(
            "\nBrak prawdziwego klucza LLM_API_KEY w pliku .env (obecnie jest brak lub wartość przykładowa).\n"
            "Uzupełnij LLM_API_KEY w .env, a następnie uruchom ponownie bez --dry-run."
        )
        return 2

    active_model = args.model or settings.llm_model
    client = LLMClient(model=args.model) if args.model else None
    t0 = time.perf_counter()
    outcome = asyncio.run(extract_document(doc_type, text, client=client, mode=args.mode))
    elapsed = time.perf_counter() - t0
    print(f"\nModel: {active_model} | Tryb: {outcome.mode} | Czas: {elapsed:.1f}s | valid={outcome.valid} | próby={outcome.attempts}")
    print(
        f"Tokeny: prompt={outcome.usage.prompt_tokens}, "
        f"completion={outcome.usage.completion_tokens}, total={outcome.usage.total_tokens}"
    )
    if outcome.validation_errors:
        print("Błędy walidacji:")
        for err in outcome.validation_errors:
            print(f"  - {err}")
    if outcome.validation_warnings:
        print("Ostrzeżenia walidacji:")
        for warn in outcome.validation_warnings:
            print(f"  - {warn}")

    print("\nWygenerowany JSON:")
    print(json.dumps(outcome.data, ensure_ascii=False, indent=2))

    if args.compare:
        if args.compare.is_file():
            _print_comparison(outcome.data, args.compare)
        else:
            print(f"\nUwaga: nie znaleziono pliku referencyjnego: {args.compare}")

    return 0 if outcome.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
