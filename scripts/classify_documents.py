"""Train and run the TF-IDF document classifier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.services.classifier import (  # noqa: E402
    CLASS_FILENAMES,
    classify_document,
    flatten_json,
    save_classifier,
    train_classifier,
)


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            return flatten_json(json.load(handle))
    return path.read_text(encoding="utf-8")


def iter_training_files(root: Path) -> Iterable[tuple[Path, str]]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if "skany" in path.parts:
            continue
        label = CLASS_FILENAMES.get(path.name.lower())
        if label is None:
            label = CLASS_FILENAMES.get(path.stem.lower())
        if label is not None:
            yield path, label.value
        elif path.suffix.lower() == ".pdf" and ("negatywne" in path.parts or path.parent.name.startswith("przyklad_")):
            yield path, "inny"


def train(train_dir: Path, model_path: Path) -> tuple[int, dict[str, int]]:
    texts: list[str] = []
    labels: list[str] = []
    counts: dict[str, int] = {}
    for path, label in iter_training_files(train_dir):
        text = read_file(path).strip()
        if not text:
            continue
        texts.append(text)
        labels.append(label)
        counts[label] = counts.get(label, 0) + 1

    model = train_classifier(texts, labels)
    saved = save_classifier(model, model_path)
    print(f"Zapisano model: {saved}")
    print(f"Próbki treningowe: {len(texts)}")
    print(json.dumps(counts, ensure_ascii=False, sort_keys=True))
    return len(texts), counts


def input_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(path for path in input_path.rglob("*") if path.is_file())
    raise FileNotFoundError(f"Nie znaleziono ścieżki wejściowej: {input_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Klasyfikacja dokumentów: odwołanie / pełnomocnictwo / przelew."
    )
    parser.add_argument("input", nargs="?", type=Path, help="Plik, PDF lub katalog")
    parser.add_argument("--train", action="store_true", help="Wytrenuj model")
    parser.add_argument("--train-dir", type=Path, default=REPO_ROOT / "dane")
    parser.add_argument("--model", type=Path, default=REPO_ROOT / "models" / "document_classifier.joblib")
    parser.add_argument("--threshold", type=float, default=0.62)
    parser.add_argument("--text", help="Tekst do klasyfikacji zamiast pliku")
    parser.add_argument("--output", type=Path, help="Zapisz wynik jako JSON")
    return parser


def main() -> None:
    args = parse_args().parse_args()

    if args.train:
        train(args.train_dir.resolve(), args.model.resolve())
        if args.input is None and not args.text:
            return

    results: list[dict[str, object]] = []
    if args.text:
        doc_type, confidence = classify_document(args.text, args.model.resolve(), args.threshold)
        results.append({"text": args.text, "class": doc_type.value, "confidence": round(confidence, 4)})
    elif args.input:
        for path in input_paths(args.input.resolve()):
            try:
                text = read_file(path)
            except Exception as exc:
                results.append({"file": str(path), "class": "inny", "confidence": 0.0, "error": str(exc)})
                continue
            doc_type, confidence = classify_document(text, args.model.resolve(), args.threshold)
            results.append({
                "file": str(path),
                "class": doc_type.value,
                "confidence": round(confidence, 4),
            })
    else:
        raise SystemExit("Podaj plik/katalog albo użyj --text.")

    payload = results[0] if len(results) == 1 and args.text else results
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
