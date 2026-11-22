"""Batch PDF -> canonical UZP JSON processor for `/opt/solution/input` -> `/opt/solution/output`."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.config import settings  # noqa: E402
from backend.core.models import DocumentType  # noqa: E402
from backend.services.classifier import classify_document  # noqa: E402
from backend.services.extractor import LLMClient, extract_document  # noqa: E402
from backend.services.normalizer import format_uzp_json  # noqa: E402
from backend.services.parser import parse_pdf  # noqa: E402


async def _process_one(
    pdf_path: Path,
    input_root: Path,
    output_root: Path,
    parse_sem: asyncio.Semaphore,
    llm_sem: asyncio.Semaphore,
    client: LLMClient | None,
    mode: str,
) -> dict[str, Any]:
    rel = pdf_path.relative_to(input_root)
    out_path = output_root / rel.with_suffix(".json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    try:
        async with parse_sem:
            parsed = await asyncio.to_thread(parse_pdf, pdf_path.read_bytes())

        doc_type, conf = classify_document(parsed.text)
        doc_label = str(getattr(doc_type, "value", doc_type))

        if doc_type == DocumentType.INNY:
            data: dict[str, Any] = {}
        else:
            async with llm_sem:
                outcome = await extract_document(doc_type, parsed.text, client=client, mode=mode)
            data = outcome.data or {}

        out_path.write_text(format_uzp_json(data) + "\n", encoding="utf-8")
        elapsed = time.perf_counter() - t0
        print(f"[{doc_label:>14}] {elapsed:5.1f}s -> {out_path}", flush=True)
        return {"file": str(rel), "doc_type": doc_label, "confidence": conf, "ok": True}
    except Exception as exc:
        out_path.write_text("{}\n", encoding="utf-8")
        elapsed = time.perf_counter() - t0
        print(f"[         ERROR] {elapsed:5.1f}s {rel}: {type(exc).__name__}: {exc}", flush=True)
        return {"file": str(rel), "doc_type": "error", "ok": False, "error": str(exc)}


async def run_batch(args: argparse.Namespace) -> int:
    input_root = args.input.resolve()
    output_root = args.output.resolve()
    if not input_root.is_dir():
        print(f"Błąd: katalog wejściowy nie istnieje: {input_root}", file=sys.stderr)
        return 1
    output_root.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(input_root.rglob("*.pdf"))
    if not pdfs:
        print(f"Brak plików PDF w {input_root}.")
        return 0

    llm_conc = args.concurrency or settings.llm_max_concurrency
    parse_sem = asyncio.Semaphore(max(1, args.parse_workers))
    llm_sem = asyncio.Semaphore(max(1, llm_conc))
    client = LLMClient(model=args.model) if args.model else None

    print(
        f"Przetwarzanie {len(pdfs)} plików PDF | input={input_root} | output={output_root}"
        f" | model={args.model or settings.llm_model} | concurrency={llm_conc}",
        flush=True,
    )
    t0 = time.perf_counter()
    results = await asyncio.gather(
        *(
            _process_one(pdf, input_root, output_root, parse_sem, llm_sem, client, args.mode)
            for pdf in pdfs
        )
    )
    wall = time.perf_counter() - t0
    ok_count = sum(1 for r in results if r["ok"])
    err_count = len(results) - ok_count
    print(f"\nZakończono w {wall:.1f}s: {ok_count}/{len(results)} OK, błędy={err_count}.", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wsadowe przetwarzanie PDF -> JSON (input/ -> output/).")
    p.add_argument("--input", type=Path, default=Path("/app/input"), help="Katalog wejściowy z plikami PDF")
    p.add_argument("--output", type=Path, default=Path("/app/output"), help="Katalog wyjściowy na pliki JSON")
    p.add_argument("--concurrency", type=int, default=0, help="Maks. liczba równoległych wywołań LLM")
    p.add_argument("--parse-workers", type=int, default=2, help="Maks. liczba równoległych workerów parsowania/OCR")
    p.add_argument("--mode", choices=("single", "section_swarm"), default=settings.llm_extraction_mode)
    p.add_argument("--model", default=None, help="Opcjonalne nadpisanie modelu LLM")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_batch(parse_args())))
