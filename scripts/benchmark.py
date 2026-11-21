"""Throughput benchmark: PDF batch -> (text layer / OCR) -> classifier -> LLM extraction.

Processes documents concurrently and reports wall time, per-stage timings, tokens,
docs/min, projected time for a target batch (default 180 docs) and field accuracy
when a same-stem reference `.json` exists next to the PDF.

Examples:
  python scripts/benchmark.py dane/skany --no-llm
  python scripts/benchmark.py dane --limit 30 --concurrency 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.config import settings  # noqa: E402
from backend.services.classifier import classify_document  # noqa: E402
from backend.services.extractor import LLMClient, extract_document  # noqa: E402
from backend.services.parser import parse_pdf  # noqa: E402


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(_flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def _compare(pred: dict[str, Any], ref: dict[str, Any]) -> tuple[int, int]:
    p, r = _flatten(pred), _flatten(ref)
    keys = set(p) | set(r)
    return sum(1 for k in keys if p.get(k) == r.get(k)), len(keys)


def collect_pdfs(paths: list[Path]) -> list[Path]:
    pdfs: list[Path] = []
    for p in paths:
        pdfs.extend(sorted(p.rglob("*.pdf")) if p.is_dir() else [p])
    return pdfs


async def process(
    pdf: Path,
    parse_sem: asyncio.Semaphore,
    llm_sem: asyncio.Semaphore,
    args: argparse.Namespace,
    client: LLMClient | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {"file": str(pdf)}
    t_start = time.perf_counter()
    async with parse_sem:
        t0 = time.perf_counter()
        parsed = await asyncio.to_thread(parse_pdf, pdf.read_bytes())
        row["parse_s"] = time.perf_counter() - t0
    row.update(pages=parsed.page_count, ocr_pages=len(parsed.ocr_pages), ocr_s=parsed.ocr_seconds, chars=len(parsed.text))
    if parsed.ocr_error:
        row["ocr_error"] = parsed.ocr_error

    doc_type, conf = classify_document(parsed.text)
    row.update(doc_type=str(getattr(doc_type, "value", doc_type)), confidence=round(float(conf), 3))

    if not args.no_llm:
        async with llm_sem:
            t0 = time.perf_counter()
            try:
                outcome = await extract_document(doc_type, parsed.text, client=client, mode=args.mode)
                row.update(valid=outcome.valid, attempts=outcome.attempts, tokens=outcome.usage.total_tokens)
                ref_path = pdf.with_suffix(".json")
                if ref_path.is_file() and outcome.data:
                    ok, total = _compare(outcome.data, json.loads(ref_path.read_text(encoding="utf-8")))
                    row.update(fields_ok=ok, fields_total=total)
            except Exception as exc:  # report and continue the batch
                row["llm_error"] = f"{type(exc).__name__}: {exc}"
            row["llm_s"] = time.perf_counter() - t0
    row["total_s"] = time.perf_counter() - t_start
    print(
        f"[{row['doc_type']:>14}] {row['total_s']:6.1f}s parse={row['parse_s']:.1f}s"
        f" ocr_pages={row['ocr_pages']} llm={row.get('llm_s', 0):.1f}s"
        + (f" fields={row['fields_ok']}/{row['fields_total']}" if "fields_ok" in row else "")
        + (f" ERR={row['llm_error']}" if "llm_error" in row else "")
        + f"  {pdf}",
        flush=True,
    )
    return row


async def run(args: argparse.Namespace) -> int:
    pdfs = collect_pdfs(args.paths)
    if args.shuffle:
        random.Random(0).shuffle(pdfs)
    if args.limit:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        print("Brak plików PDF.")
        return 1

    llm_conc = args.concurrency or settings.llm_max_concurrency
    parse_sem = asyncio.Semaphore(args.parse_workers)
    llm_sem = asyncio.Semaphore(llm_conc)
    client = LLMClient(model=args.model) if args.model else None
    print(
        f"Dokumenty: {len(pdfs)} | model: {args.model or settings.llm_model} | tryb: {args.mode}"
        f" | LLM równolegle: {llm_conc} | parsowanie równolegle: {args.parse_workers}"
        f" | OCR: {settings.ocr_engine if settings.ocr_enabled else 'wyłączony'}"
        f"{' | BEZ LLM' if args.no_llm else ''}\n"
    )

    t0 = time.perf_counter()
    rows = await asyncio.gather(*(process(p, parse_sem, llm_sem, args, client) for p in pdfs))
    wall = time.perf_counter() - t0

    n = len(rows)
    sum_ = lambda key: sum(r.get(key, 0) or 0 for r in rows)  # noqa: E731
    fields_ok, fields_total = sum_("fields_ok"), sum_("fields_total")
    projected = wall / n * args.target
    print("\n=== Podsumowanie ===")
    print(f"Czas całkowity (wall): {wall:.1f}s | {n / wall * 60:.1f} dok./min | średnio {wall / n:.2f}s/dok.")
    print(f"Suma czasów: parsowanie={sum_('parse_s'):.1f}s (OCR={sum_('ocr_s'):.1f}s, stron OCR={sum_('ocr_pages')}) | LLM={sum_('llm_s'):.1f}s")
    if not args.no_llm:
        print(f"Tokeny: {sum_('tokens')} | błędy LLM: {sum(1 for r in rows if 'llm_error' in r)} | valid: {sum(1 for r in rows if r.get('valid'))}/{n}")
    if fields_total:
        print(f"Zgodność pól z referencją: {fields_ok}/{fields_total} ({fields_ok / fields_total:.1%})")
    status = "OK" if projected <= args.budget_minutes * 60 else "ZA WOLNO"
    print(f"Prognoza dla {args.target} dok.: {projected / 60:.1f} min (limit {args.budget_minutes} min) -> {status}")

    if args.output:
        args.output.write_text(json.dumps({"wall_s": wall, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark przepustowości pipeline'u PDF -> JSON.")
    p.add_argument("paths", nargs="+", type=Path, help="Pliki PDF lub katalogi (rekurencyjnie)")
    p.add_argument("--concurrency", type=int, default=0, help="Równoległe wywołania LLM (domyślnie LLM_MAX_CONCURRENCY)")
    p.add_argument("--parse-workers", type=int, default=4, help="Równoległe parsowanie/OCR dokumentów")
    p.add_argument("--mode", choices=("single", "section_swarm"), default=settings.llm_extraction_mode)
    p.add_argument("--model", default=None, help="Nadpisanie modelu LLM")
    p.add_argument("--limit", type=int, default=0, help="Maks. liczba dokumentów")
    p.add_argument("--shuffle", action="store_true", help="Losowa (deterministyczna) kolejność przed --limit")
    p.add_argument("--no-llm", action="store_true", help="Tylko parsowanie/OCR + klasyfikacja")
    p.add_argument("--target", type=int, default=180, help="Docelowa liczba dokumentów do prognozy")
    p.add_argument("--budget-minutes", type=float, default=30.0, help="Limit czasu dla --target")
    p.add_argument("--output", type=Path, help="Zapis szczegółowego raportu JSON")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
