"""Throughput & quality benchmark: PDF batch -> (text layer / OCR) -> classifier -> LLM extraction.

Processes documents concurrently and reports wall time, per-stage timings, tokens,
docs/min, projected time for a target batch (default 180 docs), classification accuracy,
negative rejection rate, and micro-F1 / per-class / field accuracy metrics against
reference JSONs.

Examples:
  python scripts/benchmark.py dane --exclude skany --no-llm
  python scripts/benchmark.py dane --exclude skany --concurrency 8 --output benchmark_report.json
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from collections.abc import Sequence
from copy import deepcopy
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.config import get_all_document_configs, settings  # noqa: E402
from backend.services.classifier import classify_document  # noqa: E402
from backend.services.extractor import LLMClient, extract_document  # noqa: E402
from backend.services.parser import parse_pdf  # noqa: E402

_ARRAY_INDEX_RE = re.compile(r"\[\d+\]")
_BASE_EVAL_CLASSES = ("odwolanie", "pelnomocnictwo", "przelew", "inny")


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


def _find_reference_json(pdf: Path) -> Path | None:
    """Resolve the reference JSON next to a PDF, including `potwierdzenie_przelewu.pdf -> przelew.json`."""
    direct = pdf.with_suffix(".json")
    if direct.is_file():
        return direct
    if pdf.stem == "potwierdzenie_przelewu":
        przelew_ref = pdf.parent / "przelew.json"
        if przelew_ref.is_file():
            return przelew_ref
    return None


def _infer_expected_doc_type(ref_data: dict[str, Any] | None) -> str:
    """Infer canonical expected document class from reference JSON keys (empty -> `inny`)."""
    if not ref_data:
        return "inny"
    if "OdwolanieKIO" in ref_data:
        return "odwolanie"
    if "mocodawca" in ref_data or "pelnomocnicy" in ref_data or "Pelnomocnictwo" in ref_data:
        return "pelnomocnictwo"
    if "przelew_bankowy" in ref_data:
        return "przelew"
    for doc_id, cfg in get_all_document_configs().items():
        roots = {field.split(".", 1)[0] for field in cfg.normalizacja.wymagane_pola if field}
        if roots and roots.issubset(ref_data.keys()):
            return doc_id
    return "inny"


def _normalize_ref_for_eval(ref: dict[str, Any]) -> dict[str, Any]:
    """Coerce legacy single-object `Odwolujacy[].pelnomocnik` synthetic refs into a list."""
    normalized = deepcopy(ref)
    odwolujacy = normalized.get("Odwolujacy")
    if isinstance(odwolujacy, list):
        for item in odwolujacy:
            if not isinstance(item, dict) or "pelnomocnik" not in item:
                continue
            rep = item["pelnomocnik"]
            if isinstance(rep, dict):
                item["pelnomocnik"] = [rep]
            elif rep is None:
                item["pelnomocnik"] = []
    return normalized


def _values_equal(a: Any, b: Any) -> bool:
    """Compare extracted and reference leaf values with numeric/whitespace tolerance."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return round(float(a), 2) == round(float(b), 2)
    if isinstance(a, str) and isinstance(b, str):
        return a.strip() == b.strip()
    return a == b


def _score_document(pred: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    """Compute TP/FP/FN (non-null leaves), exact all-fields accuracy, and field mismatches."""
    norm_ref = _normalize_ref_for_eval(ref)
    p, r = _flatten(pred), _flatten(norm_ref)
    p_nn = {k: v for k, v in p.items() if v is not None and v != ""}
    r_nn = {k: v for k, v in r.items() if v is not None and v != ""}

    tp = fp = fn = 0
    mismatches: list[dict[str, Any]] = []
    for k in sorted(set(p_nn) | set(r_nn)):
        if k in p_nn and k in r_nn and _values_equal(p_nn[k], r_nn[k]):
            tp += 1
        else:
            if k in p_nn:
                fp += 1
            if k in r_nn:
                fn += 1
            mismatches.append({"field": k, "pred": p_nn.get(k), "ref": r_nn.get(k)})

    all_keys = set(p) | set(r)
    fields_ok = sum(1 for k in all_keys if _values_equal(p.get(k), r.get(k)))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "fields_ok": fields_ok,
        "fields_total": len(all_keys),
        "mismatches": mismatches,
    }


def _compute_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Return `(precision, recall, f1)` handling zero denominators cleanly."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def collect_pdfs(paths: Sequence[Path], excludes: Sequence[str] = ()) -> list[Path]:
    """Collect PDF paths recursively, excluding paths containing any substring in `excludes`."""
    pdfs: list[Path] = []
    for p in paths:
        pdfs.extend(sorted(p.rglob("*.pdf")) if p.is_dir() else [p])
    patterns = [ex.lower() for ex in excludes if ex]
    if not patterns:
        return pdfs
    return [p for p in pdfs if not any(pat in p.as_posix().lower() for pat in patterns)]


async def _extract_with_limit(
    doc_type: Any,
    text: str,
    llm_sem: asyncio.Semaphore,
    client: LLMClient | None,
    mode: str,
) -> Any:
    if str(getattr(doc_type, "value", doc_type)) == "inny":
        return await extract_document(doc_type, text, client=client, mode=mode)
    async with llm_sem:
        return await extract_document(doc_type, text, client=client, mode=mode)


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
    pred_type = str(getattr(doc_type, "value", doc_type))

    ref_path = _find_reference_json(pdf)
    ref_data = json.loads(ref_path.read_text(encoding="utf-8")) if ref_path else {}
    expected_doc_type = _infer_expected_doc_type(ref_data)
    row.update(
        doc_type=pred_type,
        expected_doc_type=expected_doc_type,
        cls_ok=(pred_type == expected_doc_type),
        confidence=round(float(conf), 3),
        has_ref=bool(ref_path),
    )

    if not args.no_llm:
        t0 = time.perf_counter()
        pred_data: dict[str, Any] = {}
        try:
            outcome = await _extract_with_limit(doc_type, parsed.text, llm_sem, client, args.mode)
            pred_data = outcome.data or {}
            row.update(valid=outcome.valid, attempts=outcome.attempts, tokens=outcome.usage.total_tokens)
        except Exception as exc:  # report and continue the batch
            row["llm_error"] = f"{type(exc).__name__}: {exc}"
        row.update(_score_document(pred_data, ref_data))
        row["llm_s"] = time.perf_counter() - t0
    row["total_s"] = time.perf_counter() - t_start

    cls_flag = "" if row["cls_ok"] else f"!={expected_doc_type}"
    score_txt = ""
    if "tp" in row:
        if row["tp"] + row["fp"] + row["fn"] > 0:
            _, _, doc_f1 = _compute_prf(row["tp"], row["fp"], row["fn"])
            score_txt = f" F1={doc_f1:.1%} fields={row['fields_ok']}/{row['fields_total']}"
        else:
            score_txt = " neg=OK"
    print(
        f"[{pred_type:>14}{cls_flag}] {row['total_s']:6.1f}s parse={row['parse_s']:.1f}s"
        f" ocr_pages={row['ocr_pages']} llm={row.get('llm_s', 0):.1f}s"
        + score_txt
        + (f" ERR={row['llm_error']}" if "llm_error" in row else "")
        + f"  {pdf}",
        flush=True,
    )
    return row


def _build_summary(
    rows: list[dict[str, Any]],
    wall: float,
    target: int,
    budget_minutes: float,
    no_llm: bool,
) -> dict[str, Any]:
    n = len(rows)
    sum_ = lambda key: sum(r.get(key, 0) or 0 for r in rows)  # noqa: E731
    cls_ok_count = sum(1 for r in rows if r.get("cls_ok"))
    neg_rows = [r for r in rows if r.get("expected_doc_type") == "inny"]
    neg_ok_count = sum(1 for r in neg_rows if r.get("cls_ok"))
    confusions = Counter(
        f"{r['expected_doc_type']} -> {r['doc_type']}"
        for r in rows
        if not r.get("cls_ok")
    )
    projected_s = wall / n * target
    summary: dict[str, Any] = {
        "docs": n,
        "wall_s": round(wall, 3),
        "docs_per_min": round(n / wall * 60, 2) if wall > 0 else 0.0,
        "avg_s_per_doc": round(wall / n, 3),
        "parse_s_sum": round(sum_("parse_s"), 3),
        "ocr_s_sum": round(sum_("ocr_s"), 3),
        "ocr_pages_sum": int(sum_("ocr_pages")),
        "llm_s_sum": round(sum_("llm_s"), 3),
        "cls_ok": cls_ok_count,
        "cls_accuracy": round(cls_ok_count / n, 4),
        "neg_total": len(neg_rows),
        "neg_ok": neg_ok_count,
        "neg_rejection_rate": round(neg_ok_count / len(neg_rows), 4) if neg_rows else None,
        "cls_confusions": dict(confusions.most_common()),
        "projected_s": round(projected_s, 1),
        "budget_minutes": budget_minutes,
        "within_budget": projected_s <= budget_minutes * 60,
    }

    ordered_classes = list(_BASE_EVAL_CLASSES) + sorted(
        {str(r["expected_doc_type"]) for r in rows} - set(_BASE_EVAL_CLASSES)
    )
    per_class: dict[str, dict[str, Any]] = {}
    for cls_name in ordered_classes:
        c_rows = [r for r in rows if r.get("expected_doc_type") == cls_name]
        if not c_rows:
            continue
        c_cls_ok = sum(1 for r in c_rows if r.get("cls_ok"))
        entry: dict[str, Any] = {
            "docs": len(c_rows),
            "cls_ok": c_cls_ok,
            "cls_accuracy": round(c_cls_ok / len(c_rows), 4),
        }
        if not no_llm:
            c_tp = sum(r.get("tp", 0) for r in c_rows)
            c_fp = sum(r.get("fp", 0) for r in c_rows)
            c_fn = sum(r.get("fn", 0) for r in c_rows)
            c_p, c_r, c_f1 = _compute_prf(c_tp, c_fp, c_fn)
            entry.update(
                tp=c_tp,
                fp=c_fp,
                fn=c_fn,
                precision=round(c_p, 4),
                recall=round(c_r, 4),
                f1=round(c_f1, 4),
            )
        per_class[cls_name] = entry
    summary["per_class"] = per_class

    if not no_llm:
        tp, fp, fn = int(sum_("tp")), int(sum_("fp")), int(sum_("fn"))
        precision, recall, f1 = _compute_prf(tp, fp, fn)
        fields_ok, fields_total = int(sum_("fields_ok")), int(sum_("fields_total"))
        mismatch_counts: Counter[str] = Counter()
        mismatch_samples: dict[str, dict[str, Any]] = {}
        for r in rows:
            for item in r.get("mismatches", []):
                pattern = _ARRAY_INDEX_RE.sub("[*]", str(item.get("field", "")))
                mismatch_counts[pattern] += 1
                if pattern not in mismatch_samples:
                    mismatch_samples[pattern] = {"pred": item.get("pred"), "ref": item.get("ref")}
        top_mismatches = [
            {"field": field, "count": count, **mismatch_samples[field]}
            for field, count in mismatch_counts.most_common(10)
        ]
        summary.update(
            tokens=int(sum_("tokens")),
            llm_errors=sum(1 for r in rows if "llm_error" in r),
            valid=sum(1 for r in rows if r.get("valid")),
            tp=tp,
            fp=fp,
            fn=fn,
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            fields_ok=fields_ok,
            fields_total=fields_total,
            field_accuracy=round(fields_ok / fields_total, 4) if fields_total else None,
            top_mismatches=top_mismatches,
        )
    return summary


async def run(args: argparse.Namespace) -> int:
    pdfs = collect_pdfs(args.paths, excludes=args.exclude)
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
        f"{f' | exclude={args.exclude}' if args.exclude else ''}"
        f"{' | BEZ LLM' if args.no_llm else ''}\n"
    )

    t0 = time.perf_counter()
    rows = await asyncio.gather(*(process(p, parse_sem, llm_sem, args, client) for p in pdfs))
    wall = time.perf_counter() - t0

    summary = _build_summary(rows, wall, args.target, args.budget_minutes, args.no_llm)
    print("\n=== Podsumowanie ===")
    print(
        f"Czas całkowity (wall): {summary['wall_s']:.1f}s | {summary['docs_per_min']:.1f} dok./min"
        f" | średnio {summary['avg_s_per_doc']:.2f}s/dok."
    )
    print(
        f"Suma czasów: parsowanie={summary['parse_s_sum']:.1f}s"
        f" (OCR={summary['ocr_s_sum']:.1f}s, stron OCR={summary['ocr_pages_sum']})"
        f" | LLM={summary['llm_s_sum']:.1f}s"
    )
    print(
        f"Klasyfikacja: {summary['cls_ok']}/{summary['docs']} ({summary['cls_accuracy']:.1%})"
        + (
            f" | Odrzucanie negatywnych: {summary['neg_ok']}/{summary['neg_total']}"
            f" ({summary['neg_rejection_rate']:.1%})"
            if summary["neg_total"]
            else ""
        )
    )
    if summary["cls_confusions"]:
        pairs = ", ".join(f"{k}: {v}" for k, v in summary["cls_confusions"].items())
        print(f"Błędy klasyfikacji: {pairs}")

    if not args.no_llm:
        print(
            f"Tokeny: {summary['tokens']} | błędy LLM: {summary['llm_errors']}"
            f" | valid: {summary['valid']}/{summary['docs']}"
        )
        print(
            f"Micro-F1 (niepuste pola): P={summary['precision']:.1%} R={summary['recall']:.1%}"
            f" F1={summary['f1']:.1%} (TP={summary['tp']}, FP={summary['fp']}, FN={summary['fn']})"
        )
        if summary["fields_total"]:
            print(
                f"Zgodność wszystkich pól: {summary['fields_ok']}/{summary['fields_total']}"
                f" ({summary['field_accuracy']:.1%})"
            )

    print("Per klasa:")
    for cls_name, stats in summary["per_class"].items():
        line = (
            f"  - {cls_name:<14} dok={stats['docs']:>3}"
            f" | klas={stats['cls_ok']}/{stats['docs']} ({stats['cls_accuracy']:.1%})"
        )
        if not args.no_llm:
            line += (
                f" | TP={stats['tp']} FP={stats['fp']} FN={stats['fn']}"
                f" P={stats['precision']:.1%} R={stats['recall']:.1%} F1={stats['f1']:.1%}"
            )
        print(line)

    if not args.no_llm and summary.get("top_mismatches"):
        print("Top-10 niezgodnych pól:")
        for m in summary["top_mismatches"]:
            print(f"  - {m['count']:>3}x {m['field']} (pred={m['pred']!r}, ref={m['ref']!r})")

    status = "OK" if summary["within_budget"] else "ZA WOLNO"
    print(
        f"Prognoza dla {args.target} dok.: {summary['projected_s'] / 60:.1f} min"
        f" (limit {args.budget_minutes} min) -> {status}"
    )

    if args.output:
        args.output.write_text(
            json.dumps({"wall_s": wall, "summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark przepustowości i jakości (F1) pipeline'u PDF -> JSON.")
    p.add_argument("paths", nargs="+", type=Path, help="Pliki PDF lub katalogi (rekurencyjnie)")
    p.add_argument("--exclude", action="append", default=[], help="Wyklucz ścieżki zawierające podciąg (np. --exclude skany)")
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
