"""FastAPI entrypoint for Odwołania2json."""

import asyncio
import inspect
from pathlib import Path
import time
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.core.models import ExtractionResult
from backend.services.classifier import classify_and_segment, classify_document
from backend.services.extractor import LLMClientError, extract_document
from backend.services.parser import parse_pdf, render_pdf_explorer_data

app = FastAPI(title=settings.app_name, version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Simple liveness endpoint."""
    return {"status": "ok"}


@app.post("/api/extract", response_model=ExtractionResult)
async def extract_case(file: UploadFile = File(...)) -> ExtractionResult:
    """Accept a PDF file and return extracted canonical UZP JSON."""
    content = await file.read()
    try:
        parsed = await asyncio.to_thread(parse_pdf, content)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Nie udało się odczytać PDF: {exc}") from exc

    doc_type, _ = await asyncio.to_thread(classify_document, parsed.text)
    try:
        outcome = await extract_document(doc_type, parsed.text)
    except LLMClientError as exc:
        raise HTTPException(status_code=502, detail=f"Błąd API LLM: {exc}") from exc

    segments = classify_and_segment(parsed.text)
    return ExtractionResult(
        filename=file.filename or "unknown.pdf",
        documents=segments,
        data=outcome.data,
    )


async def _process_uploaded_pdf(filename: str, content: bytes) -> dict[str, Any]:
    try:
        parsed = await asyncio.to_thread(parse_pdf, content)
    except Exception as exc:  # pypdf raises various error types for broken files
        raise HTTPException(status_code=400, detail=f"Nie udało się odczytać PDF: {exc}") from exc

    doc_type, confidence = await asyncio.to_thread(classify_document, parsed.text)
    t0 = time.perf_counter()
    try:
        extra_kwargs: dict[str, Any] = (
            {"pages": parsed.pages} if "pages" in inspect.signature(extract_document).parameters else {}
        )
        outcome = await extract_document(doc_type, parsed.text, **extra_kwargs)
    except LLMClientError as exc:
        raise HTTPException(status_code=502, detail=f"Błąd API LLM: {exc}") from exc
    elapsed = time.perf_counter() - t0
    raw_evidence = list(getattr(outcome, "evidence", None) or [])
    page_images, evidence_with_rects = await asyncio.to_thread(
        render_pdf_explorer_data,
        content,
        raw_evidence,
    )

    return {
        "filename": filename,
        "data": outcome.data,
        "meta": {
            "doc_type": str(getattr(doc_type, "value", doc_type)),
            "confidence": round(float(confidence), 4),
            "valid": outcome.valid,
            "attempts": outcome.attempts,
            "mode": outcome.mode,
            "model": settings.llm_model,
            "elapsed_seconds": round(elapsed, 2),
            "text_length": len(parsed.text),
            "page_count": parsed.page_count,
            "pages": parsed.pages,
            "page_images": page_images,
            "usage": outcome.usage.model_dump(),
            "evidence": evidence_with_rects,
            "validation_errors": outcome.validation_errors,
            "validation_warnings": outcome.validation_warnings,
        },
    }


@app.post("/api/extract/document")
async def extract_single_document(file: UploadFile = File(...)) -> dict[str, Any]:
    """Run the real pipeline (PDF -> classifier -> LLM extraction) for one PDF document.

    Returns the UZP JSON (`data`) separated from traceability metadata
    (class, confidence, page/quote evidence, validation, token usage, timing).
    """
    content = await file.read()
    return await _process_uploaded_pdf(file.filename or "unknown.pdf", content)


@app.post("/api/extract/batch")
async def extract_batch_documents(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """Process multiple uploaded PDF documents concurrently."""
    payloads = [(f.filename or "unknown.pdf", await f.read()) for f in files]
    sem = asyncio.Semaphore(max(1, settings.llm_max_concurrency))

    async def _safe_run(name: str, raw: bytes) -> dict[str, Any]:
        async with sem:
            try:
                return await _process_uploaded_pdf(name, raw)
            except HTTPException as exc:
                return {"filename": name, "data": {}, "error": exc.detail}

    t0 = time.perf_counter()
    results = await asyncio.gather(*(_safe_run(name, raw) for name, raw in payloads))
    return {
        "count": len(results),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "results": list(results),
    }


_FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "web" / "dist"
if _FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")
