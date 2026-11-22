"""Fetch real official KIO ruling PDFs from orzeczenia.uzp.gov.pl as negative (`inny`) test data."""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.services.classifier import classify_document  # noqa: E402
from backend.services.parser import parse_pdf  # noqa: E402

UZP_RESULTS_URL = "https://orzeczenia.uzp.gov.pl/Home/GetResults"
UZP_PDF_URL = "https://orzeczenia.uzp.gov.pl/Home/PdfContent/{doc_id}?Kind=KIO"
_DETAILS_ID_RE = re.compile(r"/Home/Details/(\d+)")
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def _request_with_retry(
    client: httpx.Client,
    method: str,
    url: str,
    retries: int = 4,
    **kwargs,
) -> httpx.Response:
    for attempt in range(retries):
        try:
            return client.request(method, url, **kwargs)
        except httpx.HTTPError:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("Unreachable")


def collect_kio_ids(client: httpx.Client, count: int) -> list[int]:
    """Collect up to `count` recent KIO ruling IDs from orzeczenia.uzp.gov.pl."""
    ids: list[int] = []
    seen: set[int] = set()
    page = 1
    while len(ids) < count:
        resp = _request_with_retry(
            client,
            "POST",
            UZP_RESULTS_URL,
            data={"Kind": "KIO", "Page": str(page)},
        )
        resp.raise_for_status()
        found = [int(m) for m in _DETAILS_ID_RE.findall(resp.text)]
        if not found:
            break
        added = 0
        for doc_id in found:
            if doc_id not in seen:
                seen.add(doc_id)
                ids.append(doc_id)
                added += 1
                if len(ids) >= count:
                    break
        if added == 0:
            break
        page += 1
    return ids


def fetch_negatives(output_dir: Path, count: int, verify: bool = False) -> int:
    """Download `count` official KIO PDFs into `output_dir` (never `dane/`)."""
    resolved = output_dir.resolve()
    dane_dir = (REPO_ROOT / "dane").resolve()
    if resolved == dane_dir or dane_dir in resolved.parents:
        print("Błąd: nigdy nie zapisuj pobranych plików w katalogu dane/.", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    with httpx.Client(timeout=30.0, follow_redirects=True, headers=_HEADERS) as client:
        doc_ids = collect_kio_ids(client, count)
        print(f"Znaleziono {len(doc_ids)} ID orzeczeń KIO na orzeczenia.uzp.gov.pl.")
        for doc_id in doc_ids:
            target = output_dir / f"KIO_{doc_id}.pdf"
            if target.exists() and target.stat().st_size > 0:
                downloaded.append(target)
                continue
            resp = _request_with_retry(client, "GET", UZP_PDF_URL.format(doc_id=doc_id))
            if resp.status_code == 200 and resp.content.startswith(b"%PDF"):
                target.write_bytes(resp.content)
                downloaded.append(target)
                print(f"  Pobrano {target.name} ({len(resp.content)} B)")
            else:
                print(f"  Pominięto {doc_id}: HTTP {resp.status_code}")

    print(f"Gotowe: {len(downloaded)} plików PDF w {output_dir}")

    if verify and downloaded:
        rejected = 0
        for pdf_path in downloaded:
            parsed = parse_pdf(pdf_path.read_bytes())
            doc_type, conf = classify_document(parsed.text)
            label = getattr(doc_type, "value", str(doc_type))
            if label == "inny":
                rejected += 1
            print(f"  [{label:>14} conf={conf:.3f}] {pdf_path.name} ({parsed.page_count} str.)")
        print(f"Weryfikacja klasyfikacji `inny`: {rejected}/{len(downloaded)}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pobiera prawdziwe orzeczenia KIO (PDF) z orzeczenia.uzp.gov.pl jako negatywy `inny`."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "tmp" / "uzp_negatives",
        help="Katalog docelowy poza dane/ (domyślnie: tmp/uzp_negatives)",
    )
    parser.add_argument("--count", type=int, default=10, help="Liczba PDF-ów do pobrania")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Od razu sprawdź pobrane PDF-y klasyfikatorem",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(fetch_negatives(args.output_dir, max(1, args.count), verify=args.verify))
