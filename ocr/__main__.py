"""CLI entry-point: ``python -m ocr <plik.pdf>``

Outputs a JSON object to stdout with ``ensure_ascii=False`` and UTF-8 encoding
(also on Windows, where the default console encoding may be cp1250).
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] in {"-h", "--help"}:
        print("Użycie: python -m ocr <ścieżka_do_pliku.pdf>", file=sys.stderr)
        raise SystemExit(1)

    pdf_path = Path(sys.argv[1])

    # Force UTF-8 on stdout (Windows often defaults to cp1250 or UTF-16).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    elif sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )

    from ocr.pdf import extract_pages

    try:
        result = extract_pages(pdf_path)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"BŁĄD: {exc}", file=sys.stderr)
        raise SystemExit(2)

    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
