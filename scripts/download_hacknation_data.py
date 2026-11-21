"""Download the HackNation 2026 dataset into the local repository."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


REPOSITORY = "https://github.com/Urzad-Zamowien-Publicznych/hacknation-2026.git"


def run_git(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
    )


def download_data(repo_root: Path, branch: str, target: Path) -> tuple[int, int]:
    with tempfile.TemporaryDirectory(prefix="hacknation-data-") as temp_dir:
        clone_dir = Path(temp_dir) / "source"
        run_git(
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            "--branch",
            branch,
            REPOSITORY,
            str(clone_dir),
        )
        run_git("-C", str(clone_dir), "sparse-checkout", "set", "dane", "dane/skany")

        source_dir = clone_dir / "dane"
        if not source_dir.is_dir():
            raise RuntimeError(f"Source repository does not contain {source_dir.name}/")

        target.mkdir(parents=True, exist_ok=True)
        if any(target.iterdir()):
            shutil.rmtree(target)
            target.mkdir(parents=True)
        shutil.copytree(source_dir, target, dirs_exist_ok=True)

    pdf_count = sum(1 for path in target.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf")
    json_count = sum(1 for path in target.rglob("*") if path.is_file() and path.suffix.lower() == ".json")
    return pdf_count, json_count


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Pobierz dane HackNation 2026 do katalogu dane/."
    )
    parser.add_argument("--branch", default="main", help="Gałąź repozytorium źródłowego")
    parser.add_argument(
        "--target",
        type=Path,
        default=repo_root / "dane",
        help="Katalog docelowy (domyślnie: dane/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = args.target.resolve()
    pdf_count, json_count = download_data(Path(__file__).resolve().parents[1], args.branch, target)
    print(f"Pobrano dane do: {target}")
    print(f"Pliki PDF: {pdf_count}")
    print(f"Pliki JSON: {json_count}")


if __name__ == "__main__":
    main()
