
"""
Week 1 - step 1: fetch the TrashNet dataset (Yang & Thung, 2016).

Downloads ``dataset-resized.zip`` (~41 MB) with HTTP range resume, verifies the
archive, extracts it, then reports the per-class image counts.

"""

from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import config
from utils import ensure_dirs, header, human_bytes, step

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
CHUNK = 1 << 20  # 1 MiB
MAX_ATTEMPTS = 40


# --------------------------------------------------------------------------- checks
def find_class_root(root: Path) -> Path | None:
    """Return the directory that directly contains the 6 class sub-folders."""
    wanted = set(config.CLASS_NAMES)
    for candidate in [root, *sorted(p for p in root.rglob("*") if p.is_dir())]:
        children = {p.name for p in candidate.iterdir() if p.is_dir()}
        if wanted.issubset(children):
            return candidate
    return None


def count_images(dataset_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in config.CLASS_NAMES:
        folder = dataset_dir / name
        if not folder.is_dir():
            counts[name] = 0
            continue
        counts[name] = sum(
            1 for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
    return counts


def dataset_is_ready() -> bool:
    counts = count_images(config.DATASET_DIR)
    return all(n > 0 for n in counts.values())


# --------------------------------------------------------------------------- download
def download_with_resume(url: str, dest: Path) -> Path:
    """GET ``url`` into ``dest``, resuming from the current file size."""
    ensure_dirs(dest.parent)
    attempt = 0
    while attempt < MAX_ATTEMPTS:
        attempt += 1
        existing = dest.stat().st_size if dest.exists() else 0

        headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
        if existing:
            headers["Range"] = f"bytes={existing}-"

        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=90) as response:
                resumed = existing > 0 and response.status == 206
                if existing and not resumed:
                    print("    server ignored Range -> restarting from byte 0")
                    existing = 0

                content_length = response.headers.get("Content-Length")
                total = existing + int(content_length) if content_length else None
                mode = "ab" if resumed else "wb"
                written = existing
                last_report = time.time()

                with open(dest, mode) as fh:
                    while True:
                        block = response.read(CHUNK)
                        if not block:
                            break
                        fh.write(block)
                        written += len(block)
                        if time.time() - last_report >= 3.0:
                            last_report = time.time()
                            if total:
                                pct = written / total * 100
                                print(
                                    f"    {human_bytes(written)} / {human_bytes(total)}"
                                    f"  ({pct:5.1f}%)",
                                    flush=True,
                                )
                            else:
                                print(f"    {human_bytes(written)}", flush=True)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as exc:
            print(f"    interrupted ({type(exc).__name__}: {exc}) - resuming...")
            time.sleep(min(2 * attempt, 10))
            continue

        size = dest.stat().st_size
        if total is not None and size < total:
            print(f"    short read ({size}/{total} bytes) - resuming...")
            time.sleep(2)
            continue
        if _zip_is_valid(dest):
            return dest
        print("    archive incomplete/corrupt - resuming...")
        time.sleep(2)

    raise RuntimeError(
        f"Could not download the dataset after {MAX_ATTEMPTS} attempts.\n"
        "Manual options:\n"
        "  1) Download dataset-resized.zip from https://github.com/garythung/trashnet\n"
        f"     and place it at: {config.ZIP_PATH}\n"
        "  2) Or download the 'Garbage Classification' / TrashNet dataset from Kaggle."
    )


def _zip_is_valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1_000_000:
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            return zf.testzip() is None
    except zipfile.BadZipFile:
        return False


# --------------------------------------------------------------------------- extract
def extract_zip(zip_path: Path, dataset_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(zip_path.parent)

    root = find_class_root(zip_path.parent)
    if root is None:
        raise RuntimeError(
            f"Extracted archive at {zip_path.parent} but could not find folders "
            f"for {config.CLASS_NAMES}."
        )

    ensure_dirs(dataset_dir)
    if root.resolve() != dataset_dir.resolve():
        # The archive may nest the class folders one level deeper than expected
        # (e.g. data/dataset-resized/dataset-resized/<class>). Move the class
        # folders themselves rather than renaming the parent, which would fail
        # on Windows when the target directory already exists.
        for name in config.CLASS_NAMES:
            src = root / name
            dst = dataset_dir / name
            if dst.exists():
                continue
            src.rename(dst)
    print(f"    extracted to {dataset_dir}")


# --------------------------------------------------------------------------- report
def report(counts: dict[str, int]) -> None:
    total = sum(counts.values())
    header("TrashNet dataset summary")
    print(f"{'class':<12}{'images':>8}{'share':>10}")
    print("-" * 30)
    for name in config.CLASS_NAMES:
        n = counts[name]
        share = f"{n / total * 100:.1f}%" if total else "-"
        print(f"{name:<12}{n:>8}{share:>10}")
    print("-" * 30)
    print(f"{'TOTAL':<12}{total:>8}")
    if total:
        worst = min(counts, key=counts.get)
        best = max(counts, key=counts.get)
        print(
            f"\nImbalance ratio: {counts[best] / max(counts[worst], 1):.1f}x "
            f"({best}={counts[best]} vs {worst}={counts[worst]})"
        )
        print("This is why Week 2 experiments with class weights and macro-F1.")


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download and prepare TrashNet.")
    parser.add_argument("--force", action="store_true", help="re-download from scratch")
    parser.add_argument("--verify", action="store_true", help="only verify the local copy")
    args = parser.parse_args(argv)

    header("STEP 1/3 - dataset", "-")

    if args.verify:
        counts = count_images(config.DATASET_DIR)
        report(counts)
        return 0 if all(counts.values()) else 1

    if not args.force and dataset_is_ready():
        step(f"Dataset already present at {config.DATASET_DIR} - skipping download.")
        report(count_images(config.DATASET_DIR))
        return 0

    if args.force:
        for path in (config.ZIP_PATH,):
            if path.exists():
                path.unlink()

    if not _zip_is_valid(config.ZIP_PATH):
        step(f"Downloading TrashNet (~{human_bytes(config.DATASET_EXPECTED_BYTES)})...")
        last_error: Exception | None = None
        for url in config.DATASET_URLS:
            try:
                print(f"  source: {url}")
                download_with_resume(url, config.ZIP_PATH)
                last_error = None
                break
            except Exception as exc:  # try the next mirror
                last_error = exc
                print(f"  source failed: {exc}")
        if last_error is not None:
            raise last_error
    else:
        step("Archive already downloaded and valid - skipping download.")

    step("Extracting archive...")
    if find_class_root(config.DATA_DIR) is None or not dataset_is_ready():
        extract_zip(config.ZIP_PATH, config.DATASET_DIR)

    counts = count_images(config.DATASET_DIR)
    report(counts)
    if not all(counts.values()):
        print("\n[ERROR] Some classes have no images. Re-run with --force.")
        return 1
    print("\n[OK] Dataset ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
