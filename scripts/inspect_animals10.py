"""Inspect the local Animals-10 dump (not committed)."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from PIL import Image

from data.animals10 import ANIMALS10_RAW, ITALIAN_TO_ENGLISH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect local Animals-10 images.")
    parser.add_argument("--input-dir", type=Path, default=ANIMALS10_RAW)
    return parser.parse_args()


def main() -> None:
    root = parse_args().input_dir
    if not root.is_dir():
        raise FileNotFoundError(
            f"{root} not found. Unzip archive.zip into data/animals10/ "
            "(raw-img/<italian class>/...)."
        )

    print(f"Loaded {root}")
    print()
    print("=" * 80)
    print("Animals-10 class counts")
    print("=" * 80)
    print(f"{'folder':<14}{'english':<12}{'n':>8}  size range")
    exts: Counter[str] = Counter()
    total = 0
    unreadable = 0
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        english = ITALIAN_TO_ENGLISH.get(folder.name, "?")
        files = [p for p in folder.iterdir() if p.is_file()]
        widths: list[int] = []
        heights: list[int] = []
        for path in files:
            exts[path.suffix.lower()] += 1
            try:
                with Image.open(path) as image:
                    width, height = image.size
                widths.append(width)
                heights.append(height)
            except OSError:
                unreadable += 1
        total += len(files)
        size = (
            f"{min(widths)}x{min(heights)} .. {max(widths)}x{max(heights)}"
            if widths
            else "n/a"
        )
        print(f"{folder.name:<14}{english:<12}{len(files):8,d}  {size}")

    print()
    print(f"Total images:     {total:,}")
    print(f"Unreadable:       {unreadable:,}")
    print(f"Extensions:       {dict(exts)}")
    print()
    print(
        "Folders are Italian. Bundled translate.py mixes both directions and "
        "omits ragno→spider; use data.animals10.ITALIAN_TO_ENGLISH."
    )
    print(
        "Photos are web-scraped (OIP- names, watermarks, crops). "
        "Do not commit this directory."
    )


if __name__ == "__main__":
    main()
