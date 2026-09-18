"""Local Animals-10 paths and Italian folder names.

Images are not in git. The Kaggle dump uses Italian directory names under
data/animals10/raw-img/. The bundled translate.py is incomplete; use
ITALIAN_TO_ENGLISH here.
"""

from pathlib import Path

from data.fafb import REPO_ROOT

ANIMALS10_DIR = REPO_ROOT / "data" / "animals10"
ANIMALS10_RAW = ANIMALS10_DIR / "raw-img"

ITALIAN_TO_ENGLISH = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "ragno": "spider",
    "scoiattolo": "squirrel",
}

ENGLISH_TO_ITALIAN = {
    english: italian for italian, english in ITALIAN_TO_ENGLISH.items()
}

# Coarse silhouettes for v0; cat vs dog is a later hard test.
V0_CLASSES = ("butterfly", "elephant", "spider")

IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}


def resolve_classes(names: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Return English class names, accepting English or Italian labels."""
    resolved: list[str] = []
    for name in names:
        key = name.strip().lower()
        if key in ITALIAN_TO_ENGLISH:
            resolved.append(ITALIAN_TO_ENGLISH[key])
        elif key in ENGLISH_TO_ITALIAN:
            resolved.append(key)
        else:
            raise ValueError(f"unknown Animals-10 class: {name!r}")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"duplicate classes: {resolved}")
    return tuple(resolved)


def list_class_images(
    raw_dir: Path | None = None,
    classes: tuple[str, ...] = V0_CLASSES,
) -> dict[str, list[Path]]:
    """Map English class name -> image paths. Missing folders error."""
    root = ANIMALS10_RAW if raw_dir is None else Path(raw_dir)
    if not root.is_dir():
        raise FileNotFoundError(
            f"{root} not found. Unzip Animals-10 into data/animals10/raw-img/."
        )
    english_classes = resolve_classes(classes)
    by_class: dict[str, list[Path]] = {}
    for english in english_classes:
        folder = root / ENGLISH_TO_ITALIAN[english]
        if not folder.is_dir():
            raise FileNotFoundError(f"missing class folder: {folder}")
        paths = sorted(
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not paths:
            raise FileNotFoundError(f"no images in {folder}")
        by_class[english] = paths
    return by_class
