"""Local Animals-10 paths and Italian folder names.

Images are not in git. The Kaggle dump uses Italian directory names under
data/animals10/raw-img/. The bundled translate.py is incomplete; use
ITALIAN_TO_ENGLISH here.
"""

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
