from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from data.animals10 import resolve_classes
from data.dataset import Animals10Dataset, load_rgb_tensor, stratified_split


def _write_color(path: Path, rgb: tuple[int, int, int]) -> None:
    Image.fromarray(np.full((16, 16, 3), rgb, dtype=np.uint8)).save(path)


def test_resolve_classes_accepts_italian_and_english() -> None:
    assert resolve_classes(["farfalla", "elephant", "ragno"]) == (
        "butterfly",
        "elephant",
        "spider",
    )


def test_resolve_classes_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown"):
        resolve_classes(["banana"])


def test_dataset_split_from_temp_folders(tmp_path: Path) -> None:
    for italian, color in (("farfalla", (255, 0, 0)), ("elefante", (0, 255, 0))):
        folder = tmp_path / italian
        folder.mkdir()
        for i in range(5):
            _write_color(folder / f"{i}.jpeg", color)
    train, val = Animals10Dataset.splits(
        raw_dir=tmp_path,
        classes=("butterfly", "elephant"),
        seed=0,
        image_size=8,
    )
    assert train.class_names == ("butterfly", "elephant")
    assert len(train) + len(val) == 10
    assert len(val) >= 2
    image, label = train[0]
    assert image.shape == (3, 8, 8)
    assert 0.0 <= float(image.min()) <= float(image.max()) <= 1.0
    assert label in {0, 1}


def test_load_rgb_tensor_resizes(tmp_path: Path) -> None:
    path = tmp_path / "x.png"
    Image.fromarray(np.zeros((40, 80, 3), dtype=np.uint8)).save(path)
    tensor = load_rgb_tensor(path, image_size=10)
    assert tensor.shape == (3, 10, 10)


def test_stratified_split_keeps_both_classes() -> None:
    by_class = {
        "a": [Path(f"a{i}") for i in range(4)],
        "b": [Path(f"b{i}") for i in range(4)],
    }
    train, val, names = stratified_split(by_class, seed=1, train_frac=0.75)
    assert names == ("a", "b")
    train_labels = {label for _, label in train}
    val_labels = {label for _, label in val}
    assert train_labels == {0, 1}
    assert val_labels == {0, 1}
