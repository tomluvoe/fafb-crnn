"""Animals-10 image dataset for the linear decoder.

CI must not require the Kaggle dump. Tests can point at a temporary folder.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.animals10 import V0_CLASSES, list_class_images


def load_rgb_tensor(path: Path, image_size: int) -> torch.Tensor:
    """Load an RGB image, resize to image_size², return (3, H, W) in [0, 1]."""
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize(
            (image_size, image_size), Image.Resampling.BILINEAR
        )
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def stratified_split(
    by_class: dict[str, list[Path]],
    *,
    seed: int,
    train_frac: float = 0.8,
    max_per_class: int | None = None,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], tuple[str, ...]]:
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1)")
    rng = random.Random(seed)
    class_names = tuple(by_class.keys())
    train: list[tuple[Path, int]] = []
    val: list[tuple[Path, int]] = []
    for label, english in enumerate(class_names):
        paths = list(by_class[english])
        rng.shuffle(paths)
        if max_per_class is not None:
            paths = paths[:max_per_class]
        if len(paths) < 2:
            raise ValueError(f"class {english!r} needs at least 2 images for a split")
        n_train = min(len(paths) - 1, max(1, int(round(len(paths) * train_frac))))
        train.extend((path, label) for path in paths[:n_train])
        val.extend((path, label) for path in paths[n_train:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val, class_names


class Animals10Dataset(Dataset):
    def __init__(
        self,
        items: list[tuple[Path, int]],
        class_names: tuple[str, ...],
        *,
        image_size: int = 128,
    ) -> None:
        self.items = items
        self.class_names = class_names
        self.image_size = image_size

    @classmethod
    def splits(
        cls,
        *,
        raw_dir: Path | None = None,
        classes: tuple[str, ...] = V0_CLASSES,
        seed: int = 0,
        train_frac: float = 0.8,
        max_per_class: int | None = None,
        image_size: int = 128,
    ) -> tuple[Animals10Dataset, Animals10Dataset]:
        by_class = list_class_images(raw_dir, classes)
        train_items, val_items, names = stratified_split(
            by_class, seed=seed, train_frac=train_frac, max_per_class=max_per_class
        )
        return (
            cls(train_items, names, image_size=image_size),
            cls(val_items, names, image_size=image_size),
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.items[index]
        return load_rgb_tensor(path, self.image_size), label
