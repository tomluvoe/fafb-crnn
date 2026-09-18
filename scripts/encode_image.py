"""Encode an image file onto FAFB L1/L2/L3 units.

Writes a (n_visual,) float32 array aligned with retinotopic_map rows.
Does not parse FlyWire connectivity.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from connectome.graph import PROCESSED_DIR
from vision.encoder import ColumnL123Encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample an image onto processed FAFB visual inputs."
    )
    parser.add_argument("image", type=Path, help="RGB or grayscale image file")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Directory with retinotopic_map.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .npy path (default: alongside the image)",
    )
    parser.add_argument(
        "--delta",
        choices=("spatial", "gray"),
        default="spatial",
        help="Still-image ON/OFF definition (default: spatial)",
    )
    return parser.parse_args()


def load_image(path: Path) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def main() -> None:
    args = parse_args()
    encoder = ColumnL123Encoder.from_processed_dir(args.input_dir, delta=args.delta)
    values = encoder.encode(load_image(args.image))
    output = args.output
    if output is None:
        output = args.image.with_suffix(".visual.npy")
    np.save(output, values[0].detach().cpu().numpy())
    table = encoder.map
    print(f"Wrote {output} shape={(len(table),)}")
    print(f"delta={args.delta}")
    for cell_type in ("L1", "L2", "L3"):
        mask = torch.as_tensor(table["type"].eq(cell_type).to_numpy(copy=True))
        channel = values[0, mask]
        print(f"{cell_type}: n={int(mask.sum()):,} mean={float(channel.mean()):.4f}")


if __name__ == "__main__":
    main()
