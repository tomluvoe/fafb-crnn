"""Inspect the v0 column encoder on a synthetic image.

Does not need Animals-10. Loads retinotopic_map.parquet from the
processed graph directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from connectome.graph import PROCESSED_DIR
from vision.encoder import ColumnL123Encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect L1/L2/L3 encoding of a synthetic image."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Directory with retinotopic_map.parquet",
    )
    parser.add_argument(
        "--delta",
        choices=("spatial", "gray"),
        default="spatial",
        help="Still-image ON/OFF definition (default: spatial)",
    )
    return parser.parse_args()


def _horizontal_ramp(height: int = 64, width: int = 64) -> torch.Tensor:
    ramp = torch.linspace(0.0, 1.0, width).view(1, 1, 1, width)
    return ramp.expand(1, 1, height, width).contiguous()


def main() -> None:
    args = parse_args()
    map_path = args.input_dir / "retinotopic_map.parquet"
    if not map_path.exists():
        raise FileNotFoundError(
            f"{map_path} does not exist. Run: uv run python scripts/build_graph.py"
        )

    encoder = ColumnL123Encoder.from_processed_dir(args.input_dir, delta=args.delta)
    table = encoder.map
    print(f"Loaded {map_path}")
    print()
    print("=" * 80)
    print("Retinotopic visual inputs")
    print("=" * 80)
    print(f"Neurons:           {len(table):,}")
    print(f"Unique (x, y):     {table[['x', 'y']].drop_duplicates().shape[0]:,}")
    print(f"x range:           {int(table['x'].min())} .. {int(table['x'].max())}")
    print(f"y range:           {int(table['y'].min())} .. {int(table['y'].max())}")
    print(f"ON/OFF delta:      {args.delta}")
    print()
    print(table["type"].value_counts().sort_index().to_string())
    print()
    print(table["hemisphere"].value_counts().sort_index().to_string())

    values = encoder.encode(_horizontal_ramp())
    print()
    print("=" * 80)
    print("Synthetic left-dark → right-bright ramp")
    print("=" * 80)
    for cell_type in ("L1", "L2", "L3"):
        mask = torch.as_tensor(table["type"].eq(cell_type).to_numpy(copy=True))
        channel = values[0, mask]
        print(
            f"{cell_type}: mean={float(channel.mean()):.4f} "
            f"min={float(channel.min()):.4f} max={float(channel.max()):.4f}"
        )
    print()
    print(
        "v0 default ON/OFF is spatial contrast on the column lattice, "
        "not temporal phototransduction. Both hemispheres get the same field."
    )


if __name__ == "__main__":
    main()
