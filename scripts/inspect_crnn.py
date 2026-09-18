"""Run the frozen sparse CRNN on a synthetic image.

Loads processed graph artifacts (not the connectivity CSV). Encodes a
left-to-right ramp onto L1/L2/L3, runs 5 rate steps, and prints
descending-neuron stats.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from connectome.graph import PROCESSED_DIR
from models.crnn import V0_LEAK, V0_STEPS, SparseFAFBCRNN
from models.device import resolve_device
from vision.encoder import ColumnL123Encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect frozen FAFB CRNN on a synthetic image."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Directory written by scripts/build_graph.py",
    )
    parser.add_argument("--leak", type=float, default=V0_LEAK)
    parser.add_argument("--steps", type=int, default=V0_STEPS)
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Use raw log1p(syn_count) without incoming-sum normalization",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto (MPS on Apple silicon), mps, cuda, or cpu",
    )
    return parser.parse_args()


def _horizontal_ramp(height: int = 64, width: int = 64) -> torch.Tensor:
    ramp = torch.linspace(0.0, 1.0, width).view(1, 1, 1, width)
    return ramp.expand(1, 1, height, width).contiguous()


def main() -> None:
    args = parse_args()
    print(f"Loading processed graph from {args.input_dir}...")
    device = resolve_device(args.device)
    print(f"Device: {device}")
    encoder = ColumnL123Encoder.from_processed_dir(args.input_dir).to(device)
    net = SparseFAFBCRNN.from_processed_dir(
        args.input_dir,
        leak=args.leak,
        steps=args.steps,
        normalize=not args.no_normalize,
    ).to(device)
    visual = encoder.encode(_horizontal_ramp().to(device))
    print(
        f"N={net.n_neurons:,} edges={int(net.pre.numel()):,} "
        f"steps={net.steps} leak={net.leak} normalize={net.normalize}"
    )
    print(
        "Activation is ReLU. leak and incoming-weight normalization are "
        "engineering choices, not membrane biophysics."
    )

    start = time.perf_counter()
    hidden = net.forward(visual)
    elapsed = time.perf_counter() - start
    descending = net.descending_state(hidden)

    print()
    print("=" * 80)
    print("After recurrence")
    print("=" * 80)
    print(f"Runtime:                 {elapsed:.2f} s")
    print(
        f"Hidden mean / max:       {float(hidden.mean()):.4f} / {float(hidden.max()):.4f}"
    )
    print(f"Hidden units > 0:        {int((hidden > 0).sum()):,} / {net.n_neurons:,}")
    print(
        f"Descending mean / max:   {float(descending.mean()):.4f} / "
        f"{float(descending.max()):.4f}"
    )
    print(
        f"Descending units > 0:    {int((descending > 0).sum()):,} / "
        f"{int(net.descending_indices.numel()):,}"
    )


if __name__ == "__main__":
    main()
