"""Frozen FAFB, same split and DN probe, more recurrent steps.

5 steps is when the last reachable DNs can first see the image.
10 and 20 ask whether extra mixing changes the linear readout.
They do not reach DNs that have no path from vision.

This is not a wiring control (use run_controls.py for that).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from connectome.graph import PROCESSED_DIR
from data.animals10 import V0_CLASSES
from data.dataset import Animals10Dataset
from data.fafb import REPO_ROOT
from models.crnn import SparseFAFBCRNN
from models.device import resolve_device
from models.features import extract_descending_features
from models.metrics import print_split_metrics
from models.probe import fit_linear_probe
from vision.encoder import ColumnL123Encoder

OUTPUT_DIR = REPO_ROOT / "outputs" / "steps"
DEFAULT_STEPS = (5, 10, 20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep frozen CRNN step count on the Phase 5 probe."
    )
    parser.add_argument("--graph-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--classes", nargs="+", default=list(V0_CLASSES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--extract-batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=list(DEFAULT_STEPS),
        help="Recurrent step counts (default: 5 10 20)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        "Sweeping recurrent steps on frozen FAFB. "
        "Same DN linear probe as Phase 5. Not fly behavior."
    )
    train_set, val_set = Animals10Dataset.splits(
        classes=tuple(args.classes),
        seed=args.seed,
        max_per_class=args.max_per_class,
        image_size=args.image_size,
    )
    class_names = train_set.class_names
    device = resolve_device(args.device)
    print(f"Device: {device}  train={len(train_set):,} val={len(val_set):,}")
    print(f"Steps: {args.steps}")

    encoder = ColumnL123Encoder.from_processed_dir(args.graph_dir)
    train_loader = DataLoader(
        train_set, batch_size=args.extract_batch_size, shuffle=False
    )
    val_loader = DataLoader(val_set, batch_size=args.extract_batch_size, shuffle=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    cached = REPO_ROOT / "outputs" / "features.pt"

    for n_steps in args.steps:
        print()
        print(f"--- steps={n_steps} ---")
        if n_steps == 5 and cached.exists() and args.max_per_class is None:
            blob = torch.load(cached, map_location="cpu", weights_only=False)
            if tuple(blob["class_names"]) != class_names:
                raise ValueError("cached features class_names do not match")
            print("Reusing outputs/features.pt for steps=5")
            train_x, train_y = blob["train_features"], blob["train_labels"]
            val_x, val_y = blob["val_features"], blob["val_labels"]
        else:
            crnn = SparseFAFBCRNN.from_processed_dir(args.graph_dir, steps=n_steps)
            train_x, train_y = extract_descending_features(
                train_loader,
                encoder,
                crnn,
                device,
                progress=True,
                desc=f"steps{n_steps}-train",
            )
            val_x, val_y = extract_descending_features(
                val_loader,
                encoder,
                crnn,
                device,
                progress=True,
                desc=f"steps{n_steps}-val",
            )
        result = fit_linear_probe(
            train_x,
            train_y,
            val_x,
            val_y,
            class_names,
            epochs=args.epochs,
            lr=args.lr,
            seed=args.seed,
            device=device,
        )
        print(f"Best epoch: {result['best_epoch']}")
        print_split_metrics("Train", result["train_metrics"], class_names)
        print()
        print_split_metrics("Val", result["val_metrics"], class_names)
        folder = args.output_dir / f"steps_{n_steps}"
        folder.mkdir(parents=True, exist_ok=True)
        payload = {
            "steps": n_steps,
            "best_epoch": result["best_epoch"],
            "best_val_acc": result["best_val_acc"],
            "history": result["history"],
            "train": result["train"],
            "val": result["val"],
            "claim": (
                "Extra steps test longer frozen mixing, not FAFB pairing. "
                "They do not reach vision-unreachable DNs."
            ),
        }
        (folder / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
        summary[str(n_steps)] = result["val"]

    print()
    print("=" * 80)
    print("Step sweep (val)")
    print("=" * 80)
    print(f"{'steps':<8}{'acc':>8}{'bal_acc':>10}{'macroF1':>10}")
    for n_steps, val in summary.items():
        print(
            f"{n_steps:<8}{val['accuracy']:8.3f}"
            f"{val['balanced_accuracy']:10.3f}{val['macro_f1']:10.3f}"
        )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
