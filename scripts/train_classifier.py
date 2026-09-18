"""Train a linear decoder on frozen FAFB descending activity.

Accuracy describes this model, not biological flies. Compare to Phase 6
controls before interpreting the number.

v0 classes default to butterfly / elephant / spider (coarse silhouettes).
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
from models.decoder import train_linear_decoder
from models.device import resolve_device
from models.features import extract_descending_features
from vision.encoder import ColumnL123Encoder

OUTPUT_DIR = REPO_ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a linear decoder on frozen FAFB CRNN features."
    )
    parser.add_argument("--graph-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument(
        "--classes",
        nargs="+",
        default=list(V0_CLASSES),
        help="English or Italian class names (default: butterfly elephant spider)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--extract-batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto (MPS on Apple silicon), mps, cuda, or cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        "Training a linear decoder on descending-neuron activity. "
        "This is model behavior, not evidence that flies classify animals."
    )
    print(f"Classes: {args.classes}")

    train_set, val_set = Animals10Dataset.splits(
        classes=tuple(args.classes),
        seed=args.seed,
        max_per_class=args.max_per_class,
        image_size=args.image_size,
    )
    n_classes = len(train_set.class_names)
    chance = 1.0 / n_classes
    print(f"Split: train={len(train_set):,} val={len(val_set):,} chance={chance:.3f}")

    device = resolve_device(args.device)
    print(f"Device: {device}")
    print("Loading frozen encoder + CRNN...")
    encoder = ColumnL123Encoder.from_processed_dir(args.graph_dir)
    crnn = SparseFAFBCRNN.from_processed_dir(args.graph_dir)
    if list(crnn.parameters()):
        raise RuntimeError("CRNN must have no trainable parameters")

    train_loader = DataLoader(
        train_set, batch_size=args.extract_batch_size, shuffle=False
    )
    val_loader = DataLoader(val_set, batch_size=args.extract_batch_size, shuffle=False)
    print(
        "Extracting descending features (frozen CRNN). "
        "This is the slow step; the linear fit after it is cheap."
    )
    train_x, train_y = extract_descending_features(
        train_loader, encoder, crnn, device=device, progress=True, desc="train"
    )
    val_x, val_y = extract_descending_features(
        val_loader, encoder, crnn, device=device, progress=True, desc="val"
    )
    print(
        f"Features: train {tuple(train_x.shape)} val {tuple(val_x.shape)} "
        f"n_descending={train_x.shape[1]}"
    )
    print("Z-scoring features on the train split before the linear layer.")

    decoder, history = train_linear_decoder(
        train_x,
        train_y,
        val_x,
        val_y,
        n_classes=n_classes,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        device=device,
    )
    train_acc = history["train_acc"][-1]
    val_acc = history["val_acc"][-1]
    print()
    print("=" * 80)
    print("Linear decoder on frozen FAFB CRNN")
    print("=" * 80)
    print(f"Classes:     {list(train_set.class_names)}")
    print(f"Chance:      {chance:.3f}")
    print(f"Train acc:   {train_acc:.3f}")
    print(f"Val acc:     {val_acc:.3f}")
    print(
        "Do not interpret this as fly object recognition. "
        "Phase 6 controls are required before claiming the wiring matters."
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "decoder.pt"
    metrics_path = args.output_dir / "metrics.json"
    torch.save(
        {
            "state_dict": decoder.state_dict(),
            "class_names": train_set.class_names,
            "n_features": int(train_x.shape[1]),
        },
        checkpoint,
    )
    metrics_path.write_text(
        json.dumps(
            {
                "classes": list(train_set.class_names),
                "chance": chance,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "history": history,
                "note": "Model accuracy, not biological fly behavior.",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {checkpoint} and {metrics_path}")


if __name__ == "__main__":
    main()
