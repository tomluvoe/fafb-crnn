"""Evaluate a saved linear decoder on cached descending features.

Does not re-run the CRNN. Requires outputs/features.pt from
scripts/train_classifier.py. The decoder.pt from a run before features
were saved cannot be scored this way.

Default checkpoint is the best-val weights (decoder.pt).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data.fafb import REPO_ROOT
from models.decoder import LinearDecoder, apply_standardize
from models.metrics import (
    evaluate_decoder,
    majority_class_index,
    metrics_to_dict,
    print_split_metrics,
)

OUTPUT_DIR = REPO_ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a saved decoder on cached FAFB features."
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="decoder.pt (best val) or decoder_last.pt",
    )
    return parser.parse_args()


def load_features(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Re-run: uv run python scripts/train_classifier.py "
            "(an older decoder.pt without features.pt cannot be evaluated here)."
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "train_features",
        "train_labels",
        "val_features",
        "val_labels",
        "feature_mean",
        "feature_std",
        "class_names",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"{path} missing {sorted(missing)}")
    return payload


def main() -> None:
    args = parse_args()
    features_path = args.output_dir / "features.pt"
    checkpoint_path = args.checkpoint or (args.output_dir / "decoder.pt")
    payload = load_features(features_path)
    class_names = tuple(payload["class_names"])
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    decoder = LinearDecoder(
        n_features=int(payload["train_features"].shape[1]),
        n_classes=len(class_names),
    )
    decoder.load_state_dict(ckpt["state_dict"])
    decoder.eval()

    train_z = apply_standardize(
        payload["train_features"], payload["feature_mean"], payload["feature_std"]
    )
    val_z = apply_standardize(
        payload["val_features"], payload["feature_mean"], payload["feature_std"]
    )
    majority = majority_class_index(payload["train_labels"])
    train_metrics = evaluate_decoder(
        decoder, train_z, payload["train_labels"], class_names, majority
    )
    val_metrics = evaluate_decoder(
        decoder, val_z, payload["val_labels"], class_names, majority
    )

    print("Evaluating saved decoder on cached features (no CRNN).")
    print(f"Checkpoint: {checkpoint_path}")
    if "best_epoch" in ckpt:
        print(
            f"Best epoch: {ckpt['best_epoch']}  best val acc={ckpt.get('best_val_acc')}"
        )
    print()
    print("=" * 80)
    print("Linear decoder on frozen FAFB CRNN")
    print("=" * 80)
    print(f"Classes: {list(class_names)}")
    print_split_metrics("Train", train_metrics, class_names)
    print()
    print_split_metrics("Val", val_metrics, class_names)
    print()
    print(
        "Do not interpret this as fly object recognition. "
        "Phase 6 controls are required before claiming the wiring matters."
    )

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "classes": list(class_names),
                "checkpoint": str(checkpoint_path),
                "best_epoch": ckpt.get("best_epoch"),
                "best_val_acc": ckpt.get("best_val_acc"),
                "train": metrics_to_dict(train_metrics),
                "val": metrics_to_dict(val_metrics),
                "note": "Model accuracy, not biological fly behavior.",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
