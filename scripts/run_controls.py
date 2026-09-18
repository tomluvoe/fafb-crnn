"""Phase 6 controls: same split and linear-probe protocol as Phase 5.

Default controls (trainable FAFB CRNN is opt-in; it backprops through 3.7M
edges and is slow):

  visual_probe   L1/L2/L3 samples → linear (no CRNN)
  shuffled       degree-preserving FAFB rewire → DN linear
  random_sparse  random edges, same n and weight multiset → DN linear
  optic_lobe     optic + visual-projection subgraph; VPN readout
  frozen_fafb    full FAFB, DN readout (main model; OL + central brain)
  cnn            small grayscale CNN on the same images

Claims are printed per control. None of these is fly behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from connectome.graph import PROCESSED_DIR, load_processed_graph
from connectome.populations import (
    OPTIC_SUPER_CLASS,
    VISUAL_PROJECTION_SUPER_CLASS,
    super_class_mask,
)
from connectome.rewire import filter_edges_to_mask, permute_targets, random_sparse_like
from data.animals10 import V0_CLASSES
from data.dataset import Animals10Dataset
from data.fafb import REPO_ROOT, load_classification
from models.cnn import cnn_logits, train_cnn
from models.crnn import SparseFAFBCRNN
from models.device import resolve_device
from models.features import extract_descending_features, extract_visual_features
from models.metrics import (
    evaluate_logits,
    majority_class_index,
    metrics_to_dict,
    print_split_metrics,
)
from models.probe import fit_linear_probe
from vision.encoder import ColumnL123Encoder

OUTPUT_DIR = REPO_ROOT / "outputs" / "controls"

CLAIMS = {
    "visual_probe": (
        "Linear probe on L1/L2/L3 samples only. If this matches frozen FAFB, "
        "the CRNN is not adding category information beyond the column image."
    ),
    "shuffled": (
        "Permute synapse targets, keeping in/out degree and weight stubs. "
        "If this matches frozen FAFB, who-connects-to-whom is not what matters."
    ),
    "random_sparse": (
        "Random directed edges, same count and weight multiset. "
        "If this matches frozen FAFB, generic sparse recurrence is enough."
    ),
    "optic_lobe": (
        "Edges inside optic + visual-projection neurons; readout is visual "
        "projection cells, not descending neurons. Not a nested accuracy "
        "comparison with the DN probe (different readout)."
    ),
    "frozen_fafb": (
        "Full FAFB (optic lobe + central brain), descending readout. "
        "This is the main model, not a fly."
    ),
    "cnn": (
        "Small grayscale CNN on the same photos. If this fails, the images "
        "are hard at this resolution. If this works and FAFB fails, the "
        "connectome pipeline is the bottleneck. Still not biology."
    ),
}

DEFAULT_CONTROLS = (
    "visual_probe",
    "shuffled",
    "random_sparse",
    "optic_lobe",
    "frozen_fafb",
    "cnn",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 6 classification controls.")
    parser.add_argument("--graph-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--classes", nargs="+", default=list(V0_CLASSES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--extract-batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--cnn-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--controls",
        nargs="+",
        default=list(DEFAULT_CONTROLS),
        help="Subset of: " + ", ".join(DEFAULT_CONTROLS),
    )
    return parser.parse_args()


def _crnn_with_edges(graph, pre, post, counts, readout=None) -> SparseFAFBCRNN:
    return SparseFAFBCRNN(
        n_neurons=int(graph.neuron_ids.size),
        pre_indices=pre,
        post_indices=post,
        synapse_counts=counts,
        visual_input_indices=graph.visual_input_indices,
        descending_indices=graph.descending_indices if readout is None else readout,
    )


def _save_control(output_dir: Path, name: str, payload: dict) -> None:
    folder = output_dir / name
    folder.mkdir(parents=True, exist_ok=True)
    clean = {
        k: v
        for k, v in payload.items()
        if k not in {"train_metrics", "val_metrics", "decoder"}
    }
    (folder / "metrics.json").write_text(json.dumps(clean, indent=2) + "\n")


def _print_probe(name: str, result: dict, class_names: tuple[str, ...]) -> None:
    print()
    print("=" * 80)
    print(name)
    print("=" * 80)
    print(CLAIMS[name])
    print(f"Best epoch: {result['best_epoch']}")
    print_split_metrics("Train", result["train_metrics"], class_names)
    print()
    print_split_metrics("Val", result["val_metrics"], class_names)


def main() -> None:
    args = parse_args()
    unknown = set(args.controls) - set(CLAIMS)
    if unknown:
        raise ValueError(f"unknown controls: {sorted(unknown)}")

    print("Phase 6 controls. Same split as Phase 5. Not fly behavior.")
    train_set, val_set = Animals10Dataset.splits(
        classes=tuple(args.classes),
        seed=args.seed,
        max_per_class=args.max_per_class,
        image_size=args.image_size,
    )
    class_names = train_set.class_names
    device = resolve_device(args.device)
    print(f"Device: {device}  split train={len(train_set):,} val={len(val_set):,}")
    train_loader = DataLoader(
        train_set, batch_size=args.extract_batch_size, shuffle=False
    )
    val_loader = DataLoader(val_set, batch_size=args.extract_batch_size, shuffle=False)
    cnn_train_loader = DataLoader(
        train_set, batch_size=args.extract_batch_size, shuffle=True
    )
    encoder = ColumnL123Encoder.from_processed_dir(args.graph_dir)
    graph = load_processed_graph(args.graph_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}

    probe_kw = dict(epochs=args.epochs, lr=args.lr, seed=args.seed, device=device)

    if "visual_probe" in args.controls:
        print("\n--- visual_probe: extracting L1/L2/L3 ---")
        train_x, train_y = extract_visual_features(
            train_loader, encoder, device, progress=True, desc="visual-train"
        )
        val_x, val_y = extract_visual_features(
            val_loader, encoder, device, progress=True, desc="visual-val"
        )
        result = fit_linear_probe(
            train_x, train_y, val_x, val_y, class_names, **probe_kw
        )
        result["claim"] = CLAIMS["visual_probe"]
        _print_probe("visual_probe", result, class_names)
        _save_control(args.output_dir, "visual_probe", result)
        summary["visual_probe"] = result["val"]

    if "frozen_fafb" in args.controls:
        cached = REPO_ROOT / "outputs" / "features.pt"
        if cached.exists():
            print("\n--- frozen_fafb: loading cached outputs/features.pt ---")
            blob = torch.load(cached, map_location="cpu", weights_only=False)
            if tuple(blob["class_names"]) != class_names:
                raise ValueError("cached features class_names do not match this run")
            train_x, train_y = blob["train_features"], blob["train_labels"]
            val_x, val_y = blob["val_features"], blob["val_labels"]
        else:
            print("\n--- frozen_fafb: extracting DN activity ---")
            crnn = SparseFAFBCRNN.from_processed_graph(graph)
            train_x, train_y = extract_descending_features(
                train_loader, encoder, crnn, device, progress=True, desc="fafb-train"
            )
            val_x, val_y = extract_descending_features(
                val_loader, encoder, crnn, device, progress=True, desc="fafb-val"
            )
        result = fit_linear_probe(
            train_x, train_y, val_x, val_y, class_names, **probe_kw
        )
        result["claim"] = CLAIMS["frozen_fafb"]
        _print_probe("frozen_fafb", result, class_names)
        _save_control(args.output_dir, "frozen_fafb", result)
        summary["frozen_fafb"] = result["val"]

    if "shuffled" in args.controls:
        print("\n--- shuffled: degree-preserving target permutation ---")
        pre, post, counts = permute_targets(
            graph.pre_indices, graph.post_indices, graph.synapse_counts, args.seed
        )
        crnn = _crnn_with_edges(graph, pre, post, counts)
        train_x, train_y = extract_descending_features(
            train_loader, encoder, crnn, device, progress=True, desc="shuffle-train"
        )
        val_x, val_y = extract_descending_features(
            val_loader, encoder, crnn, device, progress=True, desc="shuffle-val"
        )
        result = fit_linear_probe(
            train_x, train_y, val_x, val_y, class_names, **probe_kw
        )
        result["claim"] = CLAIMS["shuffled"]
        _print_probe("shuffled", result, class_names)
        _save_control(args.output_dir, "shuffled", result)
        summary["shuffled"] = result["val"]

    if "random_sparse" in args.controls:
        print("\n--- random_sparse ---")
        pre, post, counts = random_sparse_like(
            int(graph.neuron_ids.size),
            graph.pre_indices,
            graph.synapse_counts,
            args.seed,
        )
        crnn = _crnn_with_edges(graph, pre, post, counts)
        train_x, train_y = extract_descending_features(
            train_loader, encoder, crnn, device, progress=True, desc="random-train"
        )
        val_x, val_y = extract_descending_features(
            val_loader, encoder, crnn, device, progress=True, desc="random-val"
        )
        result = fit_linear_probe(
            train_x, train_y, val_x, val_y, class_names, **probe_kw
        )
        result["claim"] = CLAIMS["random_sparse"]
        _print_probe("random_sparse", result, class_names)
        _save_control(args.output_dir, "random_sparse", result)
        summary["random_sparse"] = result["val"]

    if "optic_lobe" in args.controls:
        print("\n--- optic_lobe: optic + VPN subgraph, VPN readout ---")
        classification = load_classification()
        keep = super_class_mask(
            classification,
            graph.neuron_ids,
            (OPTIC_SUPER_CLASS, VISUAL_PROJECTION_SUPER_CLASS),
        )
        vpn = super_class_mask(
            classification, graph.neuron_ids, (VISUAL_PROJECTION_SUPER_CLASS,)
        )
        pre, post, counts = filter_edges_to_mask(
            graph.pre_indices, graph.post_indices, graph.synapse_counts, keep
        )
        readout = torch.tensor(np.flatnonzero(vpn), dtype=torch.int64)
        print(
            f"OL+VPN neurons={int(keep.sum()):,} edges={int(pre.numel()):,} VPN={int(readout.numel()):,}"
        )
        crnn = _crnn_with_edges(graph, pre, post, counts, readout=readout)
        train_x, train_y = extract_descending_features(
            train_loader, encoder, crnn, device, progress=True, desc="ol-train"
        )
        val_x, val_y = extract_descending_features(
            val_loader, encoder, crnn, device, progress=True, desc="ol-val"
        )
        result = fit_linear_probe(
            train_x, train_y, val_x, val_y, class_names, **probe_kw
        )
        result["claim"] = CLAIMS["optic_lobe"]
        _print_probe("optic_lobe", result, class_names)
        _save_control(args.output_dir, "optic_lobe", result)
        summary["optic_lobe"] = result["val"]

    if "cnn" in args.controls:
        print("\n--- cnn: small grayscale CNN ---")
        print(CLAIMS["cnn"])
        model, history = train_cnn(
            cnn_train_loader,
            val_loader,
            n_classes=len(class_names),
            epochs=args.cnn_epochs,
            device=device,
            seed=args.seed,
        )
        train_logits, train_y = cnn_logits(model, train_loader, device)
        val_logits, val_y = cnn_logits(model, val_loader, device)
        majority = majority_class_index(train_y)
        train_metrics = evaluate_logits(train_logits, train_y, class_names, majority)
        val_metrics = evaluate_logits(val_logits, val_y, class_names, majority)
        best_epoch = int(np.argmax(history["val_acc"])) + 1
        result = {
            "best_epoch": best_epoch,
            "best_val_acc": max(history["val_acc"]),
            "history": history,
            "train": metrics_to_dict(train_metrics),
            "val": metrics_to_dict(val_metrics),
            "claim": CLAIMS["cnn"],
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        print_split_metrics("Train", train_metrics, class_names)
        print()
        print_split_metrics("Val", val_metrics, class_names)
        _save_control(args.output_dir, "cnn", result)
        summary["cnn"] = result["val"]

    print()
    print("=" * 80)
    print("Control summary (val)")
    print("=" * 80)
    print(f"{'control':<16}{'acc':>8}{'bal_acc':>10}{'macroF1':>10}{'majority':>10}")
    for name, val in summary.items():
        print(
            f"{name:<16}{val['accuracy']:8.3f}{val['balanced_accuracy']:10.3f}"
            f"{val['macro_f1']:10.3f}{val['majority_accuracy']:10.3f}"
        )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
