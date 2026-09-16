"""Build reusable FAFB graph artifacts for the CRNN.

This is a one-time (or rare) preprocess. Training should load
data/fafb/processed/, not parse connections_princeton.csv.gz again.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from connectome.graph import (
    PROCESSED_DIR,
    build_processed_graph,
    save_processed_graph,
)
from connectome.report import print_processed_graph_summary, summarize_processed_graph
from data.fafb import load_classification, load_connections, load_visual_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build processed FAFB graph artifacts."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Directory for neuron_ids.npy, edge tensors, and index lists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading FAFB tables...")
    classification = load_classification()
    visual_columns = load_visual_columns()
    connections = load_connections()

    print("Building sparse graph...")
    graph = build_processed_graph(
        classification=classification,
        connections=connections,
        visual_columns=visual_columns,
    )
    save_processed_graph(graph, args.output_dir)

    print()
    print(f"Wrote {args.output_dir}")
    print(f"Connection rows read: {len(connections):,}")
    print()
    print_processed_graph_summary(summarize_processed_graph(graph))


if __name__ == "__main__":
    main()
