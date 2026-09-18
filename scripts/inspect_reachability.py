"""Directed reachability from v0 visual inputs to descending neurons.

Computes shortest-path hops on the processed graph, writes hop maps next
to the graph artifacts, and optionally tallies neuropils on vis→DN paths
from the raw connectivity table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from connectome.graph import PROCESSED_DIR, load_processed_graph
from connectome.reachability import (
    compute_reachability,
    connecting_mask,
    neuropils_on_connecting_paths,
    print_reachability_summary,
    save_reachability,
    summarize_reachability,
)
from data.fafb import load_connections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect directed reachability from visual inputs to DNs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Directory written by scripts/build_graph.py",
    )
    parser.add_argument(
        "--skip-neuropils",
        action="store_true",
        help="Do not load connections_princeton.csv.gz for neuropil counts",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write hops_from_visual.npy / hops_to_descending.npy",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_dir.exists():
        raise FileNotFoundError(
            f"{args.input_dir} does not exist. "
            "Run: uv run python scripts/build_graph.py"
        )

    print(f"Loading processed graph from {args.input_dir}...")
    graph = load_processed_graph(args.input_dir)
    print("Computing directed shortest-path hops...")
    maps = compute_reachability(graph)
    summary = summarize_reachability(graph, maps)

    if not args.no_save:
        save_reachability(maps, args.input_dir)
        print(f"Wrote hop maps to {args.input_dir}")

    print()
    print("=" * 80)
    print("Visual → descending reachability")
    print("=" * 80)
    print_reachability_summary(summary)

    if args.skip_neuropils:
        return

    print()
    print("Loading raw connections for neuropil tally...")
    connections = load_connections()
    connecting = connecting_mask(maps)
    neuropils = neuropils_on_connecting_paths(
        connections=connections,
        neuron_ids=graph.neuron_ids,
        connecting=connecting,
    )
    print()
    print("=" * 80)
    print("Neuropils on visual → descending paths")
    print("=" * 80)
    print(f"Connecting neurons: {int(connecting.sum()):,}")
    print(f"Raw connection rows on those neurons: {int(neuropils.sum()):,}")
    print(f"Distinct neuropils: {len(neuropils):,}")
    print()
    print("Top neuropils by row count:")
    print(neuropils.head(25).to_string())


if __name__ == "__main__":
    main()
