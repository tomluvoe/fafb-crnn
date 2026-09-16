"""Inspect processed FAFB graph artifacts.

Run after scripts/build_graph.py. Does not parse the connectivity CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from connectome.graph import PROCESSED_DIR, load_processed_graph
from connectome.report import print_processed_graph_summary, summarize_processed_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect processed FAFB graph artifacts."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Directory written by scripts/build_graph.py",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_dir.exists():
        raise FileNotFoundError(
            f"{args.input_dir} does not exist. "
            "Run: uv run python scripts/build_graph.py"
        )

    graph = load_processed_graph(args.input_dir)
    print(f"Loaded {args.input_dir}")
    print()
    print("=" * 80)
    print("Processed graph")
    print("=" * 80)
    print_processed_graph_summary(summarize_processed_graph(graph))


if __name__ == "__main__":
    main()
