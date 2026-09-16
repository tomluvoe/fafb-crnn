"""Inspect raw FAFB connectivity used to build the CRNN graph.

Uses the same pair aggregation as scripts/build_graph.py. This script
does not write files.
"""

from connectome.report import print_connection_summary, summarize_connections
from data.fafb import load_classification, load_connections, load_visual_columns


def main() -> None:
    print("Loading FAFB connectivity tables...")
    classification = load_classification()
    visual_columns = load_visual_columns()
    connections = load_connections()

    print()
    print("=" * 80)
    print("Raw connectivity")
    print("=" * 80)
    print_connection_summary(
        summarize_connections(
            classification=classification,
            connections=connections,
            visual_columns=visual_columns,
        )
    )


if __name__ == "__main__":
    main()
