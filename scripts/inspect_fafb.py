from data.fafb import (
    load_cell_types,
    load_classification,
    load_connections,
    load_neurons,
    load_visual_columns,
    load_visual_neurons,
)


def print_dataset(name, df):
    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print()

    print("Columns:")
    for column in df.columns:
        print(f"  {column}")

    print()
    print("First rows:")
    print(df.head().to_string(index=False))


def main():
    print("Loading FAFB datasets...")

    classification = load_classification()
    cell_types = load_cell_types()
    neurons = load_neurons()
    visual_neurons = load_visual_neurons()
    visual_columns = load_visual_columns()
    connections = load_connections()

    print_dataset("Classification", classification)
    print_dataset("Cell Types", cell_types)
    print_dataset("Neurons / Neurotransmitters", neurons)
    print_dataset("Visual Neuron Annotations", visual_neurons)
    print_dataset("Visual Columns", visual_columns)
    print_dataset("Connections", connections)

    print()
    print("=" * 80)
    print("FAFB SUMMARY")
    print("=" * 80)

    print(f"Neurons:                  {len(classification):,}")
    print(f"Cell-type annotations:    {len(cell_types):,}")
    print(f"Visual neurons:           {len(visual_neurons):,}")
    print(f"Visual column neurons:    {len(visual_columns):,}")
    print(
        f"Visual columns:           "
        f"{visual_columns['column_id'].nunique():,}"
    )
    print(f"Connection rows:          {len(connections):,}")

    unique_pairs = (
        connections[
            ["pre_root_id", "post_root_id"]
        ]
        .drop_duplicates()
        .shape[0]
    )

    print(f"Unique neuron pairs:      {unique_pairs:,}")

    print()
    print("Visual column cell types:")
    print(
        visual_columns["type"]
        .value_counts()
        .to_string()
    )

    print()
    print("Visual neuron subsystems:")
    print(
        visual_neurons["subsystem"]
        .value_counts(dropna=False)
        .to_string()
    )

    print()
    print("Classification super classes:")
    print(
        classification["super_class"]
        .value_counts(dropna=False)
        .to_string()
    )

    print()
    print("Classification classes:")
    print(
        classification["class"]
        .value_counts(dropna=False)
        .to_string()
    )

    print()
    print("Neurotransmitter types:")
    print(
        neurons["nt_type"]
        .value_counts(dropna=False)
        .to_string()
    )


if __name__ == "__main__":
    main()
