"""Inspect FAFB visual annotations used for CRNN input.

Focus:

- photoreceptors R1-R8
- lamina monopolar cells L1-L5
- retinotopic column assignments
- which neurons can actually receive sampled image values in v0

v0 injects an approximate ON / OFF / intensity encoding into L1 / L2 / L3
at visual-column coordinates. That is an engineering mapping, not a model
of fly phototransduction. R1-R6 exist in the annotations but have no
column map in this dataset, so they are not v0 injection targets.
"""

from connectome.populations import V0_VISUAL_INPUT_TYPES
from data.fafb import load_classification, load_visual_columns, load_visual_neurons

PHOTORECEPTOR_TYPES = ("R1-6", "R7", "R8")
LAMINA_MONOPOLAR_TYPES = ("L1", "L2", "L3", "L4", "L5")


def print_header(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_value_counts(series, *, dropna: bool = False) -> None:
    print(series.value_counts(dropna=dropna).to_string())


def print_overlap(visual_neurons, visual_columns, classification) -> None:
    visual_ids = set(visual_neurons["root_id"])
    column_ids = set(visual_columns["root_id"])
    classified_ids = set(classification["root_id"])

    print_header("Table overlap")
    print(f"Visual-neuron annotations:      {len(visual_ids):,}")
    print(f"Column-assigned neurons:        {len(column_ids):,}")
    print(f"In columns, not visual types:   {len(column_ids - visual_ids):,}")
    print(f"In visual types, not columns:   {len(visual_ids - column_ids):,}")
    print(f"Columns missing classification: {len(column_ids - classified_ids):,}")
    print(f"Visual types missing classif.:  {len(visual_ids - classified_ids):,}")


def print_photoreceptors(visual_neurons, visual_columns, classification) -> None:
    print_header("Photoreceptors R1-R8")
    print(
        "FlyWire groups outer photoreceptors as type 'R1-6'. "
        "R7 and R8 are separate types."
    )
    print()

    visual_by_type = visual_neurons["type"].value_counts()
    column_by_type = visual_columns["type"].value_counts()

    print(f"{'type':<8}{'visual_neurons':>18}{'column_map':>14}")
    for cell_type in PHOTORECEPTOR_TYPES:
        n_visual = int(visual_by_type.get(cell_type, 0))
        n_column = int(column_by_type.get(cell_type, 0))
        print(f"{cell_type:<8}{n_visual:>18,}{n_column:>14,}")

    print()
    print("Photoreceptor family / subsystem / side:")
    photo = visual_neurons[visual_neurons["type"].isin(PHOTORECEPTOR_TYPES)]
    print(photo.groupby(["type", "family", "subsystem", "side"]).size().to_string())

    print()
    print("Classification of annotated photoreceptors:")
    photo_class = classification.merge(
        photo[["root_id", "type"]],
        on="root_id",
        how="inner",
    )
    print(
        photo_class.groupby(["type", "flow", "super_class", "class", "sub_class"])
        .size()
        .to_string()
    )

    print()
    print(
        "R1-6 have no retinotopic column assignment in this table, "
        "so v0 cannot sample an image at R1-6 coordinates. "
        "R7 and R8 are in the column map but are not v0 injection targets."
    )


def print_lamina(visual_neurons, visual_columns, classification) -> None:
    print_header("Lamina monopolar cells")

    visual_by_type = visual_neurons["type"].value_counts()
    column_by_type = visual_columns["type"].value_counts()

    print(f"{'type':<8}{'visual_neurons':>18}{'column_map':>14}{'unmapped':>12}")
    for cell_type in LAMINA_MONOPOLAR_TYPES:
        n_visual = int(visual_by_type.get(cell_type, 0))
        n_column = int(column_by_type.get(cell_type, 0))
        print(f"{cell_type:<8}{n_visual:>18,}{n_column:>14,}{n_visual - n_column:>12,}")

    print()
    print("L1 / L2 / L3 annotations (v0 injection types):")
    lamina = visual_neurons[visual_neurons["type"].isin(V0_VISUAL_INPUT_TYPES)]
    print(
        lamina.groupby(["type", "family", "subsystem", "category", "side"])
        .size()
        .to_string()
    )

    print()
    print("Classification of column-assigned L1 / L2 / L3:")
    mapped = visual_columns[visual_columns["type"].isin(V0_VISUAL_INPUT_TYPES)]
    mapped_class = classification.merge(
        mapped[["root_id", "type"]],
        on="root_id",
        how="inner",
    )
    print(
        mapped_class.groupby(["type", "flow", "super_class", "class", "sub_class"])
        .size()
        .to_string()
    )


def print_column_map(visual_columns) -> None:
    print_header("Retinotopic column map")
    print(f"Assigned neurons:     {len(visual_columns):,}")
    print(f"Unique neurons:       {visual_columns['root_id'].nunique():,}")
    print(f"Columnar cell types:  {visual_columns['type'].nunique():,}")
    print(f"Visual columns:       {visual_columns['column_id'].nunique():,}")
    print()
    print("Hemisphere neuron counts:")
    print_value_counts(visual_columns["hemisphere"])
    print()
    print("Unique column_id by hemisphere:")
    print(visual_columns.groupby("hemisphere")["column_id"].nunique().to_string())
    print()
    print("Coordinate ranges:")
    for column in ("p", "q", "x", "y"):
        print(
            f"  {column}: {visual_columns[column].min()}"
            f" .. {visual_columns[column].max()}"
        )
    print()
    print("All 31 columnar types (most are downstream, not inputs):")
    print_value_counts(visual_columns["type"])


def print_column_completeness(visual_columns) -> None:
    print_header("Column completeness for v0 inputs")
    print(
        "A visual column is usable for a cell type when that type is "
        "assigned to the column in that hemisphere."
    )

    for hemisphere in ("left", "right"):
        hemi = visual_columns[visual_columns["hemisphere"] == hemisphere]
        column_ids = set(hemi["column_id"])
        print()
        print(f"{hemisphere} hemisphere: {len(column_ids):,} columns")
        print(f"  {'type':<8}{'present':>10}{'missing':>10}")
        for cell_type in (*V0_VISUAL_INPUT_TYPES, "R7", "R8"):
            present = set(hemi.loc[hemi["type"] == cell_type, "column_id"])
            print(
                f"  {cell_type:<8}{len(present):>10,}{len(column_ids - present):>10,}"
            )


def print_v0_inputs(visual_columns) -> None:
    print_header("Candidate CRNN v0 visual inputs")
    print("v0 samples the image at column coordinates and injects:")
    print("  L1 <- ON / luminance-change")
    print("  L2 <- OFF / luminance-change")
    print("  L3 <- slower intensity / contrast")
    print()
    print("This mapping is an engineering approximation.")
    print()

    inputs = visual_columns[visual_columns["type"].isin(V0_VISUAL_INPUT_TYPES)]
    print(f"Column-assigned L1/L2/L3 neurons: {len(inputs):,}")
    print()
    print(inputs.groupby(["type", "hemisphere"]).size().to_string())
    print()
    print(
        "Neurons of these types without a column assignment are not "
        "usable as spatially localized visual inputs."
    )


def main() -> None:
    print("Loading FAFB visual-system tables...")
    classification = load_classification()
    visual_neurons = load_visual_neurons()
    visual_columns = load_visual_columns()

    print_header("Visual neuron annotations")
    print(f"Rows:          {len(visual_neurons):,}")
    print(f"Unique types:  {visual_neurons['type'].nunique():,}")
    print()
    print("Families:")
    print_value_counts(visual_neurons["family"], dropna=False)
    print()
    print("Subsystems:")
    print_value_counts(visual_neurons["subsystem"], dropna=False)
    print()
    print("Categories:")
    print_value_counts(visual_neurons["category"])
    print()
    print("Side:")
    print_value_counts(visual_neurons["side"], dropna=True)

    print_overlap(visual_neurons, visual_columns, classification)
    print_photoreceptors(visual_neurons, visual_columns, classification)
    print_lamina(visual_neurons, visual_columns, classification)
    print_column_map(visual_columns)
    print_column_completeness(visual_columns)
    print_v0_inputs(visual_columns)


if __name__ == "__main__":
    main()
