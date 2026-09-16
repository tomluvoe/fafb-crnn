"""Inspect FAFB descending neurons used as the CRNN v0 readout.

FAFB is the brain, not the ventral nerve cord, so the first biological
output population is descending neurons rather than motor neurons.

v0 uses classification.super_class == 'descending'. That is a subset of
flow == 'efferent'. The remaining efferents are brain motor neurons and
endocrine cells, which are not the v0 decoder input.
"""

from connectome.populations import (
    V0_DESCENDING_SUPER_CLASS,
    descending_neurons,
)
from data.fafb import (
    load_cell_types,
    load_classification,
    load_neurons,
    load_visual_neurons,
)


def print_header(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_value_counts(series, *, dropna: bool = False) -> None:
    print(series.value_counts(dropna=dropna).to_string())


def print_candidate_populations(classification) -> None:
    print_header("Candidate output populations")
    print(
        "FAFB contains descending neurons that leave the brain. "
        "It does not contain the full motor periphery."
    )
    print()

    candidates = {
        "super_class == descending": classification["super_class"].eq(
            V0_DESCENDING_SUPER_CLASS
        ),
        "flow == efferent": classification["flow"].eq("efferent"),
        "super_class == motor": classification["super_class"].eq("motor"),
        "class == brain_motor_neuron": classification["class"].eq("brain_motor_neuron"),
    }
    print(f"{'definition':<32}{'neurons':>10}")
    for name, mask in candidates.items():
        print(f"{name:<32}{int(mask.sum()):>10,}")

    descending_ids = set(
        classification.loc[
            classification["super_class"].eq(V0_DESCENDING_SUPER_CLASS),
            "root_id",
        ]
    )
    efferent_ids = set(
        classification.loc[classification["flow"].eq("efferent"), "root_id"]
    )
    motor_ids = set(
        classification.loc[classification["super_class"].eq("motor"), "root_id"]
    )

    print()
    print(f"Descending also labeled efferent:  {len(descending_ids & efferent_ids):,}")
    print(f"Descending also labeled motor:     {len(descending_ids & motor_ids):,}")
    print(f"Efferent but not descending:       {len(efferent_ids - descending_ids):,}")


def print_efferent_remainder(classification) -> None:
    print_header("Efferent cells that are not descending")
    print("These are excluded from the v0 readout.")
    remainder = classification[
        classification["flow"].eq("efferent")
        & classification["super_class"].ne(V0_DESCENDING_SUPER_CLASS)
    ]
    print()
    print(
        remainder.groupby(["super_class", "class", "sub_class"], dropna=False)
        .size()
        .to_string()
    )


def print_descending_taxonomy(descending) -> None:
    print_header("Descending-neuron taxonomy")
    print(f"v0 descending neurons: {len(descending):,}")
    print()
    print("flow:")
    print_value_counts(descending["flow"], dropna=False)
    print()
    print("class:")
    print_value_counts(descending["class"], dropna=False)
    print()
    print("sub_class:")
    print_value_counts(descending["sub_class"], dropna=False)
    print()
    print("side:")
    print_value_counts(descending["side"], dropna=False)
    print()
    print("nerve:")
    print_value_counts(descending["nerve"], dropna=False)
    print()
    print("hemilineage (top 20):")
    print(descending["hemilineage"].value_counts(dropna=False).head(20).to_string())


def print_cell_types_and_nt(descending, cell_types, neurons) -> None:
    print_header("Descending cell types and transmitters")
    descending_ids = set(descending["root_id"])
    typed = cell_types[cell_types["root_id"].isin(descending_ids)]
    print(f"Cell-type annotations: {len(typed):,} / {len(descending):,}")
    print()
    print("primary_type (top 20):")
    print(typed["primary_type"].value_counts(dropna=False).head(20).to_string())

    print()
    print("Predicted neurotransmitter:")
    nt = neurons[neurons["root_id"].isin(descending_ids)]
    print_value_counts(nt["nt_type"], dropna=False)
    print()
    print(
        "Do not treat transmitter labels as excitatory/inhibitory synapse signs in v0."
    )


def print_visual_overlap(descending, visual_neurons) -> None:
    print_header("Overlap with visual annotations")
    descending_ids = set(descending["root_id"])
    visual_ids = set(visual_neurons["root_id"])
    overlap = descending_ids & visual_ids
    print(f"Descending also in visual_neuron_types: {len(overlap):,}")
    if overlap:
        print()
        print(
            visual_neurons.loc[
                visual_neurons["root_id"].isin(overlap),
                "family",
            ]
            .value_counts(dropna=False)
            .to_string()
        )
    print()
    print(
        "A few descending neurons are annotated as visual-family DN cells. "
        "They are still descending outputs, not v0 image-injection targets."
    )


def print_v0_choice(descending) -> None:
    print_header("Candidate CRNN v0 readout")
    print("v0 decoder input: classification.super_class == 'descending'")
    print(f"Neurons: {len(descending):,}")
    print()
    print(descending.groupby(["side"]).size().to_string())
    print()
    print(
        "Motor and endocrine efferents are held out. Connectivity of this "
        "population is checked when the graph is built."
    )


def main() -> None:
    print("Loading FAFB descending-neuron tables...")
    classification = load_classification()
    cell_types = load_cell_types()
    neurons = load_neurons()
    visual_neurons = load_visual_neurons()
    descending = descending_neurons(classification)

    print_candidate_populations(classification)
    print_efferent_remainder(classification)
    print_descending_taxonomy(descending)
    print_cell_types_and_nt(descending, cell_types, neurons)
    print_visual_overlap(descending, visual_neurons)
    print_v0_choice(descending)


if __name__ == "__main__":
    main()
