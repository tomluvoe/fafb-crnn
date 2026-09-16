from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from connectome.graph import (
    aggregate_neuron_pairs,
    build_processed_graph,
    load_processed_graph,
    save_processed_graph,
    sorted_neuron_ids,
)
from connectome.populations import descending_neurons, visual_input_columns
from connectome.report import summarize_connections, summarize_processed_graph


def _classification() -> pd.DataFrame:
    # Unsorted on purpose.
    return pd.DataFrame(
        {
            "root_id": [30, 10, 20, 40],
            "super_class": ["optic", "descending", "optic", "motor"],
        }
    )


def _connections() -> pd.DataFrame:
    # Pair (10, 30) is split across neuropils. Individual syn_count < 5
    # must still be kept and summed.
    return pd.DataFrame(
        {
            "pre_root_id": [10, 10, 30, 10],
            "post_root_id": [30, 30, 40, 20],
            "neuropil": ["A", "B", "A", "A"],
            "syn_count": [2, 4, 7, 1],
        }
    )


def _visual_columns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "root_id": [30, 40],
            "hemisphere": ["left", "right"],
            "type": ["L1", "T4a"],
            "column_id": [1, 2],
            "x": [0, 1],
            "y": [0, 1],
            "p": [0, 1],
            "q": [0, 1],
        }
    )


def test_aggregate_sums_neuropils_and_keeps_small_row_counts() -> None:
    pairs = aggregate_neuron_pairs(_connections())
    assert list(pairs.columns) == ["pre_root_id", "post_root_id", "syn_count"]
    assert pairs.values.tolist() == [
        [10, 20, 1],
        [10, 30, 6],
        [30, 40, 7],
    ]


def test_neuron_ids_are_sorted_and_unique() -> None:
    neuron_ids = sorted_neuron_ids(_classification())
    assert neuron_ids.dtype == np.int64
    np.testing.assert_array_equal(neuron_ids, np.array([10, 20, 30, 40]))


def test_duplicate_root_id_is_rejected() -> None:
    classification = pd.DataFrame({"root_id": [1, 1]})
    with pytest.raises(ValueError, match="not unique"):
        sorted_neuron_ids(classification)


def test_unknown_connection_id_is_rejected() -> None:
    connections = _connections()
    connections.loc[0, "pre_root_id"] = 999
    with pytest.raises(ValueError, match="pre"):
        build_processed_graph(
            classification=_classification(),
            connections=connections,
            visual_columns=_visual_columns(),
        )


def test_build_graph_indices() -> None:
    graph = build_processed_graph(
        classification=_classification(),
        connections=_connections(),
        visual_columns=_visual_columns(),
    )

    # root_id 10,20,30,40 -> 0,1,2,3
    np.testing.assert_array_equal(graph.neuron_ids, np.array([10, 20, 30, 40]))
    assert graph.pre_indices.tolist() == [0, 0, 2]
    assert graph.post_indices.tolist() == [1, 2, 3]
    assert graph.synapse_counts.tolist() == [1, 6, 7]
    assert graph.visual_input_indices.tolist() == [2]
    assert graph.descending_indices.tolist() == [0]
    assert graph.retinotopic_map["root_id"].tolist() == [30]
    assert graph.retinotopic_map["neuron_index"].tolist() == [2]
    assert graph.retinotopic_map["type"].tolist() == ["L1"]

    n = len(graph.neuron_ids)
    assert graph.pre_indices.min() >= 0
    assert graph.post_indices.max() < n


def test_v0_populations_exclude_motor_and_t4() -> None:
    classification = _classification()
    visual_columns = _visual_columns()
    assert descending_neurons(classification)["root_id"].tolist() == [10]
    assert visual_input_columns(visual_columns)["root_id"].tolist() == [30]


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    graph = build_processed_graph(
        classification=_classification(),
        connections=_connections(),
        visual_columns=_visual_columns(),
    )
    save_processed_graph(graph, tmp_path)
    loaded = load_processed_graph(tmp_path)

    np.testing.assert_array_equal(loaded.neuron_ids, graph.neuron_ids)
    assert torch.equal(loaded.pre_indices, graph.pre_indices)
    assert torch.equal(loaded.post_indices, graph.post_indices)
    assert torch.equal(loaded.synapse_counts, graph.synapse_counts)
    assert torch.equal(loaded.visual_input_indices, graph.visual_input_indices)
    assert torch.equal(loaded.descending_indices, graph.descending_indices)
    pd.testing.assert_frame_equal(loaded.retinotopic_map, graph.retinotopic_map)


def _classification_with_isolated() -> pd.DataFrame:
    extra = pd.DataFrame({"root_id": [50], "super_class": ["optic"]})
    return pd.concat([_classification(), extra], ignore_index=True)


def test_connection_summary_matches_pair_aggregation() -> None:
    summary = summarize_connections(
        classification=_classification_with_isolated(),
        connections=_connections(),
        visual_columns=_visual_columns(),
    )
    assert summary.n_rows == 4
    assert summary.n_pairs == 3
    assert summary.n_multi_neuropil_pairs == 1
    assert summary.n_row_syn_lt_5 == 3
    assert summary.n_pairs_syn_lt_5 == 1
    assert summary.pair_syn_min == 1
    assert summary.pair_syn_max == 7
    assert summary.n_classified_no_edges == 1
    assert summary.n_visual_inputs == 1
    assert summary.n_visual_inputs_no_edges == 0
    assert summary.n_descending == 1
    assert summary.n_descending_no_edges == 0
    assert summary.n_endpoints_not_classified == 0
    assert summary.n_self_loop_rows == 0


def test_processed_graph_summary_counts_isolated_units() -> None:
    graph = build_processed_graph(
        classification=_classification_with_isolated(),
        connections=_connections(),
        visual_columns=_visual_columns(),
    )
    summary = summarize_processed_graph(graph)
    assert summary.n_neurons == 5
    assert summary.neuron_ids_sorted
    assert summary.n_edges == 3
    assert summary.n_isolated == 1
    assert summary.n_visual_isolated == 0
    assert summary.n_descending_isolated == 0
    assert summary.retinotopic_types == {"L1": 1}
    assert summary.syn_min == 1
    assert summary.syn_max == 7
