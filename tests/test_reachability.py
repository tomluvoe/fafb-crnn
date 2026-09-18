from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from connectome.graph import ProcessedGraph
from connectome.reachability import (
    assert_hops_consistent,
    build_csr,
    compute_reachability,
    connecting_mask,
    load_reachability,
    neuropils_on_connecting_paths,
    save_reachability,
    shortest_hops,
    summarize_reachability,
)


def _graph(
    *,
    n: int,
    edges: list[tuple[int, int]],
    visual: list[int],
    descending: list[int],
) -> ProcessedGraph:
    pre, post = zip(*edges, strict=True) if edges else ([], [])
    return ProcessedGraph(
        neuron_ids=np.arange(n, dtype=np.int64),
        pre_indices=torch.tensor(list(pre), dtype=torch.int64),
        post_indices=torch.tensor(list(post), dtype=torch.int64),
        synapse_counts=torch.ones(len(edges), dtype=torch.int64),
        visual_input_indices=torch.tensor(visual, dtype=torch.int64),
        descending_indices=torch.tensor(descending, dtype=torch.int64),
        retinotopic_map=pd.DataFrame(),
    )


def test_shortest_hops_on_a_line() -> None:
    offsets, neighbors = build_csr(np.array([0, 1]), np.array([1, 2]), n=4)
    hops = shortest_hops(offsets, neighbors, np.array([0]), n=4)
    np.testing.assert_array_equal(hops, np.array([0, 1, 2, -1], dtype=np.int32))


def test_shortest_hops_uses_minimum_of_two_paths() -> None:
    # 0 -> 1 -> 3 and 0 -> 2 -> 1 -> 3; shortest 0->3 is 2 via 1.
    offsets, neighbors = build_csr(np.array([0, 0, 1, 2]), np.array([1, 2, 3, 1]), n=4)
    hops = shortest_hops(offsets, neighbors, np.array([0]), n=4)
    np.testing.assert_array_equal(hops, np.array([0, 1, 1, 2], dtype=np.int32))


def test_unreachable_descending_neuron() -> None:
    graph = _graph(
        n=4,
        edges=[(0, 1), (1, 2)],
        visual=[0],
        descending=[2, 3],
    )
    maps = compute_reachability(graph)
    np.testing.assert_array_equal(
        maps.hops_from_visual, np.array([0, 1, 2, -1], dtype=np.int32)
    )
    summary = summarize_reachability(graph, maps)
    assert summary.n_descending_reachable == 1
    assert summary.n_descending_unreachable == 1
    assert summary.dn_hop_max == 2
    assert summary.suggested_recurrent_steps == 2
    assert connecting_mask(maps).tolist() == [True, True, True, False]


def test_reverse_hops_to_descending() -> None:
    graph = _graph(
        n=3,
        edges=[(0, 1), (1, 2)],
        visual=[0],
        descending=[2],
    )
    maps = compute_reachability(graph)
    np.testing.assert_array_equal(
        maps.hops_to_descending, np.array([2, 1, 0], dtype=np.int32)
    )


def test_inconsistent_hops_are_rejected() -> None:
    with pytest.raises(ValueError, match="missed"):
        assert_hops_consistent(
            np.array([0]),
            np.array([1]),
            np.array([0, -1], dtype=np.int32),
        )


def test_save_and_load_hop_maps(tmp_path: Path) -> None:
    graph = _graph(n=2, edges=[(0, 1)], visual=[0], descending=[1])
    maps = compute_reachability(graph)
    save_reachability(maps, tmp_path)
    loaded = load_reachability(tmp_path)
    np.testing.assert_array_equal(loaded.hops_from_visual, maps.hops_from_visual)
    np.testing.assert_array_equal(loaded.hops_to_descending, maps.hops_to_descending)


def test_neuropils_on_connecting_paths() -> None:
    graph = _graph(
        n=3,
        edges=[(0, 1), (1, 2)],
        visual=[0],
        descending=[2],
    )
    maps = compute_reachability(graph)
    connections = pd.DataFrame(
        {
            "pre_root_id": [0, 1, 0],
            "post_root_id": [1, 2, 99],
            "neuropil": ["ME_R", "LO_R", "UNRELATED"],
        }
    )
    counts = neuropils_on_connecting_paths(
        connections=connections,
        neuron_ids=graph.neuron_ids,
        connecting=connecting_mask(maps),
    )
    assert counts.to_dict() == {"ME_R": 1, "LO_R": 1}
