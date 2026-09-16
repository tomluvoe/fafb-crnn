"""Summaries for raw connectivity and processed graph artifacts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from connectome.graph import ProcessedGraph, aggregate_neuron_pairs
from connectome.populations import descending_neurons, visual_input_columns


@dataclass(frozen=True)
class ConnectionSummary:
    n_rows: int
    n_pairs: int
    n_self_loop_rows: int
    n_self_loop_pairs: int
    n_multi_neuropil_pairs: int
    n_row_syn_lt_5: int
    row_syn_min: int
    row_syn_max: int
    pair_syn_min: int
    pair_syn_max: int
    n_pairs_syn_lt_5: int
    n_endpoints: int
    n_endpoints_not_classified: int
    n_classified: int
    n_classified_no_edges: int
    n_visual_inputs: int
    n_visual_inputs_no_edges: int
    n_descending: int
    n_descending_no_edges: int


@dataclass(frozen=True)
class ProcessedGraphSummary:
    n_neurons: int
    neuron_ids_sorted: bool
    n_edges: int
    n_self_loops: int
    syn_min: int
    syn_max: int
    index_min: int
    index_max: int
    n_visual_inputs: int
    n_descending: int
    n_isolated: int
    n_visual_isolated: int
    n_descending_isolated: int
    retinotopic_types: dict[str, int]


def summarize_connections(
    *,
    classification: pd.DataFrame,
    connections: pd.DataFrame,
    visual_columns: pd.DataFrame,
) -> ConnectionSummary:
    pairs = aggregate_neuron_pairs(connections)
    pair_rows = connections.groupby(["pre_root_id", "post_root_id"], sort=False).size()
    classified = set(classification["root_id"].tolist())
    endpoints = set(connections["pre_root_id"].tolist()) | set(
        connections["post_root_id"].tolist()
    )
    visual_ids = set(visual_input_columns(visual_columns)["root_id"].tolist())
    descending_ids = set(descending_neurons(classification)["root_id"].tolist())
    pair_self = pairs["pre_root_id"].eq(pairs["post_root_id"])
    return ConnectionSummary(
        n_rows=len(connections),
        n_pairs=len(pairs),
        n_self_loop_rows=int(
            connections["pre_root_id"].eq(connections["post_root_id"]).sum()
        ),
        n_self_loop_pairs=int(pair_self.sum()),
        n_multi_neuropil_pairs=int((pair_rows > 1).sum()),
        n_row_syn_lt_5=int((connections["syn_count"] < 5).sum()),
        row_syn_min=int(connections["syn_count"].min()),
        row_syn_max=int(connections["syn_count"].max()),
        pair_syn_min=int(pairs["syn_count"].min()),
        pair_syn_max=int(pairs["syn_count"].max()),
        n_pairs_syn_lt_5=int((pairs["syn_count"] < 5).sum()),
        n_endpoints=len(endpoints),
        n_endpoints_not_classified=len(endpoints - classified),
        n_classified=len(classified),
        n_classified_no_edges=len(classified - endpoints),
        n_visual_inputs=len(visual_ids),
        n_visual_inputs_no_edges=len(visual_ids - endpoints),
        n_descending=len(descending_ids),
        n_descending_no_edges=len(descending_ids - endpoints),
    )


def summarize_processed_graph(graph: ProcessedGraph) -> ProcessedGraphSummary:
    n = int(graph.neuron_ids.size)
    neuron_ids_sorted = bool(
        n <= 1 or np.all(graph.neuron_ids[1:] > graph.neuron_ids[:-1])
    )
    if int(graph.pre_indices.numel()) == 0:
        syn_min = 0
        syn_max = 0
        index_min = 0
        index_max = -1
        n_self_loops = 0
        has_edge = np.zeros(n, dtype=bool)
    else:
        syn_min = int(graph.synapse_counts.min())
        syn_max = int(graph.synapse_counts.max())
        index_min = int(
            torch.minimum(graph.pre_indices.min(), graph.post_indices.min())
        )
        index_max = int(
            torch.maximum(graph.pre_indices.max(), graph.post_indices.max())
        )
        n_self_loops = int((graph.pre_indices == graph.post_indices).sum())
        has_edge = np.zeros(n, dtype=bool)
        has_edge[graph.pre_indices.cpu().numpy()] = True
        has_edge[graph.post_indices.cpu().numpy()] = True

    visual = graph.visual_input_indices.cpu().numpy()
    descending = graph.descending_indices.cpu().numpy()
    types = (
        graph.retinotopic_map["type"].value_counts().sort_index().astype(int).to_dict()
    )
    return ProcessedGraphSummary(
        n_neurons=n,
        neuron_ids_sorted=neuron_ids_sorted,
        n_edges=int(graph.pre_indices.numel()),
        n_self_loops=n_self_loops,
        syn_min=syn_min,
        syn_max=syn_max,
        index_min=index_min,
        index_max=index_max,
        n_visual_inputs=int(graph.visual_input_indices.numel()),
        n_descending=int(graph.descending_indices.numel()),
        n_isolated=int((~has_edge).sum()) if n else 0,
        n_visual_isolated=int((~has_edge[visual]).sum()) if visual.size else 0,
        n_descending_isolated=int((~has_edge[descending]).sum())
        if descending.size
        else 0,
        retinotopic_types={str(k): int(v) for k, v in types.items()},
    )


def print_connection_summary(summary: ConnectionSummary) -> None:
    print(f"Connection rows:                 {summary.n_rows:,}")
    print(f"Unique directed pairs:           {summary.n_pairs:,}")
    print(f"Pairs in more than one neuropil: {summary.n_multi_neuropil_pairs:,}")
    print(f"Self-loop rows:                  {summary.n_self_loop_rows:,}")
    print(f"Self-loop pairs:                 {summary.n_self_loop_pairs:,}")
    print(
        f"Row syn_count:                   "
        f"{summary.row_syn_min:,} .. {summary.row_syn_max:,}"
    )
    print(f"Rows with syn_count < 5:         {summary.n_row_syn_lt_5:,}")
    print(
        f"Pair syn_count:                  "
        f"{summary.pair_syn_min:,} .. {summary.pair_syn_max:,}"
    )
    print(f"Pairs with syn_count < 5:        {summary.n_pairs_syn_lt_5:,}")
    print()
    print("Do not drop rows with syn_count < 5. The 5+ filter is on pair totals.")
    print()
    print(f"Classified neurons:              {summary.n_classified:,}")
    print(f"Connection endpoints:            {summary.n_endpoints:,}")
    print(f"Endpoints missing classification: {summary.n_endpoints_not_classified:,}")
    print(f"Classified with no edges:        {summary.n_classified_no_edges:,}")
    print(f"Visual inputs (L1/L2/L3 mapped): {summary.n_visual_inputs:,}")
    print(f"Visual inputs with no edges:     {summary.n_visual_inputs_no_edges:,}")
    print(f"Descending neurons:              {summary.n_descending:,}")
    print(f"Descending with no edges:        {summary.n_descending_no_edges:,}")


def print_processed_graph_summary(summary: ProcessedGraphSummary) -> None:
    print(f"Neurons:                         {summary.n_neurons:,}")
    print(f"root_id order strictly sorted:   {summary.neuron_ids_sorted}")
    print(f"Directed edges:                  {summary.n_edges:,}")
    print(f"Self-loops:                      {summary.n_self_loops:,}")
    print(
        f"Synapse count:                   {summary.syn_min:,} .. {summary.syn_max:,}"
    )
    print(
        f"Edge index range:                {summary.index_min} .. {summary.index_max}"
    )
    print(f"Visual input neurons:            {summary.n_visual_inputs:,}")
    print(f"Descending neurons:              {summary.n_descending:,}")
    print(f"Neurons with no edges:           {summary.n_isolated:,}")
    print(f"Visual inputs with no edges:     {summary.n_visual_isolated:,}")
    print(f"Descending with no edges:        {summary.n_descending_isolated:,}")
    print()
    print("Retinotopic visual-input types:")
    for cell_type, count in summary.retinotopic_types.items():
        print(f"  {cell_type}: {count:,}")
    print()
    print(
        "Synapse counts are pair totals. log1p and recurrence are model "
        "choices, not part of this graph."
    )
