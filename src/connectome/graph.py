"""Build a sparse FAFB graph with a deterministic root_id index.

The mapping is:

    sorted unique classification root_id -> 0 .. N-1

Edges are unique directed neuron pairs. Synapse counts are summed across
neuropils. Do not drop rows with syn_count < 5: the Princeton 5+ filter
already applies to pair totals.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from connectome.populations import descending_neurons, visual_input_columns
from data.fafb import REPO_ROOT

PROCESSED_DIR = REPO_ROOT / "data" / "fafb" / "processed"

NEURON_IDS_NAME = "neuron_ids.npy"
PRE_INDICES_NAME = "pre_indices.pt"
POST_INDICES_NAME = "post_indices.pt"
SYNAPSE_COUNTS_NAME = "synapse_counts.pt"
VISUAL_INPUT_INDICES_NAME = "visual_input_indices.pt"
DESCENDING_INDICES_NAME = "descending_indices.pt"
RETINOTOPIC_MAP_NAME = "retinotopic_map.parquet"

RETINOTOPIC_COLUMNS = (
    "neuron_index",
    "root_id",
    "type",
    "hemisphere",
    "column_id",
    "x",
    "y",
    "p",
    "q",
)


@dataclass(frozen=True)
class ProcessedGraph:
    neuron_ids: np.ndarray
    pre_indices: torch.Tensor
    post_indices: torch.Tensor
    synapse_counts: torch.Tensor
    visual_input_indices: torch.Tensor
    descending_indices: torch.Tensor
    retinotopic_map: pd.DataFrame


def aggregate_neuron_pairs(connections: pd.DataFrame) -> pd.DataFrame:
    """Sum syn_count over neuropils for each directed neuron pair."""
    required = {"pre_root_id", "post_root_id", "syn_count"}
    missing = required.difference(connections.columns)
    if missing:
        raise ValueError(f"connections missing columns: {sorted(missing)}")

    pairs = (
        connections.groupby(["pre_root_id", "post_root_id"], sort=True)["syn_count"]
        .sum()
        .reset_index()
    )
    if (pairs["syn_count"] <= 0).any():
        raise ValueError("aggregated syn_count must be positive")
    return pairs


def sorted_neuron_ids(classification: pd.DataFrame) -> np.ndarray:
    """Deterministic root_id -> index order: sorted unique classification IDs."""
    if "root_id" not in classification.columns:
        raise ValueError("classification missing root_id")
    neuron_ids = np.sort(classification["root_id"].to_numpy())
    if neuron_ids.size == 0:
        raise ValueError("classification has no root_id values")
    if np.any(neuron_ids[1:] == neuron_ids[:-1]):
        raise ValueError("classification root_id values are not unique")
    return neuron_ids.astype(np.int64, copy=False)


def _index_lookup(neuron_ids: np.ndarray) -> pd.Index:
    return pd.Index(neuron_ids)


def _map_ids(
    root_ids: pd.Series | np.ndarray, index: pd.Index, *, label: str
) -> np.ndarray:
    mapped = index.get_indexer(pd.Index(root_ids))
    if np.any(mapped < 0):
        n_missing = int((mapped < 0).sum())
        raise ValueError(f"{n_missing} {label} root_ids are not in the neuron index")
    return mapped.astype(np.int64, copy=False)


def _sorted_unique_indices(
    root_ids: pd.Series, index: pd.Index, *, label: str
) -> torch.Tensor:
    unique_ids = np.sort(pd.unique(root_ids.to_numpy()))
    mapped = _map_ids(unique_ids, index, label=label)
    return torch.tensor(np.sort(mapped), dtype=torch.int64)


def _retinotopic_map(visual_inputs: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    mapped = visual_inputs.copy()
    mapped.insert(
        0,
        "neuron_index",
        _map_ids(mapped["root_id"], index, label="visual input"),
    )
    missing = [column for column in RETINOTOPIC_COLUMNS if column not in mapped.columns]
    if missing:
        raise ValueError(f"visual columns missing {missing}")
    return (
        mapped.loc[:, list(RETINOTOPIC_COLUMNS)]
        .sort_values(["neuron_index", "type"], kind="mergesort")
        .reset_index(drop=True)
    )


def build_processed_graph(
    *,
    classification: pd.DataFrame,
    connections: pd.DataFrame,
    visual_columns: pd.DataFrame,
) -> ProcessedGraph:
    neuron_ids = sorted_neuron_ids(classification)
    index = _index_lookup(neuron_ids)
    pairs = aggregate_neuron_pairs(connections)

    pre_indices = torch.tensor(
        _map_ids(pairs["pre_root_id"], index, label="pre"),
        dtype=torch.int64,
    )
    post_indices = torch.tensor(
        _map_ids(pairs["post_root_id"], index, label="post"),
        dtype=torch.int64,
    )
    synapse_counts = torch.tensor(pairs["syn_count"].to_numpy(), dtype=torch.int64)

    visual_inputs = visual_input_columns(visual_columns)
    descending = descending_neurons(classification)
    visual_input_indices = _sorted_unique_indices(
        visual_inputs["root_id"],
        index,
        label="visual input",
    )
    descending_indices = _sorted_unique_indices(
        descending["root_id"],
        index,
        label="descending",
    )
    retinotopic_map = _retinotopic_map(visual_inputs, index)

    n = neuron_ids.size
    for name, tensor in (
        ("pre_indices", pre_indices),
        ("post_indices", post_indices),
        ("visual_input_indices", visual_input_indices),
        ("descending_indices", descending_indices),
    ):
        if tensor.numel() and (int(tensor.min()) < 0 or int(tensor.max()) >= n):
            raise ValueError(f"{name} out of range for N={n}")
    if not (
        pre_indices.shape == post_indices.shape == synapse_counts.shape == (len(pairs),)
    ):
        raise ValueError("edge tensors must be 1-D and the same length")
    if len(retinotopic_map) != int(visual_input_indices.numel()):
        raise ValueError("retinotopic map size must match visual input indices")

    return ProcessedGraph(
        neuron_ids=neuron_ids,
        pre_indices=pre_indices,
        post_indices=post_indices,
        synapse_counts=synapse_counts,
        visual_input_indices=visual_input_indices,
        descending_indices=descending_indices,
        retinotopic_map=retinotopic_map,
    )


def save_processed_graph(graph: ProcessedGraph, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / NEURON_IDS_NAME, graph.neuron_ids)
    torch.save(graph.pre_indices, output_dir / PRE_INDICES_NAME)
    torch.save(graph.post_indices, output_dir / POST_INDICES_NAME)
    torch.save(graph.synapse_counts, output_dir / SYNAPSE_COUNTS_NAME)
    torch.save(graph.visual_input_indices, output_dir / VISUAL_INPUT_INDICES_NAME)
    torch.save(graph.descending_indices, output_dir / DESCENDING_INDICES_NAME)
    graph.retinotopic_map.to_parquet(output_dir / RETINOTOPIC_MAP_NAME, index=False)


def load_processed_graph(output_dir: Path) -> ProcessedGraph:
    return ProcessedGraph(
        neuron_ids=np.load(output_dir / NEURON_IDS_NAME),
        pre_indices=torch.load(
            output_dir / PRE_INDICES_NAME, map_location="cpu", weights_only=True
        ),
        post_indices=torch.load(
            output_dir / POST_INDICES_NAME, map_location="cpu", weights_only=True
        ),
        synapse_counts=torch.load(
            output_dir / SYNAPSE_COUNTS_NAME, map_location="cpu", weights_only=True
        ),
        visual_input_indices=torch.load(
            output_dir / VISUAL_INPUT_INDICES_NAME,
            map_location="cpu",
            weights_only=True,
        ),
        descending_indices=torch.load(
            output_dir / DESCENDING_INDICES_NAME,
            map_location="cpu",
            weights_only=True,
        ),
        retinotopic_map=pd.read_parquet(output_dir / RETINOTOPIC_MAP_NAME),
    )
