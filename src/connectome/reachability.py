"""Directed reachability on the processed FAFB graph.

Hop distance is the length of a shortest directed path. Unreachable
nodes are marked -1. Recurrent-step count should be at least the
maximum hop from visual inputs to reachable descending neurons.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from connectome.graph import ProcessedGraph

UNREACHABLE = -1
HOPS_FROM_VISUAL_NAME = "hops_from_visual.npy"
HOPS_TO_DESCENDING_NAME = "hops_to_descending.npy"


@dataclass(frozen=True)
class ReachabilityMaps:
    hops_from_visual: np.ndarray
    hops_to_descending: np.ndarray


@dataclass(frozen=True)
class ReachabilitySummary:
    n_neurons: int
    n_visual_inputs: int
    n_descending: int
    n_reachable_from_visual: int
    n_that_can_reach_descending: int
    n_on_visual_to_descending_paths: int
    n_visual_that_reach_descending: int
    n_descending_reachable: int
    n_descending_unreachable: int
    n_visual_also_descending: int
    dn_hop_counts: dict[int, int]
    dn_hop_min: int | None
    dn_hop_median: float | None
    dn_hop_p90: float | None
    dn_hop_p95: float | None
    dn_hop_max: int | None
    suggested_recurrent_steps: int | None


def build_csr(
    sources: np.ndarray, targets: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Directed CSR: offsets[u]:offsets[u+1] indexes neighbors of u."""
    sources = np.asarray(sources, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    if sources.size != targets.size:
        raise ValueError("sources and targets must have the same length")
    if sources.size and (int(sources.min()) < 0 or int(sources.max()) >= n):
        raise ValueError("edge sources out of range")
    if targets.size and (int(targets.min()) < 0 or int(targets.max()) >= n):
        raise ValueError("edge targets out of range")
    if sources.size == 0:
        return np.zeros(n + 1, dtype=np.int64), np.zeros(0, dtype=np.int64)
    counts = np.bincount(sources, minlength=n)
    offsets = np.empty(n + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    order = np.argsort(sources, kind="mergesort")
    return offsets, targets[order]


def shortest_hops(
    offsets: np.ndarray, neighbors: np.ndarray, seeds: np.ndarray, n: int
) -> np.ndarray:
    """Multi-source BFS hop distances. Seeds have distance 0."""
    dist = np.full(n, UNREACHABLE, dtype=np.int32)
    queue: deque[int] = deque()
    for seed in np.unique(np.asarray(seeds, dtype=np.int64)):
        seed_i = int(seed)
        if seed_i < 0 or seed_i >= n:
            raise ValueError(f"seed {seed_i} out of range for N={n}")
        if dist[seed_i] == UNREACHABLE:
            dist[seed_i] = 0
            queue.append(seed_i)
    while queue:
        node = queue.popleft()
        start = int(offsets[node])
        end = int(offsets[node + 1])
        next_dist = dist[node] + 1
        for neighbor in neighbors[start:end]:
            target = int(neighbor)
            if dist[target] == UNREACHABLE:
                dist[target] = next_dist
                queue.append(target)
    return dist


def assert_hops_consistent(
    sources: np.ndarray, targets: np.ndarray, hops: np.ndarray
) -> None:
    """Every out-neighbor of a reachable node is reachable and at most +1 hop."""
    if sources.size == 0:
        return
    from_reachable = hops[sources] >= 0
    if not np.all(hops[targets[from_reachable]] >= 0):
        raise ValueError("BFS missed a neighbor of a reachable node")
    if not np.all(hops[targets[from_reachable]] <= hops[sources[from_reachable]] + 1):
        raise ValueError("hop distances violate directed edges")


def compute_reachability(graph: ProcessedGraph) -> ReachabilityMaps:
    n = int(graph.neuron_ids.size)
    sources = graph.pre_indices.detach().cpu().numpy()
    targets = graph.post_indices.detach().cpu().numpy()
    visual = graph.visual_input_indices.detach().cpu().numpy()
    descending = graph.descending_indices.detach().cpu().numpy()
    if visual.size == 0:
        raise ValueError("graph has no visual input neurons")
    if descending.size == 0:
        raise ValueError("graph has no descending neurons")

    forward_offsets, forward_neighbors = build_csr(sources, targets, n)
    reverse_offsets, reverse_neighbors = build_csr(targets, sources, n)
    hops_from_visual = shortest_hops(forward_offsets, forward_neighbors, visual, n)
    hops_to_descending = shortest_hops(
        reverse_offsets, reverse_neighbors, descending, n
    )
    assert_hops_consistent(sources, targets, hops_from_visual)
    assert_hops_consistent(targets, sources, hops_to_descending)
    return ReachabilityMaps(
        hops_from_visual=hops_from_visual,
        hops_to_descending=hops_to_descending,
    )


def connecting_mask(maps: ReachabilityMaps) -> np.ndarray:
    """Nodes on some directed path from a visual input to a descending neuron."""
    return (maps.hops_from_visual >= 0) & (maps.hops_to_descending >= 0)


def summarize_reachability(
    graph: ProcessedGraph, maps: ReachabilityMaps
) -> ReachabilitySummary:
    visual = graph.visual_input_indices.detach().cpu().numpy()
    descending = graph.descending_indices.detach().cpu().numpy()
    dn_hops = maps.hops_from_visual[descending]
    reachable = dn_hops >= 0
    finite = dn_hops[reachable]
    hop_counts = {
        int(hop): int(count)
        for hop, count in zip(*np.unique(finite, return_counts=True), strict=True)
    }
    if finite.size:
        percentiles = np.percentile(finite.astype(np.float64), [50, 90, 95, 100])
        hop_min = int(finite.min())
        hop_median = float(percentiles[0])
        hop_p90 = float(percentiles[1])
        hop_p95 = float(percentiles[2])
        hop_max = int(finite.max())
        suggested = hop_max
    else:
        hop_min = hop_median = hop_p90 = hop_p95 = hop_max = suggested = None

    visual_set = set(visual.tolist())
    descending_set = set(descending.tolist())
    return ReachabilitySummary(
        n_neurons=int(graph.neuron_ids.size),
        n_visual_inputs=int(visual.size),
        n_descending=int(descending.size),
        n_reachable_from_visual=int((maps.hops_from_visual >= 0).sum()),
        n_that_can_reach_descending=int((maps.hops_to_descending >= 0).sum()),
        n_on_visual_to_descending_paths=int(connecting_mask(maps).sum()),
        n_visual_that_reach_descending=int(
            (maps.hops_to_descending[visual] >= 0).sum()
        ),
        n_descending_reachable=int(reachable.sum()),
        n_descending_unreachable=int((~reachable).sum()),
        n_visual_also_descending=len(visual_set & descending_set),
        dn_hop_counts=hop_counts,
        dn_hop_min=hop_min,
        dn_hop_median=hop_median,
        dn_hop_p90=hop_p90,
        dn_hop_p95=hop_p95,
        dn_hop_max=hop_max,
        suggested_recurrent_steps=suggested,
    )


def neuropils_on_connecting_paths(
    *,
    connections: pd.DataFrame,
    neuron_ids: np.ndarray,
    connecting: np.ndarray,
) -> pd.Series:
    """Neuropil counts for raw rows whose endpoints lie on vis→DN paths."""
    index = pd.Index(neuron_ids)
    pre = index.get_indexer(connections["pre_root_id"])
    post = index.get_indexer(connections["post_root_id"])
    on_path = (pre >= 0) & (post >= 0) & connecting[pre] & connecting[post]
    return connections.loc[on_path, "neuropil"].value_counts()


def save_reachability(maps: ReachabilityMaps, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / HOPS_FROM_VISUAL_NAME, maps.hops_from_visual)
    np.save(output_dir / HOPS_TO_DESCENDING_NAME, maps.hops_to_descending)


def load_reachability(output_dir: Path) -> ReachabilityMaps:
    return ReachabilityMaps(
        hops_from_visual=np.load(output_dir / HOPS_FROM_VISUAL_NAME),
        hops_to_descending=np.load(output_dir / HOPS_TO_DESCENDING_NAME),
    )


def print_reachability_summary(summary: ReachabilitySummary) -> None:
    print(f"Neurons:                            {summary.n_neurons:,}")
    print(f"Visual inputs (L1/L2/L3):           {summary.n_visual_inputs:,}")
    print(f"Descending neurons:                 {summary.n_descending:,}")
    print(f"Visual inputs that are also DN:     {summary.n_visual_also_descending:,}")
    print()
    print(f"Reachable from some visual input:   {summary.n_reachable_from_visual:,}")
    print(
        f"Can reach some descending neuron:   {summary.n_that_can_reach_descending:,}"
    )
    print(
        "On some visual → descending path:   "
        f"{summary.n_on_visual_to_descending_paths:,}"
    )
    print(
        "Visual inputs that can reach a DN:  "
        f"{summary.n_visual_that_reach_descending:,}"
    )
    print(f"Descending reachable from visual:   {summary.n_descending_reachable:,}")
    print(f"Descending unreachable from visual: {summary.n_descending_unreachable:,}")
    print()
    print("Shortest-path hops, visual inputs → descending neurons:")
    if not summary.dn_hop_counts:
        print("  (no descending neuron is reachable)")
        return
    print(f"  {'hops':<8}{'descending neurons':>22}")
    for hop in range(int(summary.dn_hop_min or 0), int(summary.dn_hop_max or 0) + 1):
        print(f"  {hop:<8}{summary.dn_hop_counts.get(hop, 0):>22,}")
    print()
    print(f"  min:    {summary.dn_hop_min}")
    print(f"  median: {summary.dn_hop_median:.1f}")
    print(f"  p90:    {summary.dn_hop_p90:.1f}")
    print(f"  p95:    {summary.dn_hop_p95:.1f}")
    print(f"  max:    {summary.dn_hop_max}")
    print()
    print(
        "Suggested v0 recurrent steps: "
        f"{summary.suggested_recurrent_steps} "
        "(max hops to a reachable descending neuron). "
        "Fewer steps can still carry signal; extra steps will not reach "
        "the unreachable descending neurons."
    )
