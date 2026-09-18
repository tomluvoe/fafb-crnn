"""Seeded graph rewires for Phase 6 controls."""

from __future__ import annotations

import numpy as np
import torch


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def permute_targets(
    pre: torch.Tensor,
    post: torch.Tensor,
    synapse_counts: torch.Tensor,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Degree-preserving directed rewire: permute post.

    Preserves each source's out-degree, each target's in-degree, and the
    per-edge synapse counts attached to the same source stub. Randomizes
    who connects to whom. Self-loops are repaired by swapping.
    """
    rng = np.random.default_rng(seed)
    pre_np = _as_numpy(pre).astype(np.int64, copy=True)
    post_np = _as_numpy(post).astype(np.int64, copy=True)
    counts = _as_numpy(synapse_counts).copy()
    post_np = rng.permutation(post_np)
    self_loop = pre_np == post_np
    n = len(post_np)
    for i in np.flatnonzero(self_loop):
        for _ in range(16):
            j = int(rng.integers(0, n))
            if pre_np[i] != post_np[j] and pre_np[j] != post_np[i]:
                post_np[i], post_np[j] = post_np[j], post_np[i]
                break
    return (
        torch.from_numpy(pre_np),
        torch.from_numpy(post_np),
        torch.from_numpy(counts),
    )


def random_sparse_like(
    n_neurons: int,
    pre: torch.Tensor,
    synapse_counts: torch.Tensor,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same number of edges and weight multiset; random endpoints."""
    rng = np.random.default_rng(seed)
    n_edges = int(pre.numel())
    new_pre = rng.integers(0, n_neurons, size=n_edges, dtype=np.int64)
    new_post = rng.integers(0, n_neurons, size=n_edges, dtype=np.int64)
    self_loop = new_pre == new_post
    while self_loop.any():
        new_post[self_loop] = rng.integers(0, n_neurons, size=int(self_loop.sum()))
        self_loop = new_pre == new_post
    counts = rng.permutation(_as_numpy(synapse_counts).copy())
    return (
        torch.from_numpy(new_pre),
        torch.from_numpy(new_post),
        torch.from_numpy(counts),
    )


def filter_edges_to_mask(
    pre: torch.Tensor,
    post: torch.Tensor,
    synapse_counts: torch.Tensor,
    keep: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep edges whose endpoints both lie in keep."""
    pre_np = _as_numpy(pre)
    post_np = _as_numpy(post)
    mask = keep[pre_np] & keep[post_np]
    return (
        torch.from_numpy(pre_np[mask].astype(np.int64)),
        torch.from_numpy(post_np[mask].astype(np.int64)),
        torch.from_numpy(_as_numpy(synapse_counts)[mask]),
    )
