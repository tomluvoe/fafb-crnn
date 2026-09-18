"""Frozen connectome-constrained rate RNN on the processed FAFB graph.

v0 dynamics (engineering, not biophysics):

    h(t+1) = (1 - leak) * h(t) + leak * relu(W h(t) + external_input(t))

W is sparse COO: edge k carries log1p(synapse_counts[k]) from pre[k] to
post[k]. Incoming weights are normalized to sum to 1 per target so five
ReLU steps stay in a usable range. Topology and weights are buffers,
not parameters.

Default steps=5 is the Phase 2 max hop from L1/L2/L3 to a reachable
descending neuron. Extra steps will not reach the unreachable DNs.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from connectome.graph import ProcessedGraph, load_processed_graph

V0_STEPS = 5
V0_LEAK = 0.5


def log1p_weights(synapse_counts: torch.Tensor) -> torch.Tensor:
    """Synapse-count proxy. Not a biological synaptic weight."""
    if int((synapse_counts < 0).any()):
        raise ValueError("synapse_counts must be non-negative")
    return torch.log1p(synapse_counts.to(dtype=torch.float32))


def normalize_incoming(
    post: torch.Tensor, weight: torch.Tensor, n_neurons: int
) -> torch.Tensor:
    """Scale edges so each target's incoming weights sum to 1.

    Engineering for rate stability, not a biological claim.
    """
    incoming = torch.zeros(n_neurons, dtype=weight.dtype, device=weight.device)
    incoming.scatter_add_(0, post, weight)
    scale = incoming.clamp(min=1e-8)
    return weight / scale[post]


class SparseFAFBCRNN(nn.Module):
    """Sparse rate network with frozen FAFB topology."""

    def __init__(
        self,
        *,
        n_neurons: int,
        pre_indices: torch.Tensor,
        post_indices: torch.Tensor,
        synapse_counts: torch.Tensor,
        visual_input_indices: torch.Tensor,
        descending_indices: torch.Tensor,
        leak: float = V0_LEAK,
        steps: int = V0_STEPS,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        if n_neurons <= 0:
            raise ValueError("n_neurons must be positive")
        if not 0.0 <= leak <= 1.0:
            raise ValueError("leak must be in [0, 1]")
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if not (pre_indices.shape == post_indices.shape == synapse_counts.shape):
            raise ValueError("edge tensors must have the same shape")
        if pre_indices.numel() and (
            int(pre_indices.min()) < 0
            or int(post_indices.min()) < 0
            or int(pre_indices.max()) >= n_neurons
            or int(post_indices.max()) >= n_neurons
        ):
            raise ValueError("edge indices out of range")

        self.n_neurons = n_neurons
        self.leak = float(leak)
        self.steps = int(steps)
        self.normalize = bool(normalize)

        weight = log1p_weights(synapse_counts.reshape(-1))
        post = post_indices.reshape(-1).to(dtype=torch.int64)
        if normalize and weight.numel():
            weight = normalize_incoming(post, weight, n_neurons)

        self.register_buffer("pre", pre_indices.reshape(-1).to(dtype=torch.int64))
        self.register_buffer("post", post)
        self.register_buffer("weight", weight)
        self.register_buffer(
            "visual_input_indices",
            visual_input_indices.reshape(-1).to(dtype=torch.int64),
        )
        self.register_buffer(
            "descending_indices",
            descending_indices.reshape(-1).to(dtype=torch.int64),
        )

    @classmethod
    def from_processed_graph(cls, graph: ProcessedGraph, **kwargs) -> SparseFAFBCRNN:
        return cls(
            n_neurons=int(graph.neuron_ids.size),
            pre_indices=graph.pre_indices,
            post_indices=graph.post_indices,
            synapse_counts=graph.synapse_counts,
            visual_input_indices=graph.visual_input_indices,
            descending_indices=graph.descending_indices,
            **kwargs,
        )

    @classmethod
    def from_processed_dir(cls, processed_dir: Path, **kwargs) -> SparseFAFBCRNN:
        return cls.from_processed_graph(load_processed_graph(processed_dir), **kwargs)

    def sparse_mm(self, h: torch.Tensor) -> torch.Tensor:
        """Compute W h without a dense N×N matrix. h is (B, N)."""
        if h.ndim != 2 or h.shape[1] != self.n_neurons:
            raise ValueError(f"h must be (B, {self.n_neurons}), got {tuple(h.shape)}")
        if self.pre.numel() == 0:
            return torch.zeros_like(h)
        messages = h.index_select(1, self.pre) * self.weight
        aggregated = torch.zeros_like(h)
        index = self.post.unsqueeze(0).expand_as(messages)
        aggregated.scatter_add_(1, index, messages)
        return aggregated

    def expand_input(self, external_input: torch.Tensor) -> torch.Tensor:
        """Accept (B, N) or (B, n_visual) and return (B, N)."""
        if external_input.ndim == 1:
            external_input = external_input.unsqueeze(0)
        if external_input.ndim != 2:
            raise ValueError("external_input must be (N,), (B, N), or (B, n_visual)")
        batch, width = external_input.shape
        if width == self.n_neurons:
            return external_input
        n_visual = int(self.visual_input_indices.numel())
        if width == n_visual:
            full = external_input.new_zeros(batch, self.n_neurons)
            full[:, self.visual_input_indices] = external_input
            return full
        raise ValueError(
            f"external_input last dim must be {self.n_neurons} or {n_visual}, got {width}"
        )

    def forward(
        self,
        external_input: torch.Tensor,
        h0: torch.Tensor | None = None,
        steps: int | None = None,
    ) -> torch.Tensor:
        """Run rate dynamics. Returns (B, N)."""
        drive = self.expand_input(external_input)
        batch = drive.shape[0]
        hidden = drive.new_zeros(batch, self.n_neurons) if h0 is None else h0
        if hidden.shape != drive.shape:
            raise ValueError("h0 must match expanded external_input")
        n_steps = self.steps if steps is None else steps
        leak = self.leak
        keep = 1.0 - leak
        for _ in range(n_steps):
            preact = self.sparse_mm(hidden) + drive
            hidden = keep * hidden + leak * F.relu(preact)
        return hidden

    def descending_state(self, h: torch.Tensor) -> torch.Tensor:
        """Readout population: (B, n_descending)."""
        return h.index_select(1, self.descending_indices)
