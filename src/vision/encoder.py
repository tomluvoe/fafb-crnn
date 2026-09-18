"""v0 visual encoder: sample an image onto FAFB L1 / L2 / L3 columns.

This is an engineering stand-in, not fly phototransduction. R1-R6 have no
column coordinates in this dataset, so the image is sampled at column
(x, y) and written into lamina monopolar cells.

Still-image ON/OFF (v0 default: spatial):

    ΔI = I - mean(I at 4-connected neighbor columns)
    L1 ← max(ΔI, 0)     ON / brighter than neighbors
    L2 ← max(-ΔI, 0)    OFF / darker than neighbors
    L3 ← I              intensity

A gray-to-image flash (delta='gray') is also available: ΔI = I - 0.5.
Both hemispheres receive the same sampled field (no stereo).
(x, y) are lattice indices, not visual degrees.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from connectome.populations import V0_VISUAL_INPUT_TYPES

DELTA_SPATIAL = "spatial"
DELTA_GRAY = "gray"
GRAY_LEVEL = 0.5
NEIGHBOR_OFFSETS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def luminance(image: torch.Tensor) -> torch.Tensor:
    """Convert (B, C, H, W) to (B, 1, H, W) luminance in [0, 1]."""
    if image.ndim != 4:
        raise ValueError(f"image must be (B, C, H, W), got {tuple(image.shape)}")
    batch, channels, _height, _width = image.shape
    pixel = image.to(dtype=torch.float32)
    if float(pixel.max()) > 1.0:
        pixel = pixel / 255.0
    if channels == 1:
        return pixel
    if channels == 3:
        weight = pixel.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
        return (pixel * weight).sum(dim=1, keepdim=True)
    return pixel.mean(dim=1, keepdim=True)


def as_batched_image(image: torch.Tensor) -> torch.Tensor:
    """Accept (H, W), (C, H, W), or (B, C, H, W); return (B, C, H, W)."""
    if image.ndim == 2:
        return image.unsqueeze(0).unsqueeze(0)
    if image.ndim == 3:
        return image.unsqueeze(0)
    if image.ndim == 4:
        return image
    raise ValueError(f"image must be 2-D, 3-D, or 4-D, got {tuple(image.shape)}")


def _sample_at_columns(
    luma: torch.Tensor, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Bilinear-sample (B, 1, H, W) at lattice x, y. Returns (B, n)."""
    x_min = float(x.min())
    x_max = float(x.max())
    y_min = float(y.min())
    y_max = float(y.max())
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1.0)
    # x_min → left, y_max → top of the image (row 0).
    grid_x = 2.0 * (x - x_min) / x_span - 1.0
    grid_y = 1.0 - 2.0 * (y - y_min) / y_span
    batch = luma.shape[0]
    n = int(x.numel())
    grid = torch.stack(
        [
            grid_x.to(device=luma.device, dtype=luma.dtype).expand(batch, n),
            grid_y.to(device=luma.device, dtype=luma.dtype).expand(batch, n),
        ],
        dim=-1,
    ).view(batch, 1, n, 2)
    sampled = F.grid_sample(
        luma,
        grid,
        mode="bilinear",
        align_corners=True,
        padding_mode="border",
    )
    return sampled[:, 0, 0, :]


def _neighbor_index(x: np.ndarray, y: np.ndarray) -> torch.Tensor:
    """For each unique column, 4-connected neighbor indices, or -1."""
    lookup = {
        (int(xi), int(yi)): i for i, (xi, yi) in enumerate(zip(x, y, strict=True))
    }
    neighbors = np.full((len(x), 4), -1, dtype=np.int64)
    for i, (xi, yi) in enumerate(zip(x, y, strict=True)):
        for k, (dx, dy) in enumerate(NEIGHBOR_OFFSETS):
            j = lookup.get((int(xi) + dx, int(yi) + dy))
            if j is not None:
                neighbors[i, k] = j
    return torch.from_numpy(neighbors)


class ColumnL123Encoder:
    """Map an image to L1/L2/L3 values using a retinotopic column map.

    `retinotopic_map` must include neuron_index, type, x, y (the processed
    FAFB parquet). Output rows follow that table's order.
    """

    def __init__(
        self,
        retinotopic_map: pd.DataFrame,
        *,
        delta: str = DELTA_SPATIAL,
    ) -> None:
        required = {"neuron_index", "type", "x", "y"}
        missing = required.difference(retinotopic_map.columns)
        if missing:
            raise ValueError(f"retinotopic_map missing {sorted(missing)}")
        if retinotopic_map.empty:
            raise ValueError("retinotopic_map is empty")
        unknown = sorted(
            set(retinotopic_map["type"].astype(str)) - set(V0_VISUAL_INPUT_TYPES)
        )
        if unknown:
            raise ValueError(f"unsupported visual types: {unknown}")
        if delta not in {DELTA_SPATIAL, DELTA_GRAY}:
            raise ValueError(f"delta must be {DELTA_SPATIAL!r} or {DELTA_GRAY!r}")

        self.delta = delta
        self.map = retinotopic_map.reset_index(drop=True)
        self.neuron_index = torch.tensor(
            self.map["neuron_index"].to_numpy(copy=True), dtype=torch.int64
        )
        self.types = self.map["type"].astype(str).to_numpy()
        unique = self.map[["x", "y"]].drop_duplicates().reset_index(drop=True)
        self._unique_x = torch.tensor(
            unique["x"].to_numpy(copy=True), dtype=torch.float32
        )
        self._unique_y = torch.tensor(
            unique["y"].to_numpy(copy=True), dtype=torch.float32
        )
        key = pd.MultiIndex.from_frame(self.map[["x", "y"]])
        unique_key = pd.MultiIndex.from_frame(unique)
        self._row_to_unique = torch.as_tensor(
            unique_key.get_indexer(key), dtype=torch.int64
        )
        if int(self._row_to_unique.min()) < 0:
            raise ValueError("retinotopic_map (x, y) failed unique-column lookup")
        self._neighbors = _neighbor_index(
            unique["x"].to_numpy(), unique["y"].to_numpy()
        )
        self._is_l1 = torch.as_tensor(self.types == "L1")
        self._is_l2 = torch.as_tensor(self.types == "L2")
        self._is_l3 = torch.as_tensor(self.types == "L3")

    @classmethod
    def from_processed_dir(cls, processed_dir: Path, **kwargs) -> ColumnL123Encoder:
        path = Path(processed_dir) / "retinotopic_map.parquet"
        return cls(pd.read_parquet(path), **kwargs)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """Return (B, n_visual) in retinotopic_map row order."""
        batched = as_batched_image(image)
        luma = luminance(batched)
        intensity = _sample_at_columns(luma, self._unique_x, self._unique_y)
        delta = self._delta_i(intensity)
        column_i = intensity[:, self._row_to_unique]
        column_d = delta[:, self._row_to_unique]
        values = torch.zeros(
            batched.shape[0],
            len(self.map),
            dtype=luma.dtype,
            device=luma.device,
        )
        values[:, self._is_l3] = column_i[:, self._is_l3]
        values[:, self._is_l1] = column_d[:, self._is_l1].clamp(min=0)
        values[:, self._is_l2] = (-column_d[:, self._is_l2]).clamp(min=0)
        return values

    def to_state(self, image: torch.Tensor, n_neurons: int) -> torch.Tensor:
        """Return (B, n_neurons) with encoded values at neuron_index."""
        visual = self.encode(image)
        if (
            int(self.neuron_index.min()) < 0
            or int(self.neuron_index.max()) >= n_neurons
        ):
            raise ValueError("neuron_index out of range for n_neurons")
        state = visual.new_zeros(visual.shape[0], n_neurons)
        state[:, self.neuron_index] = visual
        return state

    def _delta_i(self, intensity: torch.Tensor) -> torch.Tensor:
        if self.delta == DELTA_GRAY:
            return intensity - GRAY_LEVEL
        neighbors = self._neighbors.to(device=intensity.device)
        mask = neighbors >= 0
        gathered = intensity[:, neighbors.clamp(min=0)]
        counts = mask.sum(dim=1).clamp(min=1).to(dtype=intensity.dtype)
        local_mean = (gathered * mask.to(dtype=intensity.dtype)).sum(dim=-1) / counts
        none = ~mask.any(dim=1)
        local_mean = torch.where(none, intensity, local_mean)
        return intensity - local_mean
