"""Extract frozen FAFB descending features. Encoder and CRNN are not trained."""

from __future__ import annotations

import time

import torch
from torch.utils.data import DataLoader

from models.crnn import SparseFAFBCRNN
from vision.encoder import ColumnL123Encoder


@torch.no_grad()
def extract_descending_features(
    loader: DataLoader,
    encoder: ColumnL123Encoder,
    crnn: SparseFAFBCRNN,
    device: torch.device | str | None = None,
    *,
    progress: bool = False,
    desc: str = "extract",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (N, n_descending) features and (N,) labels. No grad."""
    if device is not None:
        encoder.to(device)
        crnn.to(device)
    crnn.eval()
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    n_batches = len(loader)
    seen = 0
    start = time.perf_counter()
    for i, (images, batch_labels) in enumerate(loader, start=1):
        if device is not None:
            images = images.to(device)
        visual = encoder.encode(images)
        hidden = crnn(visual)
        features.append(crnn.descending_state(hidden).cpu())
        labels.append(
            batch_labels.cpu()
            if isinstance(batch_labels, torch.Tensor)
            else torch.as_tensor(batch_labels)
        )
        seen += int(images.shape[0])
        if progress and (i == 1 or i == n_batches or i % 10 == 0):
            elapsed = time.perf_counter() - start
            print(
                f"{desc}: batch {i}/{n_batches}  images {seen}  {elapsed:.1f}s",
                flush=True,
            )
    return torch.cat(features, dim=0), torch.cat(labels, dim=0).long()


@torch.no_grad()
def extract_visual_features(
    loader: DataLoader,
    encoder: ColumnL123Encoder,
    device: torch.device | str | None = None,
    *,
    progress: bool = False,
    desc: str = "visual",
) -> tuple[torch.Tensor, torch.Tensor]:
    """L1/L2/L3 encoder samples only. No CRNN."""
    if device is not None:
        encoder.to(device)
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    n_batches = len(loader)
    seen = 0
    start = time.perf_counter()
    for i, (images, batch_labels) in enumerate(loader, start=1):
        if device is not None:
            images = images.to(device)
        features.append(encoder.encode(images).cpu())
        labels.append(
            batch_labels.cpu()
            if isinstance(batch_labels, torch.Tensor)
            else torch.as_tensor(batch_labels)
        )
        seen += int(images.shape[0])
        if progress and (i == 1 or i == n_batches or i % 10 == 0):
            elapsed = time.perf_counter() - start
            print(
                f"{desc}: batch {i}/{n_batches}  images {seen}  {elapsed:.1f}s",
                flush=True,
            )
    return torch.cat(features, dim=0), torch.cat(labels, dim=0).long()
