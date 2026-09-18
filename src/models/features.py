"""Extract frozen FAFB descending features. Encoder and CRNN are not trained."""

from __future__ import annotations

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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (N, n_descending) features and (N,) labels. No grad."""
    if device is not None:
        encoder.to(device)
        crnn.to(device)
    crnn.eval()
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for images, batch_labels in loader:
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
    return torch.cat(features, dim=0), torch.cat(labels, dim=0).long()
