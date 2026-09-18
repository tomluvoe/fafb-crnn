"""Linear readout from descending-neuron activity.

Train this module only. The FAFB CRNN stays frozen. Accuracy is a
property of this model, not evidence that flies classify animals.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class LinearDecoder(nn.Module):
    """h_descending -> class logits."""

    def __init__(self, n_features: int, n_classes: int) -> None:
        super().__init__()
        if n_features <= 0 or n_classes <= 1:
            raise ValueError("need n_features > 0 and n_classes > 1")
        self.linear = nn.Linear(n_features, n_classes)

    def forward(self, descending: torch.Tensor) -> torch.Tensor:
        return self.linear(descending)


@torch.no_grad()
def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == labels).float().mean())


def standardize_features(
    train_features: torch.Tensor, val_features: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Z-score using train statistics. Engineering for probe scale, not biology."""
    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0).clamp(min=1e-6)
    return (train_features - mean) / std, (val_features - mean) / std, mean, std


def train_linear_decoder(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    *,
    n_classes: int,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-2,
    seed: int = 0,
    standardize: bool = True,
    device: torch.device | str | None = None,
) -> tuple[LinearDecoder, dict[str, list[float]]]:
    """Fit a linear classifier on cached descending features."""
    torch.manual_seed(seed)
    if standardize:
        train_features, val_features, _, _ = standardize_features(
            train_features, val_features
        )
    if device is not None:
        train_features = train_features.to(device)
        val_features = val_features.to(device)
        train_labels = train_labels.to(device)
        val_labels = val_labels.to(device)
    n_features = int(train_features.shape[1])
    decoder = LinearDecoder(n_features, n_classes)
    if device is not None:
        decoder = decoder.to(device)
    opt = torch.optim.Adam(decoder.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(train_features, train_labels.long()),
        batch_size=batch_size,
        shuffle=True,
    )
    history: dict[str, list[float]] = {"train_acc": [], "val_acc": []}
    decoder.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(decoder(batch_x), batch_y)
            loss.backward()
            opt.step()
        decoder.eval()
        with torch.no_grad():
            train_acc = accuracy(decoder(train_features), train_labels.long())
            val_acc = accuracy(decoder(val_features), val_labels.long())
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        decoder.train()
    decoder.eval()
    return decoder, history
