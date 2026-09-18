"""Small grayscale CNN baseline. Color is not used, to match the FAFB encoder."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader

from vision.encoder import as_batched_image, luminance


class SmallCNN(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        if n_classes <= 1:
            raise ValueError("n_classes must be > 1")
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        gray = luminance(as_batched_image(images))
        pooled = self.features(gray).flatten(1)
        return self.classifier(pooled)


def train_cnn(
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_classes: int,
    *,
    epochs: int = 20,
    lr: float = 1e-3,
    device: torch.device | str,
    seed: int = 0,
) -> tuple[SmallCNN, dict[str, list[float]]]:
    torch.manual_seed(seed)
    model = SmallCNN(n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    history: dict[str, list[float]] = {"train_acc": [], "val_acc": []}
    best_val = -1.0
    best_state = deepcopy(model.state_dict())

    def _accuracy(loader: DataLoader) -> float:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                pred = model(images).argmax(1)
                correct += int((pred == labels).sum())
                total += int(labels.numel())
        return correct / total if total else 0.0

    for _ in range(epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward()
            opt.step()
        train_acc = _accuracy(train_loader)
        val_acc = _accuracy(val_loader)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        if val_acc > best_val:
            best_val = val_acc
            best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return model, history


@torch.no_grad()
def cnn_logits(model: SmallCNN, loader: DataLoader, device: torch.device | str):
    model.eval()
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for images, batch_labels in loader:
        images = images.to(device)
        logits.append(model(images).cpu())
        labels.append(
            batch_labels.cpu()
            if isinstance(batch_labels, torch.Tensor)
            else torch.as_tensor(batch_labels)
        )
    return torch.cat(logits, dim=0), torch.cat(labels, dim=0).long()
