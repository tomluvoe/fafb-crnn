"""Eval hygiene for the linear decoder. Not Phase 6 controls."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SplitMetrics:
    n: int
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    majority_accuracy: float
    chance: float
    confusion: list[list[int]]
    majority_class: str


def confusion_matrix(
    pred: torch.Tensor, labels: torch.Tensor, n_classes: int
) -> torch.Tensor:
    pred = pred.long().cpu()
    labels = labels.long().cpu()
    matrix = torch.zeros(n_classes, n_classes, dtype=torch.int64)
    for target, predicted in zip(labels, pred, strict=True):
        matrix[int(target), int(predicted)] += 1
    return matrix


def balanced_accuracy(matrix: torch.Tensor) -> float:
    recalls: list[float] = []
    for i in range(matrix.shape[0]):
        support = float(matrix[i].sum())
        recalls.append(float(matrix[i, i]) / support if support else 0.0)
    return float(sum(recalls) / len(recalls)) if recalls else 0.0


def macro_f1(matrix: torch.Tensor) -> float:
    scores: list[float] = []
    for i in range(matrix.shape[0]):
        tp = float(matrix[i, i])
        fp = float(matrix[:, i].sum() - matrix[i, i])
        fn = float(matrix[i].sum() - matrix[i, i])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        if precision + recall == 0.0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return float(sum(scores) / len(scores)) if scores else 0.0


def majority_class_index(labels: torch.Tensor) -> int:
    counts = torch.bincount(labels.long().cpu())
    return int(torch.argmax(counts))


@torch.no_grad()
def evaluate_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_names: tuple[str, ...],
    majority_index: int,
) -> SplitMetrics:
    n_classes = len(class_names)
    pred = logits.argmax(dim=1)
    matrix = confusion_matrix(pred, labels, n_classes)
    majority_pred = torch.full_like(labels, majority_index)
    majority_acc = float((majority_pred == labels).float().mean())
    return SplitMetrics(
        n=int(labels.numel()),
        accuracy=float((pred == labels).float().mean()),
        balanced_accuracy=balanced_accuracy(matrix),
        macro_f1=macro_f1(matrix),
        majority_accuracy=majority_acc,
        chance=1.0 / n_classes,
        confusion=matrix.tolist(),
        majority_class=class_names[majority_index],
    )


@torch.no_grad()
def evaluate_decoder(
    decoder: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    class_names: tuple[str, ...],
    majority_index: int,
) -> SplitMetrics:
    decoder.eval()
    logits = decoder(features)
    return evaluate_logits(logits, labels, class_names, majority_index)


def print_split_metrics(
    title: str, metrics: SplitMetrics, class_names: tuple[str, ...]
) -> None:
    print(f"{title} (n={metrics.n:,})")
    print(f"  chance (uniform):          {metrics.chance:.3f}")
    print(
        f"  majority dummy ({metrics.majority_class}): {metrics.majority_accuracy:.3f}"
    )
    print(f"  accuracy:                  {metrics.accuracy:.3f}")
    print(f"  balanced accuracy:         {metrics.balanced_accuracy:.3f}")
    print(f"  macro-F1:                  {metrics.macro_f1:.3f}")
    print("  confusion (rows=true, cols=pred):")
    header = "        " + " ".join(f"{name[:8]:>8}" for name in class_names)
    print(header)
    for name, row in zip(class_names, metrics.confusion, strict=True):
        cells = " ".join(f"{count:8d}" for count in row)
        print(f"  {name[:8]:>6} {cells}")


def metrics_to_dict(metrics: SplitMetrics) -> dict:
    return {
        "n": metrics.n,
        "chance": metrics.chance,
        "majority_class": metrics.majority_class,
        "majority_accuracy": metrics.majority_accuracy,
        "accuracy": metrics.accuracy,
        "balanced_accuracy": metrics.balanced_accuracy,
        "macro_f1": metrics.macro_f1,
        "confusion": metrics.confusion,
    }
