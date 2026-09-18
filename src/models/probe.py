"""Shared linear-probe fit + metrics for Phase 5 and Phase 6."""

from __future__ import annotations

from models.decoder import apply_standardize, train_linear_decoder
from models.metrics import (
    SplitMetrics,
    evaluate_decoder,
    majority_class_index,
    metrics_to_dict,
)


def fit_linear_probe(
    train_x,
    train_y,
    val_x,
    val_y,
    class_names: tuple[str, ...],
    *,
    epochs: int,
    lr: float,
    seed: int,
    device,
) -> dict:
    result = train_linear_decoder(
        train_x,
        train_y,
        val_x,
        val_y,
        n_classes=len(class_names),
        epochs=epochs,
        lr=lr,
        seed=seed,
        device=device,
    )
    decoder = result.decoder.to("cpu")
    train_z = apply_standardize(train_x, result.feature_mean, result.feature_std)
    val_z = apply_standardize(val_x, result.feature_mean, result.feature_std)
    majority = majority_class_index(train_y)
    train_metrics: SplitMetrics = evaluate_decoder(
        decoder, train_z, train_y, class_names, majority
    )
    val_metrics: SplitMetrics = evaluate_decoder(
        decoder, val_z, val_y, class_names, majority
    )
    return {
        "best_epoch": result.best_epoch,
        "best_val_acc": result.best_val_acc,
        "history": result.history,
        "train": metrics_to_dict(train_metrics),
        "val": metrics_to_dict(val_metrics),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "decoder": decoder,
    }
