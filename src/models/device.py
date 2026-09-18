"""Pick a torch device. Apple silicon uses MPS, not CUDA."""

from __future__ import annotations

import torch


def resolve_device(name: str = "auto") -> torch.device:
    """Return a torch.device.

    auto: mps (Apple GPU) if available, else cuda, else cpu.
    """
    key = name.strip().lower()
    if key == "cpu":
        return torch.device("cpu")
    if key == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available on this machine")
        return torch.device("mps")
    if key == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this machine")
        return torch.device("cuda")
    if key != "auto":
        raise ValueError(f"device must be auto, mps, cuda, or cpu, got {name!r}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
