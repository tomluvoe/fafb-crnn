import torch

from models.device import resolve_device


def test_auto_device_is_torch_device() -> None:
    device = resolve_device("auto")
    assert isinstance(device, torch.device)
    assert device.type in {"cpu", "mps", "cuda"}


def test_cpu_device() -> None:
    assert resolve_device("cpu").type == "cpu"
