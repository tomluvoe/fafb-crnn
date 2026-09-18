import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from data.dataset import Animals10Dataset
from models.cnn import SmallCNN, cnn_logits, train_cnn


def _write(path, rgb) -> None:
    Image.fromarray(np.full((16, 16, 3), rgb, dtype=np.uint8)).save(path)


def test_small_cnn_trains_on_fake_images(tmp_path) -> None:
    for italian, color in (("farfalla", (255, 0, 0)), ("elefante", (0, 0, 255))):
        folder = tmp_path / italian
        folder.mkdir()
        for i in range(6):
            _write(folder / f"{i}.jpeg", color)
    train, val = Animals10Dataset.splits(
        raw_dir=tmp_path,
        classes=("butterfly", "elephant"),
        seed=0,
        image_size=16,
    )
    train_loader = DataLoader(train, batch_size=4, shuffle=True)
    val_loader = DataLoader(val, batch_size=4, shuffle=False)
    model, history = train_cnn(
        train_loader,
        val_loader,
        n_classes=2,
        epochs=3,
        device="cpu",
        seed=0,
    )
    assert isinstance(model, SmallCNN)
    assert len(history["val_acc"]) == 3
    logits, labels = cnn_logits(model, val_loader, "cpu")
    assert logits.shape[0] == len(val)
    assert logits.shape[1] == 2
    assert labels.numel() == len(val)
