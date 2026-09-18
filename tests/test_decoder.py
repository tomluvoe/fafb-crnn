import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.crnn import SparseFAFBCRNN
from models.decoder import LinearDecoder, train_linear_decoder
from models.features import extract_descending_features
from vision.encoder import ColumnL123Encoder


def test_linear_decoder_shapes() -> None:
    decoder = LinearDecoder(n_features=7, n_classes=3)
    logits = decoder(torch.randn(4, 7))
    assert logits.shape == (4, 3)
    assert list(decoder.parameters())


def test_linear_decoder_fits_separable_features() -> None:
    train_x = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [1.0, 0.05], [0.05, 1.0]]
    )
    train_y = torch.tensor([0, 0, 1, 1, 0, 1])
    val_x = torch.tensor([[0.95, 0.05], [0.05, 0.95]])
    val_y = torch.tensor([0, 1])
    decoder, history = train_linear_decoder(
        train_x,
        train_y,
        val_x,
        val_y,
        n_classes=2,
        epochs=40,
        batch_size=4,
        lr=0.2,
        seed=0,
    )
    assert history["val_acc"][-1] == 1.0
    # Already-separable 2-D features; z-score must not break this.
    assert decoder(val_x).argmax(1).tolist() == [0, 1]


def test_crnn_has_no_decoder_parameters() -> None:
    net = SparseFAFBCRNN(
        n_neurons=3,
        pre_indices=torch.tensor([0, 1]),
        post_indices=torch.tensor([1, 2]),
        synapse_counts=torch.tensor([1, 1]),
        visual_input_indices=torch.tensor([0]),
        descending_indices=torch.tensor([2]),
    )
    assert list(net.parameters()) == []


def test_extract_features_uses_descending_units() -> None:
    # Tiny encoder map: one L3 at index 0.
    table = pd.DataFrame(
        {
            "neuron_index": [0],
            "type": ["L3"],
            "x": [0],
            "y": [0],
        }
    )
    encoder = ColumnL123Encoder(table, delta="spatial")
    crnn = SparseFAFBCRNN(
        n_neurons=3,
        pre_indices=torch.tensor([0]),
        post_indices=torch.tensor([2]),
        synapse_counts=torch.tensor([8]),
        visual_input_indices=torch.tensor([0]),
        descending_indices=torch.tensor([2]),
        leak=1.0,
        steps=2,
        normalize=True,
    )
    images = torch.ones(2, 3, 8, 8)
    labels = torch.tensor([0, 1])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    features, out_labels = extract_descending_features(loader, encoder, crnn)
    assert features.shape == (2, 1)
    assert out_labels.tolist() == [0, 1]
    assert list(crnn.parameters()) == []
