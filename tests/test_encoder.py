import pandas as pd
import torch

from vision.encoder import DELTA_GRAY, ColumnL123Encoder, luminance


def _map() -> pd.DataFrame:
    # Two columns on a line, both hemispheres share (x, y).
    rows = []
    neuron = 0
    for x in (0, 1):
        for hemi in ("left", "right"):
            for cell_type in ("L1", "L2", "L3"):
                rows.append(
                    {
                        "neuron_index": neuron,
                        "root_id": 1000 + neuron,
                        "type": cell_type,
                        "hemisphere": hemi,
                        "x": x,
                        "y": 0,
                    }
                )
                neuron += 1
    return pd.DataFrame(rows)


def test_luminance_rgb_weights() -> None:
    red = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]]).repeat(1, 3, 1, 1)
    red[:, 1:] = 0
    luma = luminance(red)
    assert luma.shape == (1, 1, 2, 2)
    assert abs(float(luma[0, 0, 0, 0]) - 0.2126) < 1e-5


def test_uniform_image_has_no_spatial_on_off() -> None:
    encoder = ColumnL123Encoder(_map(), delta="spatial")
    image = torch.full((1, 3, 8, 8), 0.4)
    values = encoder.encode(image)
    types = _map()["type"].to_numpy()
    assert torch.allclose(values[0, torch.tensor(types == "L3")], torch.tensor(0.4))
    assert torch.allclose(values[0, torch.tensor(types == "L1")], torch.zeros(4))
    assert torch.allclose(values[0, torch.tensor(types == "L2")], torch.zeros(4))


def test_spatial_on_off_splits_a_left_right_ramp() -> None:
    encoder = ColumnL123Encoder(_map(), delta="spatial")
    # Bright on the left half of the image, dark on the right.
    image = torch.zeros(1, 1, 4, 8)
    image[:, :, :, :4] = 1.0
    values = encoder.encode(image)
    table = _map()
    table = table.assign(value=values[0].tolist())
    left_l1 = table[(table["x"] == 0) & (table["type"] == "L1")]["value"].mean()
    left_l2 = table[(table["x"] == 0) & (table["type"] == "L2")]["value"].mean()
    right_l1 = table[(table["x"] == 1) & (table["type"] == "L1")]["value"].mean()
    right_l2 = table[(table["x"] == 1) & (table["type"] == "L2")]["value"].mean()
    assert left_l1 > left_l2
    assert right_l2 > right_l1


def test_gray_delta_is_half_wave_around_mid_gray() -> None:
    encoder = ColumnL123Encoder(_map(), delta=DELTA_GRAY)
    bright = encoder.encode(torch.ones(1, 1, 4, 4))
    dark = encoder.encode(torch.zeros(1, 1, 4, 4))
    types = _map()["type"].to_numpy()
    is_l1 = torch.tensor(types == "L1")
    is_l2 = torch.tensor(types == "L2")
    assert torch.allclose(bright[0, is_l1], torch.full((4,), 0.5))
    assert torch.allclose(bright[0, is_l2], torch.zeros(4))
    assert torch.allclose(dark[0, is_l1], torch.zeros(4))
    assert torch.allclose(dark[0, is_l2], torch.full((4,), 0.5))


def test_hemispheres_share_the_same_field() -> None:
    encoder = ColumnL123Encoder(_map(), delta="spatial")
    image = torch.linspace(0, 1, 8).view(1, 1, 1, 8).expand(1, 1, 4, 8)
    values = encoder.encode(image)
    table = _map().assign(value=values[0].tolist())
    for x in (0, 1):
        for cell_type in ("L1", "L2", "L3"):
            pair = table[(table["x"] == x) & (table["type"] == cell_type)]
            assert pair["hemisphere"].nunique() == 2
            assert abs(pair["value"].iat[0] - pair["value"].iat[1]) < 1e-6


def test_to_state_writes_only_visual_indices() -> None:
    encoder = ColumnL123Encoder(_map(), delta="spatial")
    image = torch.ones(3, 8, 8)
    visual = encoder.encode(image)
    state = encoder.to_state(image, n_neurons=20)
    assert state.shape == (1, 20)
    table = _map()
    for row, neuron_index in enumerate(table["neuron_index"].tolist()):
        assert float(state[0, neuron_index]) == float(visual[0, row])
    filled = set(table["neuron_index"].tolist())
    for i in range(20):
        if i not in filled:
            assert float(state[0, i]) == 0.0


def test_encode_does_not_use_root_id() -> None:
    table = _map().drop(columns=["root_id"])
    encoder = ColumnL123Encoder(table, delta="spatial")
    values = encoder.encode(torch.ones(1, 8, 8))
    assert values.shape == (1, len(table))
