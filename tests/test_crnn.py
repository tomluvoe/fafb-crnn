import torch

from models.crnn import SparseFAFBCRNN, log1p_weights, normalize_incoming


def _line_net(**kwargs) -> SparseFAFBCRNN:
    # 0 -> 1 -> 2, one synapse each.
    return SparseFAFBCRNN(
        n_neurons=3,
        pre_indices=torch.tensor([0, 1]),
        post_indices=torch.tensor([1, 2]),
        synapse_counts=torch.tensor([1, 1]),
        visual_input_indices=torch.tensor([0]),
        descending_indices=torch.tensor([2]),
        **kwargs,
    )


def test_log1p_weights() -> None:
    counts = torch.tensor([0, 1, 148])
    weights = log1p_weights(counts)
    assert torch.allclose(weights, torch.log1p(counts.float()))


def test_incoming_normalization_sums_to_one() -> None:
    post = torch.tensor([1, 1, 2])
    weight = torch.tensor([1.0, 3.0, 5.0])
    normed = normalize_incoming(post, weight, n_neurons=3)
    incoming = torch.zeros(3)
    incoming.scatter_add_(0, post, normed)
    assert abs(float(incoming[1]) - 1.0) < 1e-6
    assert abs(float(incoming[2]) - 1.0) < 1e-6
    assert float(incoming[0]) == 0.0


def test_no_trainable_parameters() -> None:
    net = _line_net()
    assert list(net.parameters()) == []
    assert "weight" in dict(net.named_buffers())


def test_leak_zero_leaves_state_unchanged() -> None:
    net = _line_net(leak=0.0, steps=5, normalize=True)
    h0 = torch.tensor([[0.2, 0.3, 0.4]])
    out = net.forward(external_input=torch.ones(1, 3), h0=h0)
    assert torch.allclose(out, h0)


def test_signal_reaches_the_end_of_a_line() -> None:
    net = _line_net(leak=1.0, steps=2, normalize=True)
    # leak=1, relu, incoming weights are 1: h copies along the line.
    # t=1: h = [1, 0, 0]; t=2: h = [1, 1, 0] — need 3 steps to fill unit 2.
    two = net.forward(external_input=torch.tensor([[1.0, 0.0, 0.0]]), steps=2)
    three = net.forward(external_input=torch.tensor([[1.0, 0.0, 0.0]]), steps=3)
    assert float(two[0, 2]) == 0.0
    assert float(three[0, 2]) > 0.0
    assert torch.allclose(net.descending_state(three), three[:, 2:3])


def test_visual_input_expands_to_full_state() -> None:
    net = _line_net(leak=1.0, steps=1, normalize=True)
    out = net.forward(external_input=torch.tensor([[0.7]]))
    assert out.shape == (1, 3)
    assert abs(float(out[0, 0]) - 0.7) < 1e-6
    assert float(out[0, 1]) == 0.0


def test_isolated_neuron_stays_zero_without_input() -> None:
    net = SparseFAFBCRNN(
        n_neurons=2,
        pre_indices=torch.tensor([0]),
        post_indices=torch.tensor([0]),
        synapse_counts=torch.tensor([4]),
        visual_input_indices=torch.tensor([0]),
        descending_indices=torch.tensor([1]),
        leak=1.0,
        steps=3,
        normalize=True,
    )
    out = net.forward(external_input=torch.tensor([[1.0, 0.0]]))
    assert float(out[0, 1]) == 0.0


def test_unnormalized_uses_log1p_synapse_count() -> None:
    net = SparseFAFBCRNN(
        n_neurons=2,
        pre_indices=torch.tensor([0]),
        post_indices=torch.tensor([1]),
        synapse_counts=torch.tensor([148]),
        visual_input_indices=torch.tensor([0]),
        descending_indices=torch.tensor([1]),
        leak=1.0,
        steps=2,
        normalize=False,
    )
    out = net.forward(external_input=torch.tensor([[1.0, 0.0]]))
    expected = float(torch.log1p(torch.tensor(148.0)))
    assert abs(float(out[0, 1]) - expected) < 1e-5
