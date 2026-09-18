from collections import Counter

import numpy as np
import torch

from connectome.rewire import filter_edges_to_mask, permute_targets, random_sparse_like


def test_permute_targets_preserves_degrees_and_weights() -> None:
    pre = torch.tensor([0, 0, 1, 2, 2])
    post = torch.tensor([1, 2, 2, 0, 1])
    counts = torch.tensor([1, 2, 3, 4, 5])
    new_pre, new_post, new_counts = permute_targets(pre, post, counts, seed=1)
    assert Counter(new_pre.tolist()) == Counter(pre.tolist())
    assert Counter(new_post.tolist()) == Counter(post.tolist())
    assert Counter(new_counts.tolist()) == Counter(counts.tolist())
    assert not bool((new_pre == new_post).any())


def test_random_sparse_like_preserves_n_and_weights() -> None:
    pre = torch.tensor([0, 0, 1, 2])
    counts = torch.tensor([9, 8, 7, 6])
    new_pre, new_post, new_counts = random_sparse_like(5, pre, counts, seed=2)
    assert new_pre.numel() == pre.numel()
    assert Counter(new_counts.tolist()) == Counter(counts.tolist())
    assert not bool((new_pre == new_post).any())
    assert int(new_pre.min()) >= 0
    assert int(new_post.max()) < 5


def test_filter_edges_to_mask() -> None:
    pre = torch.tensor([0, 1, 2])
    post = torch.tensor([1, 2, 0])
    counts = torch.tensor([1, 2, 3])
    keep = np.array([True, True, False])
    new_pre, new_post, new_counts = filter_edges_to_mask(pre, post, counts, keep)
    assert new_pre.tolist() == [0]
    assert new_post.tolist() == [1]
    assert new_counts.tolist() == [1]
