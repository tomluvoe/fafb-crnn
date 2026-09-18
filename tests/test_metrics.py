import torch

from models.metrics import (
    balanced_accuracy,
    confusion_matrix,
    evaluate_logits,
    macro_f1,
    majority_class_index,
)


def test_majority_and_confusion() -> None:
    labels = torch.tensor([0, 0, 0, 1, 2])
    logits = torch.tensor(
        [
            [3.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    names = ("a", "b", "c")
    metrics = evaluate_logits(logits, labels, names, majority_class_index(labels))
    assert metrics.majority_class == "a"
    assert abs(metrics.majority_accuracy - 0.6) < 1e-6
    assert abs(metrics.accuracy - 0.6) < 1e-6
    assert metrics.confusion == [[3, 0, 0], [1, 0, 0], [1, 0, 0]]


def test_balanced_accuracy_and_macro_f1() -> None:
    matrix = torch.tensor([[2, 0], [2, 2]])
    assert confusion_matrix(
        torch.tensor([0, 0, 1, 1]), torch.tensor([0, 0, 1, 1]), 2
    ).tolist() == [
        [2, 0],
        [0, 2],
    ]
    assert abs(balanced_accuracy(matrix) - 0.75) < 1e-6
    # class0: p=2/4, r=1, f1=2/3; class1: p=1, r=0.5, f1=2/3
    assert abs(macro_f1(matrix) - (2 / 3)) < 1e-6
