
import pytest
import torch

from topk import select_weights


@pytest.mark.parametrize(
    "mode",
    [
        "soft",
        "soft_fixed",
        "hard",
        "random",
        "attention",
        "uniform",
        "straight_through",
    ],
)
def test_baseline_output_shape_and_mass(mode):
    torch.manual_seed(42)

    scores = torch.tensor([
        [0.1, 0.3, 0.6, 0.9],
        [0.8, 0.2, 0.4, 0.5],
    ])

    k = 2

    weights = select_weights(
        scores,
        k,
        mode=mode,
        temperature=0.1,
        n_iters=100,
    )

    assert weights.shape == scores.shape

    assert torch.allclose(
        weights.sum(dim=-1),
        torch.full((2,), float(k)),
        atol=1e-3,
    )


def test_hard_topk_is_binary():
    scores = torch.tensor([[0.1, 0.3, 0.6, 0.9]])

    weights = select_weights(scores, 2, mode="hard")

    assert torch.all((weights == 0) | (weights == 1))
    assert weights.sum().item() == 2


def test_uniform_selection():
    scores = torch.tensor([[0.1, 0.3, 0.6, 0.9]])

    weights = select_weights(scores, 2, mode="uniform")

    expected = torch.full_like(scores, 0.5)

    assert torch.allclose(weights, expected)


def test_attention_weights_sum_to_k():
    scores = torch.tensor([[0.1, 0.3, 0.6, 0.9]])

    weights = select_weights(scores, 2, mode="attention")

    assert torch.all(weights >= 0)
    assert torch.allclose(
        weights.sum(dim=-1),
        torch.tensor([2.0]),
        atol=1e-6,
    )


@pytest.mark.parametrize("mode", ["soft", "soft_fixed", "attention",
                                  "straight_through"])
def test_trainable_selection_modes_have_gradients(mode):
    scores = torch.tensor(
        [[0.1, 0.3, 0.6, 0.9]],
        requires_grad=True,
    )

    weights = select_weights(
        scores,
        2,
        mode=mode,
        temperature=0.1,
        n_iters=100,
    )

    probe = torch.tensor([[1.0, 2.0, 4.0, 8.0]])

    loss = (weights * probe).sum()
    loss.backward()

    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()
    assert scores.grad.norm() > 1e-8


def test_unknown_selection_mode():
    scores = torch.tensor([[0.1, 0.3, 0.6, 0.9]])

    with pytest.raises(ValueError):
        select_weights(scores, 2, mode="unknown")
