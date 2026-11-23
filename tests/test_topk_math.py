import pytest
import torch

from topk import soft_topk, soft_topk_fixed


@pytest.mark.parametrize("function", [soft_topk, soft_topk_fixed])
def test_selection_mass_and_bounds(function):
    scores = torch.tensor([[0.1, 0.3, 0.6, 0.9]])
    k = 2

    weights = function(
        scores,
        k,
        temperature=0.1,
        n_iters=100,
    )

    assert torch.all(weights >= 0)
    assert torch.all(weights <= 1.001)
    assert torch.allclose(
        weights.sum(),
        torch.tensor(float(k)),
        atol=1e-3,
    )


@pytest.mark.parametrize("function", [soft_topk, soft_topk_fixed])
def test_higher_scores_receive_higher_weights(function):
    scores = torch.tensor([[0.0, 0.2, 0.7, 1.0]])

    weights = function(
        scores,
        k=2,
        temperature=0.05,
        n_iters=100,
    )

    assert weights[0, 0] < weights[0, 1]
    assert weights[0, 1] < weights[0, 2]
    assert weights[0, 2] < weights[0, 3]


@pytest.mark.parametrize("function", [soft_topk, soft_topk_fixed])
def test_permutation_equivariance(function):
    scores = torch.tensor([[0.1, 0.4, 0.7, 0.9]])
    permutation = torch.tensor([2, 0, 3, 1])

    original_weights = function(
        scores,
        k=2,
        temperature=0.1,
        n_iters=100,
    )

    permuted_weights = function(
        scores[:, permutation],
        k=2,
        temperature=0.1,
        n_iters=100,
    )

    assert torch.allclose(
        permuted_weights,
        original_weights[:, permutation],
        atol=1e-4,
    )

@pytest.mark.parametrize("function", [soft_topk, soft_topk_fixed])
def test_selecting_all_items_returns_ones(function):
    scores = torch.tensor([[0.2, 0.4, 0.6, 0.8]])

    weights = function(scores, k=4)

    assert torch.equal(weights, torch.ones_like(scores))


def test_fixed_anchor_constant_scores_are_uniform():
    scores = torch.tensor([[0.5, 0.5, 0.5, 0.5]])

    weights = soft_topk_fixed(
        scores,
        k=2,
        temperature=0.1,
        n_iters=100,
    )

    expected = torch.full_like(scores, 0.5)

    assert torch.allclose(weights, expected, atol=1e-3)


@pytest.mark.parametrize("function", [soft_topk, soft_topk_fixed])
def test_gradient_is_finite_and_nonzero(function):
    scores = torch.tensor(
    [[0.1, 0.3, 0.6, 0.9]],
    requires_grad=True,
)
    weights = function(
        scores,
        k=2,
        temperature=0.1,
        n_iters=100,
    )

    probe = torch.tensor([[1.0, 2.0, 4.0, 8.0]])
    loss = (weights * probe).sum()

    loss.backward()

    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()
    assert scores.grad.norm() > 0


@pytest.mark.parametrize("function", [soft_topk, soft_topk_fixed])
def test_invalid_temperature_is_rejected(function):
    scores = torch.tensor([0.1, 0.3, 0.6, 0.9])

    with pytest.raises(ValueError):
        function(scores, k=2, temperature=0)

    with pytest.raises(ValueError):
        function(scores, k=2, temperature=float("nan"))