
import pytest
import torch

from sinkhorn import sinkhorn_torch, marginal_errors


def test_square_doubly_stochastic_marginals():
    kernel = torch.tensor([
        [1.0, 2.0, 3.0],
        [2.0, 1.0, 2.0],
        [3.0, 2.0, 1.0],
    ])

    plan = sinkhorn_torch(kernel, n_iters=200, log_domain=False)

    row_error, column_error = marginal_errors(plan)

    assert row_error < 1e-5
    assert column_error < 1e-5


def test_rectangular_default_marginals():
    kernel = torch.rand(4, 6) + 0.1

    plan = sinkhorn_torch(
        kernel,
        n_iters=200,
        log_domain=False,
    )

    assert torch.allclose(
        plan.sum(dim=-1),
        torch.ones(4),
        atol=1e-5,
    )

    assert torch.allclose(
        plan.sum(dim=-2),
        torch.full((6,), 4 / 6),
        atol=1e-5,
    )


def test_arbitrary_marginals():
    kernel = torch.rand(3, 2) + 0.1

    row_mass = torch.tensor([1.0, 2.0, 1.0])
    column_mass = torch.tensor([2.0, 2.0])

    plan = sinkhorn_torch(
        kernel,
        n_iters=200,
        log_domain=False,
        row_mass=row_mass,
        column_mass=column_mass,
    )

    assert torch.allclose(
        plan.sum(dim=-1), row_mass, atol=1e-5
    )
    assert torch.allclose(
        plan.sum(dim=-2), column_mass, atol=1e-5
    )


def test_zero_marginal_is_preserved():
    kernel = torch.rand(3, 3) + 0.1

    row_mass = torch.tensor([1.0, 1.0, 1.0])
    column_mass = torch.tensor([2.0, 1.0, 0.0])

    plan = sinkhorn_torch(
        kernel,
        n_iters=100,
        log_domain=False,
        row_mass=row_mass,
        column_mass=column_mass,
    )

    assert torch.equal(plan[:, 2], torch.zeros(3))
    assert torch.allclose(
        plan.sum(dim=-1), row_mass, atol=1e-5
    )


def test_mismatched_total_mass_is_rejected():
    kernel = torch.rand(3, 2) + 0.1

    with pytest.raises(ValueError, match="equal total mass"):
        sinkhorn_torch(
            kernel,
            log_domain=False,
            row_mass=torch.tensor([1.0, 1.0, 1.0]),
            column_mass=torch.tensor([1.0, 1.0]),
        )


def test_negative_marginal_is_rejected():
    kernel = torch.rand(2, 2) + 0.1

    with pytest.raises(ValueError, match="nonnegative"):
        sinkhorn_torch(
            kernel,
            log_domain=False,
            row_mass=torch.tensor([1.0, -1.0]),
            column_mass=torch.tensor([0.0, 0.0]),
        )


def test_gradient_flows_through_sinkhorn():
    kernel = torch.rand(
        3, 3, dtype=torch.double, requires_grad=True
    ) + 0.5

    # Make this a leaf tensor for gradient inspection.
    kernel = kernel.detach().requires_grad_(True)

    plan = sinkhorn_torch(
        kernel,
        n_iters=100,
        log_domain=False,
    )

    probe = torch.tensor(
        [[1.0, 2.0, 3.0],
         [3.0, 1.0, 2.0],
         [2.0, 3.0, 1.0]],
        dtype=torch.double,
    )

    loss = (plan * probe).sum()
    loss.backward()

    assert kernel.grad is not None
    assert torch.isfinite(kernel.grad).all()
    assert kernel.grad.abs().sum() > 0
