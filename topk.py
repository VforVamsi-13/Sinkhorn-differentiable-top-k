"""Differentiable and baseline top-k selection utilities."""
from __future__ import annotations
from typing import Literal
import torch 
import math
from sinkhorn import sinkhorn_torch

def _validate_scores(scores: torch.Tensor, k: int) -> None:
    if scores.ndim < 2:
        raise ValueError("scores must have shape [..., n_items]")
    if not scores.is_floating_point():
        raise TypeError("scores must be a floating-point tensor")
    if not torch.isfinite(scores).all():
        raise ValueError("scores must contain only finite values")
    if not 1 <= k <= scores.shape[-1]:
        raise ValueError("k must be between one and the number of items")

def soft_topk(scores, k, temperature=0.1, n_iters=50):
    """
    Differentiable top-k relaxation using data-dependent min/max anchors.

    Returns soft selection weights in [0, 1], with total mass
    approximately equal to k.
    """
    _validate_scores(scores, k)

    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")

    if not isinstance(n_iters, int) or n_iters < 1:
        raise ValueError("n_iters must be a positive integer")

    n = scores.shape[-1]

    # Selecting every item is an exact, trivial case.
    if k == n:
        return torch.ones_like(scores)

    # Data-dependent anchors: minimum and maximum score.
    lo = scores.amin(dim=-1, keepdim=True)
    hi = scores.amax(dim=-1, keepdim=True)

    anchors = torch.stack([lo, hi], dim=-1)

    # Each score is compared with the two anchors.
    log_kernel = -(
        (scores.unsqueeze(-1) - anchors) ** 2
    ) / temperature

    column_mass = torch.tensor(
        [n - k, k],
        device=scores.device,
        dtype=scores.dtype,
    )

    plan = sinkhorn_torch(
        log_kernel,
        n_iters=n_iters,
        row_mass=1.0,
        column_mass=column_mass,
    )

    # The second column represents selected mass.
    return plan[..., 1]
def soft_topk_fixed(scores, k, temperature=0.1, n_iters=50):
    """
    Differentiable top-k relaxation using fixed anchors.

    Scores are normalized to [0, 1], and compared with fixed
    unselected and selected anchors at 0 and 1.

    This is a fixed-anchor reference variant, not a claim of
    exact reproduction of the original SOFT top-k algorithm.
    """
    _validate_scores(scores, k)

    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")

    if not isinstance(n_iters, int) or n_iters < 1:
        raise ValueError("n_iters must be a positive integer")

    n = scores.shape[-1]

    if k == n:
        return torch.ones_like(scores)

    # Normalize scores independently for each batch element.
    lo = scores.amin(dim=-1, keepdim=True)
    hi = scores.amax(dim=-1, keepdim=True)
    span = hi - lo

    # For equal scores, use 0.5 for every item.
    # This produces uniform selection weights k/n.
    normalized = torch.where(
        span > 0,
        (scores - lo) / span.clamp_min(
            torch.finfo(scores.dtype).eps
        ),
        torch.full_like(scores, 0.5),
    )

    # Fixed anchors: 0 = unselected, 1 = selected.
    anchors = torch.tensor(
        [0.0, 1.0],
        device=scores.device,
        dtype=scores.dtype,
    )

    log_kernel = -(
        (normalized.unsqueeze(-1) - anchors) ** 2
    ) / temperature

    column_mass = torch.tensor(
        [n - k, k],
        device=scores.device,
        dtype=scores.dtype,
    )

    plan = sinkhorn_torch(
        log_kernel,
        n_iters=n_iters,
        row_mass=1.0,
        column_mass=column_mass,
    )

    return plan[..., 1]

def hard_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Return a non-differentiable binary top-k mask."""
    _validate_scores(scores, k)
    indices = scores.topk(k, dim=-1).indices
    return torch.zeros_like(scores).scatter(-1, indices, 1.0)

def random_selection(scores: torch.Tensor, k: int,
                     *, generator: torch.Generator | None = None) -> torch.Tensor:
    """Return a random binary k-selection mask."""
    _validate_scores(scores, k)
    noise = torch.rand(scores.shape, device=scores.device, generator=generator)
    return torch.zeros_like(scores).scatter(-1, noise.topk(k, dim=-1).indices, 1.0)

def softmax_attention(scores: torch.Tensor, k: int) -> torch.Tensor:
    """
    Softmax attention baseline.

    Scales attention weights by k so their total mass is k.
    Individual weights are not constrained to be <= 1.
    """
    _validate_scores(scores, k)
    return torch.softmax(scores, dim=-1) * k


def uniform_selection(scores: torch.Tensor, k: int) -> torch.Tensor:
    """
    Uniform all-patch pooling baseline.

    Every patch receives weight k / n, so total weight is k.
    """
    _validate_scores(scores, k)
    n = scores.shape[-1]
    return torch.full_like(scores, k / n)

def straight_through_topk(
    scores: torch.Tensor,
    k: int,
    temperature: float = 0.1,
    n_iters: int = 50,
) -> torch.Tensor:
    """
    Hard top-k in the forward pass, soft top-k gradients
    in the backward pass.

    This is a biased straight-through gradient estimator.
    """
    hard = hard_topk(scores, k)

    soft = soft_topk(
        scores,
        k,
        temperature=temperature,
        n_iters=n_iters,
    )

    return hard.detach() - soft.detach() + soft



def select_weights(
    scores: torch.Tensor,
    k: int,
    *,
    mode: Literal[
        "soft",
        "soft_fixed",
        "hard",
        "random",
        "attention",
        "uniform",
        "straight_through",
    ] = "soft",
    temperature: float = 0.1,
    n_iters: int = 50,
) -> torch.Tensor:

    if mode == "soft":
        return soft_topk(
            scores, k,
            temperature=temperature,
            n_iters=n_iters,
        )

    if mode == "soft_fixed":
        return soft_topk_fixed(
            scores, k,
            temperature=temperature,
            n_iters=n_iters,
        )

    if mode == "hard":
        return hard_topk(scores, k)

    if mode == "random":
        return random_selection(scores, k)

    if mode == "attention":
        return softmax_attention(scores, k)

    if mode == "uniform":
        return uniform_selection(scores, k)

    if mode == "straight_through":
        return straight_through_topk(
            scores, k,
            temperature=temperature,
            n_iters=n_iters,
        )

    raise ValueError(f"unknown selection mode: {mode}")


