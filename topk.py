"""Differentiable and baseline top-k selection utilities."""
from __future__ import annotations
from typing import Literal
import torch
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

def soft_topk(scores, k, *, temperature=0.1, n_iters=50):
    _validate_scores(scores, k)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    n = scores.shape[-1]
    lo = scores.amin(-1, keepdim=True)
    hi = scores.amax(-1, keepdim=True)
    anchors = torch.stack([lo, hi], dim=-1)                       # (..., 1, 2)
    log_kernel = -((scores.unsqueeze(-1) - anchors) ** 2) / temperature   # (..., n, 2)

    col_mass = torch.tensor([n - k, k], device=scores.device, dtype=scores.dtype)
    plan = sinkhorn_torch(log_kernel, n_iters=n_iters,
                          row_mass=1.0, column_mass=col_mass)
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

def select_weights(scores: torch.Tensor, k: int, *,
                   mode: Literal["soft", "hard", "random"] = "soft",
                   temperature: float = 0.1, n_iters: int = 50) -> torch.Tensor:
    if mode == "soft":
        return soft_topk(scores, k, temperature=temperature, n_iters=n_iters)
    if mode == "hard":
        return hard_topk(scores, k)
    if mode == "random":
        return random_selection(scores, k)
    raise ValueError(f"unknown selection mode: {mode}")
