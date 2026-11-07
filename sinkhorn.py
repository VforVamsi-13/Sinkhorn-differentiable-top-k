"""Numerically stable Sinkhorn normalization for NumPy and PyTorch."""

import numpy as np


def sinkhorn(A, tol=1e-8, max_iter=10000):
    """Alternately normalize a nonnegative square matrix.

    Returns an approximately doubly stochastic matrix and the number of
    normalization iterations performed.
    """
    if tol <= 0 or max_iter < 1:
        raise ValueError("tol must be positive and max_iter must be at least one")

    A = np.asarray(A, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")

    if not np.all(np.isfinite(A)) or np.any(A < 0):
        raise ValueError("A must be finite and nonnegative")

    if np.any(A.sum(axis=0) == 0) or np.any(A.sum(axis=1) == 0):
        raise ValueError("A must not contain zero rows or columns")

    B = A.copy()

    for iteration in range(1, max_iter + 1):
        B /= B.sum(axis=1, keepdims=True)
        B /= B.sum(axis=0, keepdims=True)

        row_error = np.max(np.abs(B.sum(axis=1) - 1))
        col_error = np.max(np.abs(B.sum(axis=0) - 1))

        if max(row_error, col_error) < tol:
            return B, iteration

    raise RuntimeError(
        "Sinkhorn iteration did not reach tolerance within max_iter"
    )


def sinkhorn_torch(matrix, *, n_iters=50, epsilon=1e-8, log_domain=True):
    """Return a differentiable tensor with compatible transport marginals.

    ``matrix`` may contain arbitrary finite real scores. Scores are converted
    to positive weights with a log-domain normalization, which avoids
    underflow for the small temperatures commonly used by SOFT top-k. For a
    square input both marginals are one; for a rectangular input each row has
    mass one and each column has mass ``rows / columns``.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "sinkhorn_torch requires PyTorch; install the project torch extra"
        ) from exc

    if matrix.ndim < 2:
        raise ValueError("matrix must have at least two dimensions")
    if n_iters < 1 or epsilon <= 0:
        raise ValueError("n_iters must be positive and epsilon must be positive")
    if not torch.is_floating_point(matrix):
        matrix = matrix.float()
    if not torch.isfinite(matrix).all():
        raise ValueError("matrix must contain only finite values")

    if log_domain:
        log_values = matrix - matrix.amax(dim=(-2, -1), keepdim=True)
        log_values = log_values.clamp_min(torch.log(torch.as_tensor(epsilon,
                                                                      device=matrix.device,
                                                                      dtype=matrix.dtype)))
        values = log_values.exp()
    else:
        values = matrix.clamp_min(epsilon)

    rows, columns = values.shape[-2:]
    row_mass = 1.0
    column_mass = rows / columns
    for _ in range(n_iters):
        values = values / values.sum(dim=-1, keepdim=True).clamp_min(epsilon)
        values = values * row_mass
        values = values / values.sum(dim=-2, keepdim=True).clamp_min(epsilon)
        values = values * column_mass
    return values
