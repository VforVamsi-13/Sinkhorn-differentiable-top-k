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

# def sinkhorn_torch(matrix, *, n_iters=50, epsilon=1e-8, log_domain=True):
#     """Return a differentiable tensor normalized with alternating row/column scaling.

#     ``matrix`` must contain nonnegative or log-domain scores.
#     For square inputs, each row and column sums to 1.
#     For rectangular inputs of shape [..., rows, cols], each row has mass 1.0 and
#     each column has mass ``rows / cols``.
#     """
#     try:
#         import torch
#     except ImportError as exc:
#         raise ImportError(
#             "sinkhorn_torch requires PyTorch; install the project torch extra"
#         ) from exc

#     if matrix.ndim < 2:
#         raise ValueError("matrix must have at least two dimensions")
#     if n_iters < 1 or epsilon <= 0:
#         raise ValueError("n_iters must be positive and epsilon must be positive")
#     if not torch.is_floating_point(matrix):
#         matrix = matrix.float()
#     if not torch.isfinite(matrix).all():
#         raise ValueError("matrix must contain only finite values")

#     if log_domain:
#         log_values = matrix - matrix.amax(dim=(-2, -1), keepdim=True)
#         log_values = log_values.clamp_min(torch.log(torch.as_tensor(epsilon,
#                                                                       device=matrix.device,
#                                                                       dtype=matrix.dtype)))
#         values = log_values.exp()
#     else:
#         values = matrix.clamp_min(epsilon)

#     rows, columns = values.shape[-2:]
#     row_mass = 1.0
#     column_mass = rows / columns
#     for _ in range(n_iters):
#         values = values / values.sum(dim=-1, keepdim=True).clamp_min(epsilon)
#         values = values * row_mass
#         values = values / values.sum(dim=-2, keepdim=True).clamp_min(epsilon)
#         values = values * column_mass
#     return values


def _prepare_marginal(mass, batch_shape, size, *, device, dtype, name):
    """Convert a marginal to shape [..., size] and validate it."""
    import torch

    mass = torch.as_tensor(mass, device=device, dtype=dtype)

    try:
        mass = torch.broadcast_to(mass, (*batch_shape, size))
    except RuntimeError as exc:
        raise ValueError(
            f"{name} must be broadcastable to shape "
            f"{(*batch_shape, size)}"
        ) from exc

    if not torch.isfinite(mass).all():
        raise ValueError(f"{name} must contain only finite values")

    if (mass < 0).any():
        raise ValueError(f"{name} must be nonnegative")

    return mass


def marginal_errors(plan, row_mass=None, column_mass=None):
    """Return maximum absolute row and column marginal errors."""
    import torch

    if plan.ndim < 2:
        raise ValueError("plan must have at least two dimensions")

    rows, columns = plan.shape[-2:]
    batch_shape = plan.shape[:-2]

    if row_mass is None:
        row_mass = 1.0

    if column_mass is None:
        column_mass = rows / columns

    r = _prepare_marginal(
        row_mass, batch_shape, rows,
        device=plan.device, dtype=plan.dtype, name="row_mass"
    )
    c = _prepare_marginal(
        column_mass, batch_shape, columns,
        device=plan.device, dtype=plan.dtype, name="column_mass"
    )

    row_error = (plan.sum(dim=-1) - r).abs().amax()
    column_error = (plan.sum(dim=-2) - c).abs().amax()

    return row_error, column_error


def sinkhorn_torch(
    matrix,
    *,
    n_iters=50,
    epsilon=1e-8,
    log_domain=True,
    row_mass=None,
    column_mass=None,
):
    """Compute an entropy-regularized transport plan using Sinkhorn scaling.

    Args:
        matrix:
            Log-kernel if log_domain=True; strictly positive kernel otherwise.
            Shape [..., rows, columns].
        n_iters:
            Number of alternating normalization iterations.
        epsilon:
            Absolute tolerance used when checking total marginal mass.
        log_domain:
            Whether matrix contains log-kernel values.
        row_mass:
            Desired row sums. Defaults to 1 for every row.
        column_mass:
            Desired column sums. Defaults to rows / columns.

    Returns:
        Transport plan with the requested marginals up to numerical
        convergence error.
    """
    import torch

    if matrix.ndim < 2:
        raise ValueError("matrix must have at least two dimensions")

    if n_iters < 1:
        raise ValueError("n_iters must be at least one")

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    if not torch.is_floating_point(matrix):
        matrix = matrix.float()

    if not torch.isfinite(matrix).all():
        raise ValueError("matrix must contain only finite values")

    rows, columns = matrix.shape[-2:]
    batch_shape = matrix.shape[:-2]

    if row_mass is None:
        row_mass = 1.0

    if column_mass is None:
        column_mass = rows / columns

    r = _prepare_marginal(
        row_mass, batch_shape, rows,
        device=matrix.device, dtype=matrix.dtype, name="row_mass"
    )
    c = _prepare_marginal(
        column_mass, batch_shape, columns,
        device=matrix.device, dtype=matrix.dtype, name="column_mass"
    )

    # Both marginals must describe the same total transported mass.
    total_r = r.sum(dim=-1)
    total_c = c.sum(dim=-1)

    if not torch.allclose(
        total_r, total_c, atol=epsilon, rtol=1e-5
    ):
        raise ValueError(
            "row_mass and column_mass must have equal total mass"
        )

    if (total_r <= 0).any():
        raise ValueError("total marginal mass must be positive")

    if log_domain:
        log_kernel = matrix
    else:
        if (matrix <= 0).any():
            raise ValueError(
                "matrix must be strictly positive when log_domain=False"
            )
        log_kernel = matrix.log()

    # Represent zero marginal entries as -inf in log space.
    # This preserves exact zeros instead of replacing them with epsilon.
    log_r = torch.where(
        r > 0, r.log(), torch.full_like(r, -torch.inf)
    ).unsqueeze(-1)

    log_c = torch.where(
        c > 0, c.log(), torch.full_like(c, -torch.inf)
    ).unsqueeze(-2)

    log_u = torch.zeros_like(log_r)
    log_v = torch.zeros_like(log_c)

    # Alternating updates of row and column scaling potentials.
    for _ in range(n_iters):
        row_lse = torch.logsumexp(
            log_kernel + log_v, dim=-1, keepdim=True
        )
        log_u = torch.where(
            r.unsqueeze(-1) > 0,
            log_r - row_lse,
            torch.full_like(log_r, -torch.inf),
        )

        column_lse = torch.logsumexp(
            log_kernel + log_u, dim=-2, keepdim=True
        )
        log_v = torch.where(
            c.unsqueeze(-2) > 0,
            log_c - column_lse,
            torch.full_like(log_c, -torch.inf),
        )

    return (log_kernel + log_u + log_v).exp()
