import numpy as np


def sinkhorn(A, tol=1e-8, max_iter=10000):
    """Alternately normalize a nonnegative square matrix.

    Returns an approximately doubly stochastic matrix and the number of
    normalization iterations performed.
    """
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
