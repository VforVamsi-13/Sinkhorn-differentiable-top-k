
import numpy as np


def _validate_matrix(A):
    """Validate a finite, nonnegative square matrix."""
    A = np.asarray(A, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")

    if not np.all(np.isfinite(A)) or np.any(A < 0):
        raise ValueError("A must be finite and nonnegative")

    return A


def _has_perfect_matching(S):
    """Check whether the Boolean square matrix S has a
    perfect matching, using augmenting paths.
    """
    n = S.shape[0]
    matched_row = [-1] * n  # Row currently matched to each column.

    def augment(row, visited):
        for col in range(n):
            if not S[row, col] or visited[col]:
                continue

            visited[col] = True

            if (
                matched_row[col] == -1
                or augment(matched_row[col], visited)
            ):
                matched_row[col] = row
                return True

        return False

    for row in range(n):
        if not augment(row, [False] * n):
            return False

    return True


def has_support(A):
    """True iff A has at least one positive diagonal."""
    A = _validate_matrix(A)
    return _has_perfect_matching(A > 0)


def has_total_support(A):
    """True iff A is nonzero and every positive entry
    belongs to at least one positive diagonal.
    """
    A = _validate_matrix(A)
    n = A.shape[0]

    if not np.any(A > 0):
        return False

    S = A > 0

    for i, j in zip(*np.nonzero(S)):
        # Force entry (i, j) into the diagonal.
        # The remaining rows and columns must admit
        # a perfect matching.
        remaining = np.delete(
            np.delete(S, i, axis=0), j, axis=1
        )

        if not _has_perfect_matching(remaining):
            return False

    return True


def is_fully_indecomposable(A):
    """True iff A has total support and its bipartite
    support graph is connected.
    """
    A = _validate_matrix(A)
    n = A.shape[0]

    if not has_total_support(A):
        return False

    S = A > 0

    # Graph vertices: rows 0..n-1, columns n..2n-1.
    adjacency = [[] for _ in range(2 * n)]

    for i, j in zip(*np.nonzero(S)):
        col_vertex = n + j
        adjacency[i].append(col_vertex)
        adjacency[col_vertex].append(i)

    visited = {0}
    stack = [0]

    while stack:
        vertex = stack.pop()

        for neighbor in adjacency[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)

    return len(visited) == 2 * n
