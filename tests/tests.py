#small tests currently


import numpy as np
import pytest

from sinkhorn import sinkhorn


def test_positive_matrix():
    A = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    B, iterations = sinkhorn(A)

    assert np.all(B >= 0)
    assert np.allclose(B.sum(axis=1), 1.0, atol=1e-8)
    assert np.allclose(B.sum(axis=0), 1.0, atol=1e-8)
    assert iterations > 0


def test_already_doubly_stochastic():
    A = np.array([
        [0.5, 0.5],
        [0.5, 0.5],
    ])

    B, _ = sinkhorn(A)

    assert np.allclose(B, A)


def test_diagonal_matrix_with_zeros():
    A = np.diag([2.0, 5.0, 3.0])

    B, _ = sinkhorn(A)

    assert np.allclose(B, np.eye(3))


def test_zero_row_is_rejected():
    A = np.array([
        [1.0, 2.0],
        [0.0, 0.0],
    ])

    with pytest.raises(ValueError):
        sinkhorn(A)


def test_input_is_not_modified():
    A = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])
    original = A.copy()

    sinkhorn(A)

    assert np.array_equal(A, original)
