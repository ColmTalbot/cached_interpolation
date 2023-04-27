from functools import lru_cache

import numpy as np


def _tridiagonal(n_nodes, diagonal, off_diagonal):
    above = np.arange(1, n_nodes)
    below = np.arange(n_nodes - 1)
    output = np.zeros(shape=(n_nodes, n_nodes))
    np.fill_diagonal(output, diagonal)
    output[above, below] = off_diagonal
    output[below, above] = off_diagonal
    return output


def _apply_boundary(matrix, boundary):
    matrix[0] = boundary
    matrix[-1] = boundary[::-1]


def basic(n_nodes, bound_1=None, bound_2=None):
    tmp1 = _tridiagonal(n_nodes=n_nodes, diagonal=2 / 3, off_diagonal=1 / 6)
    if bound_1 is not None:
        _apply_boundary(matrix=tmp1, boundary=bound_1)

    tmp2 = _tridiagonal(n_nodes=n_nodes, diagonal=-2, off_diagonal=1)
    if bound_2 is not None:
        _apply_boundary(matrix=tmp2, boundary=bound_2)

    return np.linalg.solve(tmp1, tmp2)


@lru_cache
def periodic(n_nodes):
    bound_1 = np.zeros(n_nodes)
    bound_1[0] = 2 / 3
    bound_1[1] = 1 / 6
    bound_1[-2] = 1 / 6
    bound_2 = np.zeros(n_nodes)
    bound_2[0] = -2
    bound_2[1] = 1
    bound_2[-2] = 1
    return basic(n_nodes=n_nodes, bound_1=bound_1, bound_2=bound_2)


@lru_cache
def clamped(n_nodes):
    bound_1 = np.zeros(n_nodes)
    bound_1[:2] = np.array([2, 1]) / 6
    bound_2 = np.zeros(n_nodes)
    bound_2[:2] = np.array([-1, 1])
    return basic(n_nodes=n_nodes, bound_1=bound_1, bound_2=bound_2)


@lru_cache
def natural(n_nodes):
    bound_1 = np.zeros(n_nodes)
    bound_1[0] = 2 / 3
    bound_2 = np.zeros(n_nodes)
    return basic(n_nodes=n_nodes, bound_1=bound_1, bound_2=bound_2)


@lru_cache
def not_a_knot(n_nodes):
    bound_1 = np.zeros(n_nodes)
    bound_1[:3] = np.array([-1, 2, -1])
    bound_2 = np.zeros(n_nodes)
    return basic(n_nodes=n_nodes, bound_1=bound_1, bound_2=bound_2)


MAPPING = {
    "clamped": clamped,
    "natural": natural,
    "not-a-knot": not_a_knot,
    "periodic": periodic,
}
