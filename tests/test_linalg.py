import pytest

from ambulo.linalg import (
    Matrix,
)


@pytest.fixture
def A():
    return Matrix([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
    ])


@pytest.fixture
def B():
    return Matrix([
        [1, 2],
        [3, 4],
        [5, 6],
    ])


def test_matrix_init_sets_dimension_properties(A, B):
    assert A.m == 3
    assert A.n == 4

    assert B.m == 3
    assert B.n == 2


def test_matrix_init_raises_value_error():
    with pytest.raises(ValueError):
        Matrix([
            [1, 2, 3],
            [1, 2, 3, 4],
        ])


def test_matrix_getitem(A, B):
    assert A[0, 0] == 1
    assert A[0, 1] == 2

    assert B[0, 0] == 1
    assert B[1, 0] == 3


def test_matrix_setitem(A):
    assert A[0, 0] == 1
    A[0, 0] = 2
    assert A[0, 0] == 2


def test_matrix_iter(A):
    for i, row in enumerate(A):
        assert row == list(range(i * 4 + 1, (i + 1) * 4 + 1))


def test_matrix_eq(A):
    assert A == Matrix([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
    ])

    assert A != Matrix([
        [1, 2, 4, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
    ])


def test_matrix_scalar_mul(B):
    assert 3 * B == Matrix([
        [3, 6],
        [9, 12],
        [15, 18],
    ])

    assert B * 3 == Matrix([
        [3, 6],
        [9, 12],
        [15, 18],
    ])


def test_matrix_matrix_add_raises_value_error(A, B):
    with pytest.raises(ValueError):
        A + B


def test_matrix_matrix_mul(B):
    assert B + B == Matrix([
        [2, 4],
        [6, 8],
        [10, 12],
    ])

    assert B + Matrix([
        [1, 1],
        [1, 1],
        [1, 1],
    ]) == Matrix([
        [2, 3],
        [4, 5],
        [6, 7],
    ])


def test_matrix_matrix_sub(B):
    assert B - Matrix([
        [1, 1],
        [1, 1],
        [1, 1],
    ]) == Matrix([
        [0, 1],
        [2, 3],
        [4, 5],
    ])


def test_matrix_T(A):
    assert A.T == Matrix([
        [1, 5, 9],
        [2, 6, 10],
        [3, 7, 11],
        [4, 8, 12],
    ])
