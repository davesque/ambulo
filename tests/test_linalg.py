import pytest

from ambulo.linalg import (
    Tensor,
    TensorError,
)


@pytest.fixture
def A():
    return Tensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
    ])


@pytest.fixture
def B():
    return Tensor([
        [1, 2],
        [3, 4],
        [5, 6],
    ])


class TestTensor:
    def test_tensor_init_sets_dimension_properties(self, A, B):
        assert A.m == 3
        assert A.n == 4

        assert B.m == 3
        assert B.n == 2

    @pytest.mark.parametrize(
        'tensor, expected',
        (
            (Tensor([1]), 1),
            (Tensor([1, 1]), 1),
            (Tensor([[1], [1]]), 2),
            (Tensor([[1, 1], [1, 1]]), 2),
            (Tensor([[[1], [1]], [[1], [1]]]), 3),
            (Tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]), 3),
        ),
    )
    def test_rank(self, tensor, expected):
        assert tensor.rank == expected

    @pytest.mark.parametrize(
        'tensor, expected',
        (
            (Tensor([1]), (1,)),
            (Tensor([1, 1]), (2,)),
            (Tensor([[1], [1]]), (2, 1)),
            (Tensor([[1, 1], [1, 1]]), (2, 2)),
            (Tensor([[[1], [1]], [[1], [1]]]), (2, 2, 1)),
            (Tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]), (2, 2, 2)),
        ),
    )
    def test_shape(self, tensor, expected):
        assert tensor.shape == expected

    @pytest.mark.parametrize(
        'tensor, new_shape, expected',
        (
            (
                Tensor([1, 1]),
                (2, 1),
                Tensor([[1], [1]]),
            ),
            (
                Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]),
                (3, 3),
                Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            ),
            (
                Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
                (4, 3),
                Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
            ),
            (
                Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
                (2, 6),
                Tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]),
            ),
        ),
    )
    def test_reshape(self, tensor, new_shape, expected):
        assert tensor.reshape(*new_shape) == expected

    @pytest.mark.parametrize(
        'tensor, new_shape',
        (
            (
                Tensor([1, 1]),
                (1,),
            ),
            (
                Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]),
                (3, 4),
            ),
            (
                Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
                (4, 3, 2),
            ),
            (
                Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
                (2, 7),
            ),
        ),
    )
    def test_reshape_raises_tensor_error(self, tensor, new_shape):
        with pytest.raises(TensorError):
            tensor.reshape(*new_shape)

    def test_tensor_init_raises_value_error(self):
        with pytest.raises(TensorError):
            Tensor([
                [1, 2, 3],
                [1, 2, 3, 4],
            ])

    def test_tensor_getitem(self, A, B):
        assert A[0, 0] == 1
        assert A[0, 1] == 2

        assert B[0, 0] == 1
        assert B[1, 0] == 3

    def test_tensor_setitem(self, A):
        assert A[0, 0] == 1
        A[0, 0] = 2
        assert A[0, 0] == 2

    def test_tensor_iter(self, A):
        i = 1
        for x in A:
            assert x == i
            i += 1

    def test_tensor_eq(self, A, B):
        assert A == Tensor([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ])

        assert A != Tensor([
            [1, 2, 4, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ])

        assert A != B

    def test_tensor_scalar_mul(self, B):
        assert 3 * B == Tensor([
            [3, 6],
            [9, 12],
            [15, 18],
        ])

        assert B * 3 == Tensor([
            [3, 6],
            [9, 12],
            [15, 18],
        ])

    def test_tensor_tensor_add_raises_value_error(self, A, B):
        with pytest.raises(TensorError):
            A + B

    def test_tensor_tensor_add(self, B):
        assert B + B == Tensor([
            [2, 4],
            [6, 8],
            [10, 12],
        ])

        assert B + Tensor([
            [1, 1],
            [1, 1],
            [1, 1],
        ]) == Tensor([
            [2, 3],
            [4, 5],
            [6, 7],
        ])

    def test_tensor_tensor_sub(self, B):
        assert B - Tensor([
            [1, 1],
            [1, 1],
            [1, 1],
        ]) == Tensor([
            [0, 1],
            [2, 3],
            [4, 5],
        ])

    def test_tensor_T(self, A):
        assert A.T == Tensor([
            [1, 5, 9],
            [2, 6, 10],
            [3, 7, 11],
            [4, 8, 12],
        ])

    def test_tensor_tensor_mul_raises_value_error(self, A, B):
        with pytest.raises(TensorError):
            A @ B

    def test_tensor_tensor_mul(self, A, B):
        assert A.T @ B == Tensor([
            [61, 76],
            [70, 88],
            [79, 100],
            [88, 112],
        ])
