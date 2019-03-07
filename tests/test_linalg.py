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

    def test_tensor_init_raises_value_error(self):
        with pytest.raises(TensorError):
            Tensor([
                [1, 2, 3],
                [1, 2, 3, 4],
            ])

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
    def test_order(self, tensor, expected):
        assert tensor.order == expected

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

    @pytest.mark.parametrize(
        'tensor, new_order, expected',
        (
            (
                Tensor([[1, 2]]),
                (1, 0),
                Tensor([[1], [2]]),
            ),
            (
                Tensor(list(range(1, 7))).reshape(2, 3),
                (1, 0),
                Tensor([
                    [1, 4],
                    [2, 5],
                    [3, 6],
                ]),
            ),
            (
                Tensor(list(range(1, 25))).reshape(4, 3, 2),
                (1, 2, 0),
                Tensor([
                    [[1, 7, 13, 19], [2, 8, 14, 20]],
                    [[3, 9, 15, 21], [4, 10, 16, 22]],
                    [[5, 11, 17, 23], [6, 12, 18, 24]],
                ]),
            ),
            (
                Tensor(list(range(1, 25))).reshape(4, 3, 2),
                (2, 0, 1),
                Tensor([
                    [[1, 3, 5], [7, 9, 11], [13, 15, 17], [19, 21, 23]],
                    [[2, 4, 6], [8, 10, 12], [14, 16, 18], [20, 22, 24]],
                ]),
            ),
            (
                Tensor(list(range(1, 25))).reshape(4, 3, 2),
                (2, 1, 0),
                Tensor([
                    [[1, 7, 13, 19], [3, 9, 15, 21], [5, 11, 17, 23]],
                    [[2, 8, 14, 20], [4, 10, 16, 22], [6, 12, 18, 24]],
                ]),
            ),
        ),
    )
    def test_rearrange(self, tensor, new_order, expected):
        assert tensor.rearrange(*new_order) == expected

    @pytest.mark.parametrize(
        'tensor, new_order',
        (
            (
                Tensor([[1, 2]]),
                (1, 1),
            ),
            (
                Tensor(list(range(1, 7))).reshape(2, 3),
                (2, 1, 0),
            ),
            (
                Tensor(list(range(1, 25))).reshape(4, 3, 2),
                (1, 1, 0),
            ),
            (
                Tensor(list(range(1, 25))).reshape(4, 3, 2),
                (2, 0, 1, 5),
            ),
            (
                Tensor(list(range(1, 25))).reshape(4, 3, 2),
                (2,),
            ),
        ),
    )
    def test_rearrange_raises_tensor_error(self, tensor, new_order):
        with pytest.raises(TensorError):
            tensor.rearrange(*new_order)

    @pytest.mark.parametrize(
        'tensor, expected',
        (
            (Tensor([[1, 2]]), 1),
            (Tensor(list(range(1, 7))).reshape(2, 3), 2),
            (Tensor(list(range(1, 25))).reshape(4, 3, 2), 4),
            (Tensor(list(range(1, 25))).reshape(3, 2, 4), 3),
            (Tensor(list(range(1, 25))).reshape(2, 4, 3), 2),
        ),
    )
    def test_m(self, tensor, expected):
        assert tensor.m == expected

    @pytest.mark.parametrize(
        'tensor, expected',
        (
            (Tensor([[1, 2]]), 2),
            (Tensor(list(range(1, 7))).reshape(2, 3), 3),
            (Tensor(list(range(1, 25))).reshape(4, 3, 2), 2),
            (Tensor(list(range(1, 25))).reshape(3, 2, 4), 4),
            (Tensor(list(range(1, 25))).reshape(2, 4, 3), 3),
        ),
    )
    def test_n(self, tensor, expected):
        assert tensor.n == expected

    @pytest.mark.parametrize(
        'tensor, idx, expected',
        (
            (Tensor([[1, 2]]), (0, 0), 1),
            (Tensor([[1, 2]]), (0, 1), 2),
            (Tensor([1, 2]), (0,), 1),
            (Tensor([1, 2]), (1,), 2),
            (Tensor(list(range(1, 25))).reshape(4, 3, 2), (2, 1, 1), 16),
        ),
    )
    def test_getitem(self, tensor, idx, expected):
        assert tensor[idx] == expected

    def test_setitem(self):
        A = Tensor([0 for _ in range(24)]).reshape(4, 3, 2)
        A[2, 1, 1] = 16
        assert A[2, 1, 1] == 16

    def test_iter(self):
        A = Tensor(list(range(24)))

        assert list(A.reshape(4, 3, 2)) == list(range(24))
        assert list(A.reshape(2, 3, 4)) == list(range(24))
        assert list(A.reshape(1, 1, 24)) == list(range(24))

    def test_eq(self, A, B):
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

    def test_scalar_mul(self, B):
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

    def test_tensor_tensor_add_raises_value_error(self, A, B):
        with pytest.raises(TensorError):
            A + B

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

    def test_matmul(self, A, B):
        assert A.T @ B == Tensor([
            [61, 76],
            [70, 88],
            [79, 100],
            [88, 112],
        ])

    def test_matmul_raises_value_error(self, A, B):
        with pytest.raises(TensorError):
            A @ B

    @pytest.mark.parametrize(
        'tensor, expected',
        (
            (
                Tensor([[1, 2]]),
                [[1, 2]],
            ),
            (
                Tensor(list(range(1, 7))).reshape(2, 3),
                [
                    [1, 2, 3],
                    [4, 5, 6],
                ],
            ),
            (
                Tensor(list(range(1, 25))).reshape(4, 3, 2).rearrange(1, 2, 0),
                [
                    [[1, 7, 13, 19], [2, 8, 14, 20]],
                    [[3, 9, 15, 21], [4, 10, 16, 22]],
                    [[5, 11, 17, 23], [6, 12, 18, 24]],
                ],
            ),
            (
                Tensor(list(range(1, 25))).reshape(4, 3, 2).rearrange(2, 0, 1),
                [
                    [[1, 3, 5], [7, 9, 11], [13, 15, 17], [19, 21, 23]],
                    [[2, 4, 6], [8, 10, 12], [14, 16, 18], [20, 22, 24]],
                ],
            ),
            (
                Tensor(list(range(1, 25))).reshape(4, 3, 2).rearrange(2, 1, 0),
                [
                    [[1, 7, 13, 19], [3, 9, 15, 21], [5, 11, 17, 23]],
                    [[2, 8, 14, 20], [4, 10, 16, 22], [6, 12, 18, 24]],
                ],
            ),
        ),
    )
    def test_tolist(self, tensor, expected):
        assert tensor.tolist() == expected

    @pytest.mark.parametrize(
        'tensor, expected',
        (
            (
                Tensor([[1, 2]]),
                '[[1, 2]]',
            ),
            (
                Tensor(list(range(1, 7))).reshape(2, 3),
                '[[1, 2, 3], [4, 5, 6]]',
            ),
            (
                Tensor(list(range(1, 25))).reshape(4, 3, 2).rearrange(1, 2, 0),
                '''
[[[1, 7, 13, 19], [2, 8, 14, 20]],
 [[3, 9, 15, 21], [4, 10, 16, 22]],
 [[5, 11, 17, 23], [6, 12, 18, 24]]]
'''[1:-1],
            ),
            (
                Tensor(list(range(1, 25))).reshape(4, 3, 2).rearrange(2, 0, 1),
                '''
[[[1, 3, 5], [7, 9, 11], [13, 15, 17], [19, 21, 23]],
 [[2, 4, 6], [8, 10, 12], [14, 16, 18], [20, 22, 24]]]
'''[1:-1],
            ),
            (
                Tensor(list(range(1, 25))).reshape(4, 3, 2).rearrange(2, 1, 0),
                '''
[[[1, 7, 13, 19], [3, 9, 15, 21], [5, 11, 17, 23]],
 [[2, 8, 14, 20], [4, 10, 16, 22], [6, 12, 18, 24]]]
'''[1:-1],
            ),
        ),
    )
    def test_str(self, tensor, expected):
        assert str(tensor) == expected
        assert repr(tensor) == expected
