import functools

import pytest

from ambulo.utils import (
    chunks,
    flatten,
    get_seq_dims,
    seq_has_dims,
    to_tuple,
    unflatten,
)


@pytest.mark.parametrize(
    'lst, n, expected',
    (
        (
            [1, 2, 3, 4, 5, 6], 3,
            [[1, 2, 3], [4, 5, 6]],
        ),
        (
            [1, 2, 3, 4, 5, 6], 4,
            [[1, 2, 3, 4], [5, 6]],
        ),
        (
            [1, 2, 3, 4, 5, 6], 6,
            [[1, 2, 3, 4, 5, 6]],
        ),
        (
            [1], 6,
            [[1]],
        ),
        (
            [], 6,
            [],
        ),
    ),
)
def test_chunks(lst, n, expected):
    assert list(chunks(lst, n)) == expected


def test_flatten():
    assert flatten([1, 2, [3, 4, [5, 6]]]) == [1, 2, 3, 4, 5, 6]

    heavily_nested = functools.reduce(lambda a, i: (a, i), range(1000))

    assert flatten(heavily_nested) == list(range(1000))


@pytest.mark.parametrize(
    'lst, multipliers, expected',
    (
        (
            list(range(3 ** 3)),
            (9, 3),
            [
                [[0, 1, 2],
                 [3, 4, 5],
                 [6, 7, 8]],
                [[9, 10, 11],
                 [12, 13, 14],
                 [15, 16, 17]],
                [[18, 19, 20],
                 [21, 22, 23],
                 [24, 25, 26]],
            ]
        ),
        (
            list(range(2 ** 4)),
            (8, 4, 2),
            [
                [[[0, 1], [2, 3]],
                 [[4, 5], [6, 7]]],
                [[[8, 9], [10, 11]],
                 [[12, 13], [14, 15]]],
            ],
        ),
    ),
)
def test_unflatten(lst, multipliers, expected):
    assert unflatten(lst, multipliers) == expected


def test_to_tuple():
    @to_tuple
    def my_gen(it):
        for i in it:
            yield i

    assert my_gen(range(100)) == tuple(range(100))
    assert my_gen('asdf') == tuple('asdf')


@pytest.mark.parametrize(
    'seq, expected',
    (
        ([1, 2, 3, 4], (4,)),
        ([[1, 2], [3, 4]], (2, 2)),
        ([[1, 2]], (1, 2)),
        ([[1], [2]], (2, 1)),
        (
            [
                [[0, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 0], [0, 0]],
            ],
            (4, 3, 2),
        ),
        ([], (0,)),
        ([[]], (1, 0)),
        (1, ()),
    ),
)
def test_get_seq_dims(seq, expected):
    assert get_seq_dims(seq) == expected


@pytest.mark.parametrize(
    'seq',
    (
        [[1, 2], [3]],
        [[1], []],
        [
            [[0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 0]],
            [[0, 0], [0], [0, 0]],
            [[0, 0], [0, 0], [0, 0]],
        ],
    ),
)
def test_get_seq_dims_raises_value_error(seq):
    with pytest.raises(ValueError):
        get_seq_dims(seq)


def test_seq_has_dims():
    assert seq_has_dims([
        [0, 0],
        [0, 0],
        [0, 0],
    ], (3, 2))

    assert seq_has_dims([
        [[0, 0], [0, 0], [0, 0]],
        [[0, 0], [0, 0], [0, 0]],
        [[0, 0], [0, 0], [0, 0]],
        [[0, 0], [0, 0], [0, 0]],
    ], (4, 3, 2))

    assert not seq_has_dims([
        [0, 0],
        [0, 0, 0],
        [0, 0],
    ], (3, 2))

    assert not seq_has_dims([
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
    ], (4, 3, 2))
