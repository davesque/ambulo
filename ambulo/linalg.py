import functools
import operator
import pprint
from typing import (
    Iterable,
    Sequence,
)

from .utils import (
    flatten,
    get_seq_dims,
    to_tuple,
    unflatten,
)
from .types import (
    Number,
)


def product(seq: Sequence[Number]) -> Number:
    """
    Returns the product of all elements in ``seq``.
    """
    return functools.reduce(operator.mul, seq)


def dot(A: Iterable[Number], B: Iterable[Number]) -> Number:
    """
    Returns the dot product of the iterable vectors ``A`` and ``B``.
    """
    return sum(a * b for a, b in zip(A, B))


@to_tuple
def get_idx_multipliers(dims: Sequence[Number]) -> Iterable[Number]:
    """
    For dimensions with sizes given in ``dims``, returns the number of elements
    identified by walking down each dimension.

    For example, let dimensions ``(3, 3, 3, 3)`` represent the dimensions of a
    rank-4 tensor with 4 indices and 3 possible values for each index.
    Providing a value for the first index identifies 27 possible elements.
    Providing a value for the second index identifies 9 elements within those
    27.  Providing a value for the third index identifies 3 elements within
    those 9.  Providing the last index uniquely identifies a single element
    within those 3.  Thus, the resulting "index multipliers" are ``(27, 9, 3,
    1)``.
    """
    multiplier = product(dims)

    for d in dims:
        multiplier //= d
        yield multiplier


class Tensor:
    def __init__(self, lst, dims=None):
        self._lst = flatten(lst)

        if dims is not None:
            self.dims = tuple(dims)

            if len(lst) != product(dims):
                raise ValueError('Given sequence cannot be cast into given dimensions')
        else:
            self.dims = tuple(get_seq_dims(lst))

        self._idx_mul = get_idx_multipliers(self.dims)

    @property
    def m(self):
        return self.dims[0]

    @property
    def n(self):
        return self.dims[1]

    def __getitem__(self, key):
        return self._lst[dot(key, self._idx_mul)]

    def __setitem__(self, key, value):
        self._lst[dot(key, self._idx_mul)] = value

    def __iter__(self):
        return iter(self._lst)

    def __eq__(self, other):
        if self.dims != other.dims:
            return False

        for x, y in zip(self, other):
            if x != y:
                return False

        return True

    def __mul__(self, other):
        return type(self)([other * x for x in self], self.dims)

    __rmul__ = __mul__

    def __add__(self, other):
        if self.dims != other.dims:
            raise ValueError('Tensors must have same dimensions')

        return type(self)([x + y for x, y in zip(self, other)], self.dims)

    def __sub__(self, other):
        return self + -1 * other

    @property
    def T(self):
        return type(self)([
            [self[i, j] for i in range(self.m)]
            for j in range(self.n)
        ])

    def __matmul__(self, other):
        if self.dims[-1] != other.dims[0]:
            raise ValueError('Tensors must have compatible dimensions')

        return type(self)([
            [
                sum(self[i, j] * other[j, k] for j in range(self.n))
                for k in range(other.n)
            ]
            for i in range(self.m)
        ])

    def __str__(self):
        return pprint.pformat(
            unflatten(self._lst, self._idx_mul[:-1]),
        )

    def __repr__(self):
        return str(self)
