import pprint

from .utils import (
    dot,
    flatten,
    get_idx_multipliers,
    get_seq_dims,
    product,
    unflatten,
)


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
        return self.dims[-1]

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
