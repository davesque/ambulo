import pprint

from .utils import (
    dot,
    flatten,
    get_idx_multipliers,
    get_seq_dims,
    product,
    str_to_lst,
    unflatten,
)


class TensorError(Exception):
    pass


class Tensor:
    def __init__(self, spec, shape=None):
        if isinstance(spec, str):
            lst = str_to_lst(spec)
        else:
            lst = spec

        self._lst = flatten(lst)

        if shape is None:
            try:
                shape = tuple(get_seq_dims(lst))
            except ValueError as e:
                raise TensorError(e.args[0]) from e

        self.reshape(*shape)

    @property
    def rank(self):
        return len(self._shape)

    @property
    def shape(self):
        return self._shape

    def reshape(self, *shape):
        if len(self._lst) != product(shape):
            raise TensorError('Tensor cannot be fit into given shape')

        self._shape = shape
        self._idx_mul = get_idx_multipliers(shape)

    def _rearrange(self, indices, all_idx, result):
        curr_idx, rest_indices = indices[0], indices[1:]
        curr_shp = self._shape[curr_idx]

        if len(rest_indices) == 0:
            # Terminal case.  This is the last re-ordered index.
            for i in range(curr_shp):
                # Iterate through possible values for this index and incoporate
                # them into the current compound index
                all_idx[curr_idx] = i
                result.append(self[all_idx])
        else:
            # General case.  There are more re-ordered indices to traverse.
            for i in range(curr_shp):
                # Iterate through possible values for this index and incoporate
                # them into the current compound index
                all_idx[curr_idx] = i
                self._rearrange(rest_indices, all_idx, result)

    def rearrange(self, *indices):
        """
        Return a new tensor with rearranged indices.  Indices are identified by
        their position in the tensor's shape tuple e.g. for a tensor of rank 2,
        ``0`` refers to the first index and ``1`` to the second.  This methods
        expects for all indices in the tensor to be identified in some order.
        For example, to take the transpose of a rank 2 tensor::

            >>> tensor.rearrange(1, 0)

        This will return a new tensor with the second index swapped with the
        first.
        """
        rank = self.rank
        expected = tuple(range(rank))

        if set(indices) != set(expected) or len(indices) != rank:
            raise TensorError(
                f'Expected exactly the following indices in some order: {expected}',
            )

        new_lst = []
        self._rearrange(
            indices,
            [0 for _ in self._shape],
            new_lst,
        )

        new_shape = [self._shape[i] for i in indices]
        return type(self)(new_lst, new_shape)

    @property
    def m(self):
        return self._shape[0]

    @property
    def n(self):
        return self._shape[-1]

    def __getitem__(self, key):
        return self._lst[dot(key, self._idx_mul)]

    def __setitem__(self, key, value):
        self._lst[dot(key, self._idx_mul)] = value

    def __iter__(self):
        return iter(self._lst)

    def __eq__(self, other):
        if self._shape != other._shape:
            return False

        for x, y in zip(self, other):
            if x != y:
                return False

        return True

    def __mul__(self, other):
        return type(self)([other * x for x in self], self._shape)

    __rmul__ = __mul__

    def __add__(self, other):
        if self._shape != other._shape:
            raise TensorError('Tensors must have same shape')

        return type(self)([x + y for x, y in zip(self, other)], self._shape)

    def __sub__(self, other):
        return self + -1 * other

    @property
    def T(self):
        return self.rearrange(*reversed(range(self.rank)))

    def __matmul__(self, other):
        if self._shape[-1] != other._shape[0]:
            raise TensorError('Tensors must have compatible shape')

        return type(self)([
            [
                sum(
                    self[i, j] * other[j, k]
                    for j in range(self.n)
                )
                for k in range(other.n)
            ]
            for i in range(self.m)
        ])

    def to_list(self):
        return unflatten(self._lst, self._idx_mul[:-1])

    def __str__(self):
        return pprint.pformat(self.to_list())

    def __repr__(self):
        return str(self)
