import pprint
from typing import (
    Any,
    Generic,
    Iterator,
    List,
)

from .types import (
    Dims,
    NumberT,
    RawTensor,
)
from .utils import (
    dot,
    flatten,
    get_idx_multipliers,
    get_seq_dims,
    product,
    unflatten,
)


class TensorError(Exception):
    pass


class Tensor(Generic[NumberT]):
    _lst: List[NumberT]
    _shape: Dims
    _idx_mul: Dims

    def __init__(self, lst: RawTensor[NumberT], shape: Dims = None):
        self._lst = flatten(lst)

        if shape is None:
            try:
                shape = tuple(get_seq_dims(lst))
            except ValueError as e:
                raise TensorError(e.args[0]) from e
        else:
            shape = tuple(shape)

        if len(self._lst) != product(shape):
            raise TensorError('Tensor cannot be fit into given shape')

        self._shape = shape
        self._idx_mul = get_idx_multipliers(shape)

    @property
    def order(self) -> int:
        """
        The number of indices of a tensor.
        """
        return len(self._shape)

    @property
    def shape(self) -> Dims:
        """
        The number of values that each of a tensor's indices can take.
        """
        return self._shape

    def reshape(self, *shape: int) -> 'Tensor[NumberT]':
        """
        Returns a new tensor with the given shape.
        """
        return type(self)(self._lst, shape)

    def _rearrange(self, indices: Dims, all_idx: List[int], result: List[NumberT]) -> None:
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

    def rearrange(self, *indices: int) -> 'Tensor[NumberT]':
        """
        Returns a new tensor with rearranged indices.  Indices are identified
        by their position in the tensor's shape tuple e.g. for a tensor of order
        2, ``0`` refers to the first index and ``1`` to the second.  This
        method expects for all indices in the tensor to be identified in some
        order.  For example, to take the transpose of an order 2 tensor::

            >>> tensor.rearrange(1, 0)

        This will return a new tensor with the second index swapped with the
        first.
        """
        order = self.order
        expected = tuple(range(order))

        if set(indices) != set(expected) or len(indices) != order:
            raise TensorError(
                f'Expected exactly the following indices in some order: {expected}',
            )

        new_lst: List[NumberT] = []
        self._rearrange(
            indices,
            [0 for _ in self._shape],
            new_lst,
        )

        new_shape = tuple(self._shape[i] for i in indices)
        return type(self)(new_lst, new_shape)

    @property
    def m(self) -> int:
        """
        The number of possible values for a tensor's first index.
        """
        return self._shape[0]

    @property
    def n(self) -> int:
        """
        The number of possible values for a tensor's last index.
        """
        return self._shape[-1]

    def __getitem__(self, key: Dims) -> NumberT:
        """
        Returns the item at the given index.
        """
        return self._lst[dot(key, self._idx_mul)]

    def __setitem__(self, key: Dims, value: NumberT) -> None:
        """
        Sets the value of the item at the given index.
        """
        self._lst[dot(key, self._idx_mul)] = value

    def __iter__(self) -> Iterator[NumberT]:
        """
        Returns an iterator that iterates through all the values in a tensor
        depth-first.
        """
        return iter(self._lst)

    def __eq__(self, other: Any) -> bool:
        """
        Returns ``True`` if all values in a tensor are equal.
        """
        if not isinstance(other, Tensor) or self._shape != other._shape:
            return False

        for x, y in zip(self, other):
            if x != y:
                return False

        return True

    def __mul__(self, other: 'NumberT') -> 'Tensor[NumberT]':
        """
        Multiplies a tensor by a scalar value.
        """
        return type(self)([other * x for x in self], self._shape)

    __rmul__ = __mul__

    def __add__(self, other: 'Tensor[NumberT]') -> 'Tensor[NumberT]':
        """
        Returns the element-wise sum of two tensors.
        """
        if self._shape != other._shape:
            raise TensorError('Tensors must have same shape')

        return type(self)([x + y for x, y in zip(self, other)], self._shape)

    def __sub__(self, other):
        return self + -1 * other

    @property
    def T(self) -> 'Tensor[NumberT]':
        """
        The transpose of a tensor i.e. a new tensor with all indices reversed.
        """
        return self.rearrange(*reversed(range(self.order)))

    def __matmul__(self, other: 'Tensor[NumberT]') -> 'Tensor[NumberT]':
        """
        Returns the matrix product of two tensors i.e. the contraction between
        the first tensor's last index and the second tensor's first.
        """
        if self.n != other.m:
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

    def tolist(self) -> List:
        """
        Returns a python list representation of a tensor.
        """
        return unflatten(self._lst, self._idx_mul[:-1])

    def __str__(self) -> str:
        return pprint.pformat(self.tolist())

    def __repr__(self) -> str:
        return str(self)
