import functools
import operator
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Sequence,
    Tuple,
    Type,
    TypeVar,
)

from .types import (
    Dims,
    Number,
    Vector,
)

T = TypeVar('T')


def chunks(seq: Sequence[T], n: int) -> Iterator[Sequence[T]]:
    """
    Yields consecutive ``n``-length sub-sequences of the sequence ``seq``.
    """
    i = 0
    len_lst = len(seq)

    while i < len_lst:
        yield seq[i:i + n]
        i += n


def flatten(seq: Sequence,
            seqtypes: Tuple[Type[Sequence], ...] = (list, tuple)) -> List:
    """
    Converts the sequence ``seq``, containing arbitrarily deep nestings of
    sequence types ``seqtypes``, into a flat list.
    """
    # Make copy and convert to list
    seq = list(seq)

    # Flatten list in-place
    for i, _ in enumerate(seq):
        while isinstance(seq[i], seqtypes):
            seq[i:i + 1] = seq[i]

    return seq


def unflatten(seq: Sequence, multipliers: Sequence[int]) -> List:
    """
    Converts the flat sequence ``seq`` into an abritraily nested list with the
    number of elements composing each nested element provided in ``multiplers``.
    """
    if len(multipliers) == 1:
        return list(chunks(seq, multipliers[0]))

    multipliers_ = multipliers[1:]
    return [
        unflatten(seq_, multipliers_)
        for seq_ in chunks(seq, multipliers[0])
    ]


def to_tuple(old_fn: Callable[..., Iterable[T]]) -> Callable[..., Tuple[T, ...]]:
    """
    Decorates the function ``old_fn`` to convert its results into a tuple.
    """
    @functools.wraps(old_fn)
    def new_fn(*args: Any, **kwargs: Any) -> Tuple[T, ...]:
        return tuple(old_fn(*args, **kwargs))

    return new_fn


def get_seq_dims(seq: Sequence,
                 seqtypes: Tuple[Type[Sequence], ...] = (list, tuple)) -> Dims:
    """
    Returns the dimensions of each level of nesting in a nested sequence
    ``seq`` and verifies that the dimensions are square across the structure of
    the sequence.
    """
    dims = []
    seq_ = seq

    while isinstance(seq_, seqtypes):
        dims.append(len(seq_))
        try:
            seq_ = seq_[0]
        except IndexError:
            break

    if not seq_has_dims(seq, dims):
        raise ValueError('Sequence dimensions are not square')

    return tuple(dims)


def seq_has_dims(seq: Sequence, dims: Dims) -> bool:
    """
    Verifies that a nested sequence ``seq`` is square with respect to the given
    dimensions in ``dims``.
    """
    if len(dims) == 0:
        return True

    if len(seq) != dims[0]:
        return False

    dims_ = dims[1:]
    return all(seq_has_dims(seq_, dims_) for seq_ in seq)


def product(x: Vector) -> Number:
    """
    Returns the product of all elements in ``x``.
    """
    if len(x) == 0:
        raise ValueError('Cannot determine product of vector with no elements')

    return functools.reduce(operator.mul, x)


def dot(x: Vector, y: Vector) -> Number:
    """
    Returns the dot product of the elements in the vectors ``x`` and ``y``.
    """
    len_x, len_y = len(x), len(y)

    if len_x != len_y:
        raise ValueError('Cannot take dot product of vectors with different dimensions')
    if len_x == 0 or len_y == 0:
        raise ValueError('Cannot take dot product of vectors with no elements')

    return sum(a * b for a, b in zip(x, y))


@to_tuple
def get_idx_multipliers(dims: Dims) -> Iterator[int]:
    """
    For dimensions with sizes given in ``dims``, returns the number of elements
    identified by walking down each dimension.

    For example, let dimensions ``(3, 3, 3, 3)`` represent the dimensions of an
    order-4 tensor with 4 indices and 3 possible values for each index.
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
