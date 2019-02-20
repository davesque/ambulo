import functools
from typing import (
    Any,
    Callable,
    Iterator,
    List,
    Sequence,
    Tuple,
    Type,
    TypeVar,
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


def flatten(
    seq: Sequence,
    seqtypes: Tuple[Type[Sequence], ...] = (list, tuple),
) -> List:
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


def to_tuple(old_fn: Callable[..., Any]) -> Callable[..., Tuple[Any, ...]]:
    """
    Decorates the function ``old_fn`` to convert its results into a tuple.
    """
    @functools.wraps(old_fn)
    def new_fn(*args, **kwargs) -> Tuple[Any, ...]:
        return tuple(old_fn(*args, **kwargs))

    return new_fn


def get_seq_dims(
    seq: Sequence,
    seqtypes: Tuple[Type[Sequence], ...] = (list, tuple),
) -> Tuple[int, ...]:
    """
    Returns the dimensions of each level of nesting in a nested sequence ``seq``
    and verifies that the dimensions are square across the structure of the
    sequence.
    """
    dims = []
    seq_ = seq

    while isinstance(seq_, seqtypes):
        dims.append(len(seq_))
        seq_ = seq_[0]

    if not seq_has_dims(seq, dims):
        raise ValueError('Sequence dimensions are not square')

    return tuple(dims)


def seq_has_dims(seq, dims):
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
