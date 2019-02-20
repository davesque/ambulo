from typing import (
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


def flatten(seq: Sequence, seqtypes: Tuple[Type, ...] = (list, tuple)) -> List:
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
