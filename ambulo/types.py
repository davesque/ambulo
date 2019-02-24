from decimal import (
    Decimal,
)
from fractions import (
    Fraction,
)
import typing
from typing import (
    Mapping,
    Sequence,
    Union,
)

if typing.TYPE_CHECKING:
    from .node import BaseNode  # noqa: F401


Label = str
Number = Union[float, int, Decimal, Fraction]
Vector = Sequence[Number]
Dims = Sequence[int]
Workspace = Mapping['BaseNode', Number]

RawTensor = Sequence[Union[Vector, 'RawTensor']]
