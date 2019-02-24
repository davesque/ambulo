from decimal import (
    Decimal,
)
from fractions import (
    Fraction,
)
from typing import (
    Mapping,
    Sequence,
    Union,
)


Label = str
Number = Union[float, int, Decimal, Fraction]
Vector = Sequence[Number]
Dims = Sequence[int]
Workspace = Mapping[Label, Number]

RawTensor = Sequence[Union[Vector, 'RawTensor']]
