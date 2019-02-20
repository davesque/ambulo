from decimal import (
    Decimal,
)
from fractions import (
    Fraction,
)
from typing import (
    Union,
    Tuple,
    Dict,
)


Label = str
Number = Union[float, int, Decimal, Fraction]
Vector = Tuple[Number, ...]
Workspace = Dict[Label, Number]
