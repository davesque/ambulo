import typing
from typing import (
    MutableMapping,
    Sequence,
    TypeVar,
    Union,
)

if typing.TYPE_CHECKING:
    from sympy import Expr  # noqa: F401
    from .node import BaseNode  # noqa: F401


Label = str
Number = Union[float, int, 'Expr']
NumberT = TypeVar('NumberT', float, int, 'Expr')
Vector = Sequence[NumberT]
Dims = Sequence[int]
Workspace = MutableMapping['BaseNode', Number]

RawTensorItem = Union[NumberT, Vector, 'RawTensor']
RawTensor = Sequence[RawTensorItem[NumberT]]
