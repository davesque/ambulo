import typing
from typing import (
    MutableMapping,
    Sequence,
    Union,
)

if typing.TYPE_CHECKING:
    from .node import BaseNode  # noqa: F401


Label = str
Number = Union[float, int]
Vector = Sequence[Number]
Dims = Sequence[int]
Workspace = MutableMapping['BaseNode', Number]

RawTensor = Sequence[Union[Vector, 'RawTensor']]
