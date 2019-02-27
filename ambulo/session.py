import pprint
import typing
from typing import (
    Optional,
)

from .types import (
    Number,
    Workspace,
)

if typing.TYPE_CHECKING:
    from .node import BaseNode  # noqa: F401


class SessionError(Exception):
    pass


class Session:
    values: Workspace
    deltas: Workspace

    def __init__(self,
                 values: Optional[Workspace] = None,
                 deltas: Optional[Workspace] = None):
        self.values = values or {}
        self.deltas = deltas or {}

    def set_value(self, node: 'BaseNode', value: Number) -> None:
        self.values[node] = value

    def set_delta(self, node: 'BaseNode', delta: Number) -> None:
        self.deltas[node] = delta

    def get_value(self, node: 'BaseNode') -> Number:
        try:
            return self.values[node]
        except KeyError as e:
            raise SessionError(f'Cannot find value for {repr(node)}') from e

    def get_delta(self, node: 'BaseNode') -> Number:
        try:
            return self.deltas[node]
        except KeyError as e:
            raise SessionError(f'Cannot find delta for {repr(node)}') from e

    def has_value(self, node: 'BaseNode') -> bool:
        return node in self.values

    def has_delta(self, node: 'BaseNode') -> bool:
        return node in self.deltas

    def __repr__(self):
        values = pprint.pformat(self.values)
        deltas = pprint.pformat(self.deltas)

        return f'Values:\n{values}\n\nDeltas:\n{deltas}'
