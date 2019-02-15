import pprint
from typing import Optional

from .types import (
    Number,
    Workspace,
)


class Session:
    values: Workspace
    deltas: Workspace

    def __init__(self, values: Optional[Workspace]=None, deltas: Optional[Workspace]=None):
        self.values = values or {}
        self.deltas = deltas or {}

    def set_value(self, node: 'BaseNode', value: Number):
        self.values[node] = value

    def set_delta(self, node: 'BaseNode', delta: Number):
        self.deltas[node] = delta

    def get_value(self, node: 'BaseNode') -> Number:
        return self.values[node]

    def get_delta(self, node: 'BaseNode') -> Number:
        return self.deltas[node]

    def has_value(self, node: 'BaseNode') -> bool:
        return node in self.values

    def has_delta(self, node: 'BaseNode') -> bool:
        return node in self.deltas

    def __repr__(self):
        values = pprint.pformat(self.values)
        deltas = pprint.pformat(self.deltas)

        return f'Values:\n{values}\n\nDeltas:\n{deltas}'
