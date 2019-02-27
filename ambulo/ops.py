import math
import operator
import typing

from .node import (
    Node,
)
from .types import (
    Number,
)

if typing.TYPE_CHECKING:
    from .session import Session  # noqa: F401


class Const(Node):
    def __init__(self, value: Number):
        self.value = value  # type: ignore

        super().__init__(label=str(value))

    def f(self, *args: Number) -> Number:
        return self.value

    def df(self, *args: Number) -> Number:
        return 0


class Id(Node):
    def do_df(self, sess: 'Session') -> Number:
        if len(self.outputs) == 0:
            return sess.get_delta(self)

        return super().do_df(sess)

    def f(self, *args: Number) -> Number:
        x, = args
        return x

    def df(self, *args: Number) -> Number:
        _, dx = args
        return dx

    @property
    def name_expr(self):
        x = self.inputs[0]
        return f'{x.label}'

    @property
    def full_expr(self):
        x = self.inputs[0]
        return f'{x.full_expr}'


class Unary(Node):
    op = staticmethod(lambda x: x)  # to satisfy mypy

    def f(self, *args: Number) -> Number:
        x, = args
        return self.op(x)

    @property
    def name_expr(self):
        x = self.inputs[0]
        return f'{self.op_str}({x.label})'

    @property
    def full_expr(self):
        x = self.inputs[0]
        return f'{self.op_str}({x.full_expr})'


class Ln(Unary):
    op = staticmethod(math.log)
    op_str = 'ln'

    def df(self, *args: Number) -> Number:
        x, dx = args
        return dx / x


class Sin(Unary):
    op = staticmethod(math.sin)
    op_str = 'sin'

    def df(self, *args: Number) -> Number:
        x, dx = args
        return math.cos(x) * dx


class Binary(Node):
    op = staticmethod(lambda x, y: 0)  # to satisfy mypy
    op_str = ''  # to satisfy mypy

    def f(self, *args: Number) -> Number:
        x, y = args
        return self.op(x, y)

    @property
    def name_expr(self):
        x, y = self.inputs
        return f'{x.label}{self.op_str}{y.label}'

    @property
    def full_expr(self):
        x, y = self.inputs

        x_exp, y_exp = x.full_expr, y.full_expr

        x_exp = f'({x_exp})' if x.wrap_in_parens else f'{x_exp}'
        y_exp = f'({y_exp})' if y.wrap_in_parens else f'{y_exp}'

        return f'{x_exp}{self.op_str}{y_exp}'


class Add(Binary):
    op = staticmethod(operator.add)
    op_str = ' + '

    def df(self, *args: Number) -> Number:
        _, _, dx, dy = args
        return dx + dy


class Sub(Binary):
    op = staticmethod(operator.sub)
    op_str = ' - '

    def df(self, *args: Number) -> Number:
        _, _, dx, dy = args
        return dx - dy


class Mul(Binary):
    op = staticmethod(operator.mul)
    op_str = ' '

    wrap_in_parens = False

    def df(self, *args: Number) -> Number:
        x, y, dx, dy = args
        return x * dy + dx * y


class Div(Binary):
    op = staticmethod(operator.truediv)
    op_str = ' / '

    def df(self, *args: Number) -> Number:
        x, y, dx, dy = args
        return (dx * y - x * dy) / y ** 2
