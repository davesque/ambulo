import abc
import math
import operator
import pprint
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)


Label = str
Number = Union[float, int]
Vector = Tuple[Number, ...]
Workspace = Dict[Label, Number]


def one_hot_vec(v: Number, i: int, n: int) -> Vector:
    """
    Returns a one-hot vector of ``n`` elements with the value ``v`` as the
    element at zero-index ``i``.
    """
    if i >= n:
        raise ValueError('Element index must be less than vector length')

    return (0,) * i + (v,) + (0,) * (n - i - 1)


class Env:
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


class BaseNode(abc.ABC):
    label: Label
    outputs: List['Node']

    wrap_in_parens = False

    def __init__(self, label: Label):
        self.outputs = []
        self.label = label

    @abc.abstractmethod
    def eval(self, env: Env) -> Number:
        pass

    @abc.abstractmethod
    def df_dx(self, env: Env) -> Number:
        pass

    def dy_df(self, env) -> Number:
        if env.has_delta(self):
            return env.get_delta(self)

        deltas_ = (o.df_dv(env, self) for o in self.outputs)

        delta = sum(deltas_)
        env.set_delta(self, delta)

        return delta

    def __str__(self) -> str:
        return self.label

    @property
    def name_expr(self) -> str:
        return str(self)

    @property
    def full_expr(self) -> str:
        return str(self)

    def __repr__(self) -> str:
        return str(self)

    def __add__(self, other: 'BaseNode') -> 'Add':
        return Add(self, other)

    def __sub__(self, other: 'BaseNode') -> 'Sub':
        return Sub(self, other)

    def __mul__(self, other: 'BaseNode') -> 'Mul':
        return Mul(self, other)

    def __div__(self, other: 'BaseNode') -> 'Div':
        return Div(self, other)


class Node(BaseNode):
    inputs: Tuple['Node', ...]

    def __init__(self, *inputs: Tuple['Node', ...], label=None):
        for i in inputs:
            i.outputs.append(self)

        self.inputs = inputs
        self.outputs = []
        self.label = label

    def eval(self, env: Env) -> Number:
        if env.has_value(self):
            return env.get_value(self)

        inputs_ = (i.eval(env) for i in self.inputs)

        value = self.f(*inputs_)
        env.set_value(self, value)

        return value

    def df_dx(self, env: Env) -> Number:
        if env.has_delta(self):
            return env.get_delta(self)

        inputs_ = (i.eval(env) for i in self.inputs)
        deltas_ = (i.df_dx(env) for i in self.inputs)

        delta = self.df(*inputs_, *deltas_)
        env.set_delta(self, delta)

        return delta

    def df_dv(self, env, input) -> Number:
        inputs_ = (i.eval(env) for i in self.inputs)
        delta_ = self.dy_df(env)

        i = self.inputs.index(input)
        n = len(self.inputs)

        return self.df(*inputs_, *one_hot_vec(delta_, i, n))

    @property
    def arity(self) -> int:
        return len(self.inputs)

    @property
    def is_unary(self) -> bool:
        return self.arity == 1

    @property
    def wrap_in_parens(self) -> bool:
        return not self.is_unary

    @abc.abstractmethod
    def f(self, *args) -> Number:
        pass

    @abc.abstractmethod
    def df(self, *args) -> Number:
        pass


class VarError(Exception):
    pass


class Var(BaseNode):
    def eval(self, env: Env) -> Number:
        return env.get_value(self)

    def df_dx(self, env: Env) -> Number:
        return env.get_delta(self)


class Id(Node):
    def dy_df(self, env) -> Number:
        if len(self.outputs) == 0:
            return env.get_delta(self)

        return super().dy_df(env)

    def f(self, x):
        return x

    def df(self, x, dx):
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
    def f(self, x):
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

    def df(self, x, dx):
        return dx / x


class Sin(Unary):
    op = staticmethod(math.sin)
    op_str = 'sin'

    def df(self, x, dx):
        return math.cos(x) * dx


class Binary(Node):
    op = None
    op_str = None

    def f(self, x, y):
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

    def df(self, x, y, dx, dy):
        return dx + dy


class Sub(Binary):
    op = staticmethod(operator.sub)
    op_str = ' - '

    def df(self, x, y, dx, dy):
        return dx - dy


class Mul(Binary):
    op = staticmethod(operator.mul)
    op_str = ' '

    wrap_in_parens = False

    def df(self, x, y, dx, dy):
        return x * dy + dx * y


class Div(Binary):
    op = staticmethod(operator.truediv)
    op_str = ' / '

    def df(self, x, y, dx, dy):
        return (dx * y - x * dy) / y ** 2
