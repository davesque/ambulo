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
Workspace = Dict[Label, Number]


class Env:
    values: Workspace
    deltas: Workspace

    def __init__(self, values: Optional[Workspace]=None, deltas: Optional[Workspace]=None):
        self.values = values or {}
        self.deltas = deltas or {}

    def set_value(self, label, value):
        self.values[label] = value

    def set_delta(self, label, delta):
        self.deltas[label] = delta

    def get_value(self, label: Label) -> Number:
        return self.values[label]

    def get_delta(self, label: Label) -> Number:
        return self.deltas[label]

    def has_value(self, label: Label) -> bool:
        return label in self.values

    def has_delta(self, label: Label) -> bool:
        return label in self.deltas

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
    def frwd(self, env: Env) -> Number:
        pass

    @abc.abstractmethod
    def back(self, env: Env) -> Number:
        pass

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
        label = self.label
        if env.has_value(label):
            return env.get_value(label)

        inputs = (i.eval(env) for i in self.inputs)

        value = self.f(*inputs)
        env.set_value(label, value)

        return value

    def frwd(self, env: Env) -> Number:
        label = self.label
        if env.has_delta(label):
            return env.get_delta(label)

        inputs = (i.eval(env) for i in self.inputs)
        deltas = (i.frwd(env) for i in self.inputs)

        delta = self.df(*inputs, *deltas)
        env.set_delta(label, delta)

        return delta

    def back(self, env):
        inputs = tuple(inp.eval(**env) for inp in self.inputs)

        out_indices = tuple(out.inputs.index(self) for out in self.outputs)
        vo_bars = tuple(out.back(**env)[i] for i, out in zip(out_indices, self.outputs))

        return self.di(*inputs, *vo_bars)

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
        return env.get_value(self.label)

    def frwd(self, env: Env) -> Number:
        return env.get_delta(self.label)

    def back(self, env: Env) -> Number:
        return env.get_value(self.label)


class Id(Node):
    def back(self, env):
        if len(self.outputs) == 0:
            try:
                return [env[self.label]]
            except KeyError:
                raise VarError(f'Cannot evaluate output variable "{self}"')

        return super().back(**env)

    def f(self, x):
        return x

    def df(self, x, dx):
        return dx

    def di(self, x, *vo_bars):
        v_bar = sum(vo_bars)
        return [v_bar]

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

    def di(self, x, *vo_bars):
        v_bar = sum(vo_bars)
        return [v_bar / x]


class Sin(Unary):
    op = staticmethod(math.sin)
    op_str = 'sin'

    def df(self, x, dx):
        return math.cos(x) * dx

    def di(self, x, *vo_bars):
        v_bar = sum(vo_bars)
        return [v_bar * math.cos(x)]


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

    def di(self, x, y, *vo_bars):
        v_bar = sum(vo_bars)
        return [v_bar, v_bar]


class Sub(Binary):
    op = staticmethod(operator.sub)
    op_str = ' - '

    def df(self, x, y, dx, dy):
        return dx - dy

    def di(self, x, y, *vo_bars):
        v_bar = sum(vo_bars)
        return [v_bar, -v_bar]


class Mul(Binary):
    op = staticmethod(operator.mul)
    op_str = ' '

    wrap_in_parens = False

    def df(self, x, y, dx, dy):
        return x * dy + dx * y

    def di(self, x, y, *vo_bars):
        v_bar = sum(vo_bars)
        return [y * v_bar, x * v_bar]


class Div(Binary):
    op = staticmethod(operator.truediv)
    op_str = ' / '

    def df(self, x, y, dx, dy):
        return (dx * y - x * dy) / y ** 2

    def di(self, x, y, *vo_bars):
        v_bar = sum(vo_bars)
        return [v_bar / y, v_bar * (-x / y ** 2)]
