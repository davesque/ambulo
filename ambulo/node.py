import abc
import math
import operator
from typing import (
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)


Number = Union[float, int]

Name = str

Env = Dict[Name, Number]
ValEnv = Dict[Name, 'Val']
MixEnv = Dict[Name, Union['Val', Number, Tuple[Number, Number]]]


class Val(NamedTuple):
    v: Optional[Number]
    dv: Optional[Number]

    @classmethod
    def from_tuple(cls, t: Tuple[Number, Number]) -> 'Val':
        if not isinstance(t, tuple) or len(t) != 2:
            raise ValueError('Val instances can only be created from 2-tuples')

        return cls(*t)


def to_valenv(env: MixEnv) -> ValEnv:
    valenv: ValEnv = {}

    for k, v in env.items():
        if isinstance(v, tuple):
            valenv[k] = Val.from_tuple(v)
        else:
            valenv[k] = Val(v, 0)

    return valenv


class BaseNode(abc.ABC):
    name: Name
    outputs: List['Node']

    wrap_in_parens = False

    def __init__(self, name: Name):
        self.outputs = []
        self.name = name

    @abc.abstractmethod
    def eval(self, **env: Env) -> Number:
        pass

    def fwd(self, **env: MixEnv) -> Val:
        return self._fwd(**to_valenv(env))

    @abc.abstractmethod
    def _fwd(self, **env: ValEnv) -> Val:
        pass

    def __str__(self) -> str:
        return self.name

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

    def __init__(self, *inputs: Tuple['Node', ...], name=None):
        for i in inputs:
            i.outputs.append(self)

        self.inputs = inputs
        self.outputs = []
        self.name = name

    def eval(self, **env: Env) -> Number:
        nums = (i.eval(**env) for i in self.inputs)

        return self.f(*nums)

    def _fwd(self, **env: ValEnv) -> Val:
        vals = tuple(i._fwd(**env) for i in self.inputs)

        vs = tuple(val.v for val in vals)
        dvs = tuple(val.dv for val in vals)

        return Val(self.f(*vs), self.df(*vs, *dvs))

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
    def eval(self, **env: Env) -> Number:
        try:
            return env[self.name]
        except KeyError:
            raise VarError(f'Cannot evaluate variable "{self}"')

    def _fwd(self, **env: ValEnv) -> Val:
        try:
            return env[self.name]
        except KeyError:
            raise VarError(f'Cannot evaluate variable "{self}"')


class Id(Node):
    def f(self, x):
        return x

    def df(self, x, dx):
        return dx

    @property
    def name_expr(self):
        x = self.inputs[0]
        return f'{x.name}'

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
        return f'{self.op_str}({x.name})'

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
        return f'{x.name}{self.op_str}{y.name}'

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
        return (dx * y - x * dy) / (y * y)
