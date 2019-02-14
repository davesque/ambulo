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
MixEnv = Dict[Name, Union['Val', Number, Tuple[Number, Optional[Number]]]]


class Val(NamedTuple):
    v: Optional[Number]
    dv: Optional[Number]

    @classmethod
    def from_tuple(cls, t):
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


class ReprMixin:
    def __repr__(self):
        return str(self)


class OpsMixin:
    def __add__(self, other):
        return Add(self, other)

    def __sub__(self, other):
        return Sub(self, other)

    def __mul__(self, other):
        return Mul(self, other)

    def __div__(self, other):
        return Div(self, other)


class BaseNode(OpsMixin, ReprMixin, abc.ABC):
    name: Name
    outputs: List['Node']

    def __init__(self, name: Name):
        self.outputs = []
        self.name = name

    @abc.abstractproperty
    def arity(self):
        pass

    @property
    def is_unary(self):
        return self.arity == 1

    @property
    def wrap_in_parens(self):
        return not self.is_unary

    @abc.abstractmethod
    def eval(self, **env: Env) -> Number:
        pass

    @abc.abstractmethod
    def fwd(self, **env: MixEnv) -> Val:
        pass

    @abc.abstractmethod
    def _fwd(self, **env: ValEnv) -> Val:
        pass

    @abc.abstractproperty
    def name_expr(self):
        pass

    @abc.abstractproperty
    def full_expr(self):
        pass

    def __str__(self):
        return self.name


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

    def fwd(self, **env: MixEnv) -> Val:
        return self._fwd(**to_valenv(env))

    def _fwd(self, **env: ValEnv) -> Val:
        vals = tuple(i._fwd(**env) for i in self.inputs)

        vs = tuple(val.v for val in vals)
        dvs = tuple(val.dv for val in vals)

        return Val(self.f(*vs), self.df(*vs, *dvs))

    @property
    def arity(self):
        return len(self.inputs)

    @abc.abstractmethod
    def f(self, *args):
        pass

    @abc.abstractmethod
    def df(self, *args):
        pass


class VarError(Exception):
    pass


class Var(BaseNode):
    arity = 0
    wrap_in_parens = False

    def eval(self, **env: Env) -> Number:
        try:
            return env[self.name]
        except KeyError:
            raise VarError(f'Cannot evaluate variable "{self}"')

    def fwd(self, **env: MixEnv) -> Val:
        return self._fwd(**to_valenv(env))

    def _fwd(self, **env: ValEnv) -> Val:
        try:
            return env[self.name]
        except KeyError:
            raise VarError(f'Cannot evaluate variable "{self}"')

    @property
    def name_expr(self):
        return str(self)

    @property
    def full_expr(self):
        return str(self)


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
        return f'{x}'


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
        return f'{self.op_str}({x})'


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

        x = f'({x})' if x.wrap_in_parens else f'{x}'
        y = f'({y})' if y.wrap_in_parens else f'{y}'

        return f'{x}{self.op_str}{y}'


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
