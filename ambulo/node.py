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


class VarError(Exception):
    pass


class Val(NamedTuple):
    v: Optional[Number]
    dv: Optional[Number]


class Var(OpsMixin, ReprMixin, abc.ABC):
    name: Name
    outputs: List['Node']

    def __init__(self, name: Name):
        self.name = name
        self.outputs = []

    def eval(self, **env: Env) -> Number:
        try:
            return env[self.name]
        except KeyError:
            raise VarError(f'Cannot evaluate variable "{self}"')

    def fwd(self, **env: ValEnv) -> Val:
        try:
            return env[self.name]
        except KeyError:
            raise VarError(f'Cannot evaluate variable "{self}"')

    arity = 1
    is_unary = True

    def __str__(self):
        return self.name


class Node(OpsMixin, ReprMixin, abc.ABC):
    inputs: Tuple['Node', ...]
    outputs: List['Node']

    def __init__(self, *inputs: Tuple['Node', ...]):
        for i in inputs:
            i.outputs.append(self)

        self.inputs = inputs
        self.outputs = []

    def eval(self, **env: Env) -> Number:
        nums = (i.eval(**env) for i in self.inputs)

        return self.f(*nums)

    def fwd(self, **env: ValEnv) -> Val:
        vals = tuple(i.fwd(**env) for i in self.inputs)

        vs = tuple(val.v for val in vals)
        dvs = tuple(val.dv for val in vals)

        return Val(self.f(*vs), self.df(*vs, *dvs))

    @property
    def arity(self):
        return len(self.inputs)

    @property
    def is_unary(self):
        return self.arity == 1

    @abc.abstractmethod
    def f(self, *args):
        pass

    @abc.abstractmethod
    def df(self, *args):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass


class Unary(Node):
    def f(self, x):
        return self.op(x)

    def __str__(self):
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

    def __str__(self):
        x, y = self.inputs[:2]

        x = f'{x}' if x.is_unary else f'({x})'
        y = f'{y}' if y.is_unary else f'({y})'

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
    op_str = ' * '

    def df(self, x, y, dx, dy):
        return x * dy + dx * y


class Div(Binary):
    op = staticmethod(operator.truediv)
    op_str = ' / '

    def df(self, x, y, dx, dy):
        return (dx * y - x * dy) / (y * y)
