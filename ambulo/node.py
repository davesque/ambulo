import abc
import functools
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


class cache:
    def __init__(self, method):
        self.method = method
        self.cache_attr = f'_{method.__name__}_cache'

    def __get__(self, obj=None, objtype=None):
        if obj is None:
            return self.method

        old_method = self.method
        cache_attr = self.cache_attr

        @functools.wraps(old_method)
        def new_method(*args, **kwargs):
            if hasattr(obj, cache_attr):
                return getattr(obj, cache_attr)

            cache_val = old_method(obj, *args, **kwargs)
            setattr(obj, cache_attr, cache_val)

            return cache_val

        def clear_cache():
            try:
                delattr(obj, cache_attr)
            except AttributeError:
                pass

        new_method.clear_cache = clear_cache

        return new_method


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


def to_env(mixenv: MixEnv) -> Env:
    env: Env = {}

    for k, v in mixenv.items():
        if isinstance(v, tuple):
            env[k] = v[0]
        elif isinstance(v, Val):
            env[k] = v[0]
        else:
            env[k] = v

    return env


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

    def back(self, **env):
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
    def back(self, **env):
        if len(self.outputs) == 0:
            try:
                return [env[self.name]]
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
