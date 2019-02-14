import math

import pytest

from ambulo.node import (
    Node,
    Var,
    VarError,
)


class MySin(Node):
    def f(self, x):
        return math.sin(x)

    def df(self, x, dx):
        return dx * math.cos(x)

    @property
    def name_expr(self):
        x = self.inputs[0]
        return f'sin({x.name})'

    @property
    def full_expr(self):
        x = self.inputs[0]
        return f'sin({x})'


class MyMul(Node):
    def f(self, x, y):
        return x * y

    def df(self, x, y, dx, dy):
        return x * dy + y * dx

    @property
    def name_expr(self):
        x, y = self.inputs
        return f'{x.name} * {y.name}'

    @property
    def full_expr(self):
        x, y = self.inputs
        return f'{x} * {y}'


@pytest.fixture
def x():
    return Var('x')


@pytest.fixture
def y():
    return Var('y')


@pytest.fixture
def sin_x(x):
    return MySin(x)


@pytest.fixture
def mul_xy(x, y):
    return MyMul(x, y)


def test_var_str(x):
    assert str(x) == 'x'


def test_var_eval(x):
    assert x.eval(x=2) == 2


def test_var_eval_raises_val_error(x):
    with pytest.raises(VarError):
        x.eval()

    with pytest.raises(VarError):
        x.eval(y=2)


def test_node_init_connets_nodes_both_ways(mul_xy, x, y):
    assert x in mul_xy.inputs
    assert y in mul_xy.inputs

    assert mul_xy in x.outputs
    assert mul_xy in y.outputs


def test_node_eval_gets_numerical_result(mul_xy):
    assert mul_xy.eval(x=2, y=5) == 10


def test_node_arity(mul_xy, sin_x):
    assert mul_xy.arity == 2
    assert sin_x.arity == 1


def test_node_is_unary(mul_xy, sin_x):
    assert not mul_xy.is_unary
    assert sin_x.is_unary
