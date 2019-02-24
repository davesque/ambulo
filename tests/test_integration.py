import pytest

from ambulo import (
    Session,
    Add,
    Id,
    Ln,
    Mul,
    Sin,
    Sub,
    Var,
)

x1 = Var('x1')
x2 = Var('x2')

vn1 = Id(x1, label='vn1')
v0 = Id(x2, label='v0')

v1 = Ln(vn1, label='v1')
v2 = Mul(vn1, v0, label='v2')
v3 = Sin(v0, label='v3')
v4 = Add(v1, v2, label='v4')
v5 = Sub(v4, v3, label='v5')

y = Id(v5, label='y')


@pytest.fixture
def sess():
    return Session(
        {x1: 2, x2: 5},
    )


@pytest.fixture
def s_back():
    return Session(
        {x1: 2, x2: 5},
        {y: 1},
    )


def test_forward_tangent_mode_x1(sess):
    sess.set_delta(x1, 1)
    sess.set_delta(x2, 0)

    assert y.df_di(sess) == 5.5


def test_forward_tangent_mode_x2(sess):
    sess.set_delta(x1, 0)
    sess.set_delta(x2, 1)

    assert round(y.df_di(sess), 6) == 1.716338


def test_reverse_adjoint_mode(sess):
    sess.set_delta(y, 1)

    assert x1.do_df(sess) == 5.5
    assert round(x2.do_df(sess), 6) == 1.716338
