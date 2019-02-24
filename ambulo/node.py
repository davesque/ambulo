import abc
from typing import (
    List,
    Optional,
    Tuple,
)

from .session import Session
from .types import (
    Label,
    Number,
    Vector,
)


def one_hot_vec(v: Number, i: int, n: int) -> Vector:
    """
    Returns a one-hot vector of ``n`` elements with the value ``v`` as the
    element at zero-index ``i``.
    """
    if i >= n:
        raise ValueError('Element index must be less than vector length')

    return (0,) * i + (v,) + (0,) * (n - i - 1)


class BaseNode(abc.ABC):
    label: Label
    outputs: List['Node']

    wrap_in_parens = False

    def __init__(self, label: Label):
        self.outputs = []
        self.label = label

    @abc.abstractmethod
    def eval(self, sess: Session) -> Number:
        pass

    @abc.abstractmethod
    def df_di(self, sess: Session) -> Number:
        pass

    def do_df(self, sess: Session) -> Number:
        if sess.has_delta(self):
            return sess.get_delta(self)

        deltas_ = (o.df_df(sess, self) for o in self.outputs)

        delta = sum(deltas_)
        sess.set_delta(self, delta)

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
        from .ops import Add

        return Add(self, other)

    def __sub__(self, other: 'BaseNode') -> 'Sub':
        from .ops import Sub

        return Sub(self, other)

    def __mul__(self, other: 'BaseNode') -> 'Mul':
        from .ops import Mul

        return Mul(self, other)

    def __truediv__(self, other: 'BaseNode') -> 'Div':
        from .ops import Div

        return Div(self, other)


class Node(BaseNode):
    inputs: Tuple['Node', ...]

    def __init__(self,
                 *inputs: Tuple['Node', ...],
                 label: Optional[Label] = None):
        for i in inputs:
            i.outputs.append(self)

        self.inputs = inputs
        self.outputs = []
        self.label = label

    def eval(self, sess: Session) -> Number:
        if sess.has_value(self):
            return sess.get_value(self)

        inputs_ = (i.eval(sess) for i in self.inputs)

        value = self.f(*inputs_)
        sess.set_value(self, value)

        return value

    def df_di(self, sess: Session) -> Number:
        if sess.has_delta(self):
            return sess.get_delta(self)

        inputs_ = (i.eval(sess) for i in self.inputs)
        deltas_ = (i.df_di(sess) for i in self.inputs)

        delta = self.df(*inputs_, *deltas_)
        sess.set_delta(self, delta)

        return delta

    def df_df(self, sess: Session, input: 'BaseNode') -> Number:
        inputs_ = (i.eval(sess) for i in self.inputs)
        delta_ = self.do_df(sess)

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
    def f(self, *args: Vector) -> Number:
        pass

    @abc.abstractmethod
    def df(self, *args: Vector) -> Number:
        pass


class Var(BaseNode):
    def eval(self, sess: Session) -> Number:
        return sess.get_value(self)

    def df_di(self, sess: Session) -> Number:
        return sess.get_delta(self)
