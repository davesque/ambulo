import abc
import typing
from typing import (
    List,
    Optional,
    Tuple,
)

from .session import (
    Session,
)
from .types import (
    Label,
    Number,
    NumberT,
    Vector,
)

if typing.TYPE_CHECKING:
    from .ops import (  # noqa: F401
        Add,
        Sub,
        Mul,
        Div,
    )


def one_hot_vec(v: NumberT, i: int, n: int) -> Vector[NumberT]:
    """
    Returns a vector of ``n`` zero-elements except for the value ``v`` at
    zero-index ``i``.
    """
    if i >= n:
        raise ValueError('Element index must be less than vector length')

    pref: Tuple[NumberT, ...] = (0,) * i
    suff: Tuple[NumberT, ...] = (0,) * (n - i - 1)

    return pref + (v,) + suff


class BaseNode(abc.ABC):
    label: Optional[Label]
    outputs: List['Node']

    def __init__(self, label: Optional[Label] = None):
        self.label = label
        self.outputs = []

    @abc.abstractmethod
    def eval(self, sess: Session) -> Number:
        pass

    @abc.abstractmethod
    def du_dx(self, sess: Session) -> Number:
        pass

    def df_du(self, sess: Session) -> Number:
        if sess.has_delta(self):
            return sess.get_delta(self)

        deltas_ = (o.du_du(sess, self) for o in self.outputs)

        delta = sum(deltas_)
        sess.set_delta(self, delta)

        return delta

    def __str__(self) -> str:
        if self.label is None:
            return f'{type(self).__name__}(id={id(self)})'

        return self.label

    @property
    def brief_expr(self) -> str:
        return str(self)

    @property
    def full_expr(self) -> str:
        return str(self)

    def __repr__(self) -> str:
        return str(self)

    @property
    def wrap_in_parens(self) -> bool:
        return False

    def __add__(self, other: 'BaseNode') -> 'Add':
        from .ops import Add  # noqa: F811

        return Add(self, other)

    def __sub__(self, other: 'BaseNode') -> 'Sub':
        from .ops import Sub  # noqa: F811

        return Sub(self, other)

    def __mul__(self, other: 'BaseNode') -> 'Mul':
        from .ops import Mul  # noqa: F811

        return Mul(self, other)

    def __truediv__(self, other: 'BaseNode') -> 'Div':
        from .ops import Div  # noqa: F811

        return Div(self, other)


class Node(BaseNode):
    inputs: Tuple['BaseNode', ...]

    def __init__(self,
                 *inputs: 'BaseNode',
                 label: Optional[Label] = None):
        for i in inputs:
            i.outputs.append(self)

        self.inputs = inputs

        super().__init__(label)

    def eval(self, sess: Session) -> Number:
        if sess.has_value(self):
            return sess.get_value(self)

        inputs_ = (i.eval(sess) for i in self.inputs)

        value = self.f(*inputs_)
        sess.set_value(self, value)

        return value

    def du_dx(self, sess: Session) -> Number:
        if sess.has_delta(self):
            return sess.get_delta(self)

        inputs_ = (i.eval(sess) for i in self.inputs)
        deltas_ = (i.du_dx(sess) for i in self.inputs)

        delta = self.df(*inputs_, *deltas_)
        sess.set_delta(self, delta)

        return delta

    def du_du(self, sess: Session, input: 'BaseNode') -> Number:
        inputs_ = (i.eval(sess) for i in self.inputs)
        delta_ = self.df_du(sess)

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
    def f(self, *args: Number) -> Number:
        pass

    @abc.abstractmethod
    def df(self, *args: Number) -> Number:
        pass


class Var(BaseNode):
    def eval(self, sess: Session) -> Number:
        return sess.get_value(self)

    def du_dx(self, sess: Session) -> Number:
        return sess.get_delta(self)
