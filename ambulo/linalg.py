import functools
import operator
import pprint


def chunks(lst, n):
    i = 0
    len_lst = len(lst)

    while i < len_lst:
        yield lst[i:i + n]
        i += n


def flatten(seq, seqtypes=(list, tuple)):
    # Make copy and convert to list
    seq = list(seq)

    # Flatten list in-place
    for i, _ in enumerate(seq):
        while isinstance(seq[i], seqtypes):
            seq[i:i + 1] = seq[i]

    return seq


def unflatten(seq, multipliers):
    if len(multipliers) == 0:
        return seq

    lst = []

    for seq_ in chunks(seq, multipliers[0]):
        lst.append(unflatten(seq_, multipliers[1:]))

    return lst


def get_seq_dims(seq):
    dims = [len(seq)]
    seq_ = seq[0]

    while isinstance(seq_, (list, tuple)):
        dims.append(len(seq_))
        seq_ = seq_[0]

    if not seq_has_dims(seq, dims):
        raise ValueError('Sequence dimensions are not square')

    return dims


def seq_has_dims(seq, dims):
    if len(dims) == 0:
        return True

    if len(seq) != dims[0]:
        return False

    return all(seq_has_dims(seq_, dims[1:]) for seq_ in seq)


def to_tuple(old_fn):
    @functools.wraps(old_fn)
    def new_fn(*args, **kwargs):
        return tuple(old_fn(*args, **kwargs))

    return new_fn


@to_tuple
def get_idx_multipliers(dims):
    total = functools.reduce(operator.mul, dims)

    for d in dims:
        total //= d
        yield total


def get_flat_idx(indices, multipliers):
    return sum(i * d for i, d in zip(indices, multipliers))


class Matrix:
    def __init__(self, lst, dims=None):
        self._lst = flatten(lst)

        if dims is not None:
            self.dims = tuple(dims)
            if len(lst) != functools.reduce(operator.mul, dims):
                raise ValueError('Given sequence cannot be cast into given dimensions')
        else:
            self.dims = tuple(get_seq_dims(lst))

        self._idx_mul = get_idx_multipliers(self.dims)

    @property
    def m(self):
        return self.dims[0]

    @property
    def n(self):
        return self.dims[1]

    def __getitem__(self, key):
        return self._lst[get_flat_idx(key, self._idx_mul)]

    def __setitem__(self, key, value):
        self._lst[get_flat_idx(key, self._idx_mul)] = value

    def __iter__(self):
        return iter(self._lst)

    def __eq__(self, other):
        if self.dims != other.dims:
            return False

        for x, y in zip(self, other):
            if x != y:
                return False

        return True

    def __mul__(self, other):
        return type(self)([
            other * x for x in self
        ], self.dims)

    __rmul__ = __mul__

    def __add__(self, other):
        if self.dims != other.dims:
            raise ValueError('Tensors must have same dimensions')

        return type(self)([
            x + y for x, y in zip(self, other)
        ], self.dims)

    def __sub__(self, other):
        return self + -1 * other

    @property
    def T(self):
        return type(self)([
            [self[i, j] for i in range(self.m)]
            for j in range(self.n)
        ])

    def __matmul__(self, other):
        if self.dims[-1] != other.dims[0]:
            raise ValueError('Matrices must have compatible dimensions')

        return type(self)([
            [
                sum(self[i, j] * other[j, k] for j in range(self.n))
                for k in range(other.n)
            ]
            for i in range(self.m)
        ])

    def __str__(self):
        return pprint.pformat(
            unflatten(self._lst, self._idx_mul[:-1]),
        )

    def __repr__(self):
        return str(self)
