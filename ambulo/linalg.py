class Matrix:
    def __init__(self, lst):
        self.m = len(lst)
        self.n = len(lst[0])

        for row in lst[1:]:
            if len(row) != self.n:
                raise ValueError('Matrix rows must have same length')

        self.lst = lst

    def __getitem__(self, key):
        return self.lst[key[0]][key[1]]

    def __setitem__(self, key, value):
        self.lst[key[0]][key[1]] = value

    def __iter__(self):
        return iter(self.lst)

    def __eq__(self, other):
        for s_row, o_row in zip(self, other):
            for x, y in zip(s_row, o_row):
                if x != y:
                    return False

        return True

    def __mul__(self, other):
        return type(self)(
            [[other * x for x in row]
             for row in self]
        )
    __rmul__ = __mul__

    def __add__(self, other):
        if (self.m, self.n) != (other.m, other.n):
            raise ValueError('Matrices must have same dimensions')

        return type(self)(
            [[x + y for x, y in zip(s_row, o_row)]
             for s_row, o_row in zip(self, other)]
        )