from collections.abc import MutableSet

class CaseInsensitiveDict(dict):
    __slots__ = ('_values',)

    @staticmethod
    def _fold(key):
        return key.lower() if isinstance(key, str) else key

    def __init__(self, *args, **kwargs):
        self._values = {}
        super().__init__()
        for k, v in dict(*args, **kwargs).items():
            self._values[self._fold(k)] = (k, v)

    def __getitem__(self, key):
        _, val = self._values[self._fold(key)]
        return val

    def __setitem__(self, key, value):
        fk = self._fold(key)
        orig_key = key if fk not in self._values else self._values[fk][0]
        self._values[fk] = (orig_key, value)

    def __delitem__(self, key):
        del self._values[self._fold(key)]

    def __iter__(self):
        return (orig for orig, _ in self._values.values())

    def __len__(self):
        return len(self._values)

    def items(self):
        return ((orig, val) for orig, val in self._values.values())

    def keys(self):
        return (orig for orig, _ in self._values.values())

    def values(self):
        return (val for _, val in self._values.values())

    def get(self, key, default=None):
        return self._values.get(self._fold(key), (None, default))[1]


class CaseInsensitiveSet(MutableSet):
    __slots__ = ('_values',)

    def __init__(self, iterable=None):
        self._values = {}
        if iterable:
            for item in iterable:
                self.add(item)

    def __contains__(self, item):
        return self._fold(item) in self._values

    def add(self, item):
        self._values[self._fold(item)] = item

    def discard(self, item):
        self._values.pop(self._fold(item), None)

    def __iter__(self):
        return iter(self._values.values())

    def __len__(self):
        return len(self._values)

    @staticmethod
    def _fold(value):
        return value.lower() if isinstance(value, str) else value



class StringCaseInsensitiveSet(MutableSet):
    """String set that preserves case but tests for containment by case-folded value

    E.g. 'Foo' in CasePreservingSet(['FOO']) is True. Preserves case of *last*
    inserted variant.

    """
    def __init__(self, *args):
        self._values = {}
        if len(args) > 1:
            raise TypeError(
                f"{type(self).__name__} expected at most 1 argument, "
                f"got {len(args)}"
            )
        values = args[0] if args else ()
        self._fold = str.casefold  # Python 3
        for v in values:
            if type(v) == str:
                self.add(v)
            else:
                raise TypeError('expected element of type str, got {} which is {}'.format(v,type(v)))

    def __repr__(self):
        return '<{}{} at {:x}>'.format(
            type(self).__name__, tuple(self._values.values()), id(self))

    def __ior__(self, it):
        for v in it:
            if type(v) == str:
                self.add(v)
            else:
                raise TypeError('expected element of type str, got {} which is {}'.format(v, type(v)))
        return self

    def update(self,it):
        self.__ior__(it)

    def isdisjoint(self, other):
        'Return True if two sets have a null intersection.'
        for value in other:
            if value.casefold() in self or value in self:
                return False
        return True

    def intersection (self,other):
        interseccion = set()

        if self.isdisjoint(other) == False:
            for value in other:
                if value.casefold() in self or value in self:
                    interseccion.add(value)
            return interseccion
        else:
            return None

    def issuperset (self,other):
        if len(self)>len(other):
            for value in other:
                if ((value.casefold() in self) == False) and ((value in self)==False):
                    return False
            return True
        else:
            return False

    def __eq__(self, other):

        if isinstance(other, StringCaseInsensitiveSet):
            if len(other._values) == len(self._values):
                for v in list(other._values.keys()):
                    if v not in list( self._values.keys()):
                        return False
                return True
            return False
        return NotImplemented

    def __contains__(self, value):
        return self._fold(value) in self._values

    def __iter__(self):
        try:
            # Python 2
            return self._values.itervalues()
        except AttributeError:
            # Python 3
            return iter(self._values.values())

    def __len__(self):
        return len(self._values)

    def add(self, value):
        self._values[self._fold(value)] = value

    def discard(self, value):
        try:
            del self._values[self._fold(value)]
        except KeyError:
            pass

