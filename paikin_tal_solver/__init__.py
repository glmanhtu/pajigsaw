import sys


def cmp(a, b):
    return (a > b) - (a < b)


_convert = {
    '__eq__': lambda self, other: self.__cmp__(other) == 0,
    '__ne__': lambda self, other: self.__cmp__(other) != 0,
    '__lt__': lambda self, other: self.__cmp__(other) < 0,
    '__le__': lambda self, other: self.__cmp__(other) <= 0,
    '__gt__': lambda self, other: self.__cmp__(other) > 0,
    '__ge__': lambda self, other: self.__cmp__(other) >= 0,
}


def PY3__cmp__(cls):
    """Class decorator that fills in missing ordering methods when
       Python2-style `__cmp__(self, other)` method is provided."""
    if not hasattr(cls, '__cmp__'):
        raise ValueError('must define the __cmp__ Python2-style method')
    if sys.version_info < (3, 0, 0):
        return cls
    for op, opfunc in _convert.items():
        # Overwrite the `raise NotImplemented` comparisons inherited from object
        if getattr(cls, op, None) is getattr(object, op, None):
            setattr(cls, op, opfunc)
    return cls
