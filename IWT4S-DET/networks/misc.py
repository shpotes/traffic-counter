from typing import Iterable
from functools import reduce

def prod(x : Iterable[int]) -> int:
    """multiply all elements of an iterable object"""
    return reduce(lambda a, b: a * b, x)
