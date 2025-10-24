from __future__ import annotations

from typing import TypeVar

K = TypeVar('K') # key type
V = TypeVar('V') # value type

class AttrDict(dict[K, V]):
    """
    A class to make dictionary accesible by attribute notation.
    """
    __getattr__ = dict.__getitem__ # forward attrs to items