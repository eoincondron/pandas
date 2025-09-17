# Module for implementing nanops in numba

from typing import Any, Callable, Tuple, TypeVar

import numpy as np
import numba as nb

from numba.core.extending import overload

T = TypeVar("T")
R = TypeVar("R")
F = TypeVar("F", bound=Callable[..., Any])


MIN_INT = np.iinfo(np.int64).min


def is_null(x):
    """
    Check if a value is considered null/NA.

    Parameters
    ----------
    x : scalar
        Value to check

    Returns
    -------
    bool
        True if value is null, False otherwise

    Notes
    -----
    This function is overloaded with specialized implementations for
    various numeric types via Numba's overload mechanism.
    """
    dtype = np.asarray(x).dtype
    if np.issubdtype(dtype, np.float64):
        return np.isnan(x)

    elif np.issubdtype(dtype, np.int64):
        return x == MIN_INT

    else:
        return False


@overload(is_null)
def jit_is_null(x):
    if isinstance(x, nb.types.Float) or isinstance(x, float):

        def is_null(x):
            return np.isnan(x)

        return is_null

    if isinstance(x, nb.types.Integer):

        def is_null(x):
            return x == MIN_INT

        return is_null

    elif isinstance(x, nb.types.Boolean):

        def is_null(x):
            return False

        return is_null

    else:
        return is_null


def _get_first_non_null(arr) -> Tuple[int, T]:
    """
    Find the first non-null value in an array. Return its location and value

    Parameters
    ----------
    arr : array-like
        Array to search for non-null values

    Returns
    -------
    tuple
        (index, value) of first non-null value, or (-1, np.nan) if all values are null

    Notes
    -----
    This function is JIT-compiled with Numba for performance.
    """
    for i, x in enumerate(arr):
        if not is_null(x):
            return i, x
    return -1, np.nan


@overload(_get_first_non_null, nogil=True)
def jit_get_first_non_null(arr):
    if isinstance(arr.dtype, nb.types.Float):

        return _get_first_non_null

    elif isinstance(arr.dtype, nb.types.Integer):

        def f(arr):
            for i, x in enumerate(arr):
                if not is_null(x):
                    return i, x
            return -1, MIN_INT

        return f

    elif isinstance(arr.dtype, nb.types.Boolean):

        def f(arr):
            return 0, arr[0]

        return f
