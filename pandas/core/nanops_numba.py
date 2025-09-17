# Module for implementing nanops in numba

from typing import Any, Callable, Tuple, TypeVar

import numpy as np
import numba as nb

from numba.core.extending import overload
from numba.typed import List as NumbaList

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


def _numba_staticmethod(func):
    return staticmethod(nb.njit(nogil=True)(func))


class NumbaReductionOps:
    """
    Collection of numba implementations of scalar reduction ops"""

    @_numba_staticmethod
    def count(x, y):
        return x + 1

    @_numba_staticmethod
    def min(x, y):
        return x if x <= y else y

    @_numba_staticmethod
    def max(x, y):
        return x if x >= y else y

    @_numba_staticmethod
    def sum(x, y):
        return x + y

    @_numba_staticmethod
    def sum_square(x, y):
        return x + y**2


@nb.njit(nogil=True)
def _nb_reduce_single_arr(
    reduce_func: Callable,
    arr: np.ndarray,
    skipna: bool = True,
    find_initial_value: bool = True,
) -> Tuple[float | int, int]:
    """
    Apply a reduction function to a numpy array, with NA/null handling.
    Returns the count of non-nulls as well as the reduction.

    Parameters
    ----------
    reduce_func : callable
        Function that combines two values (e.g., min, max, sum)
    arr : array-like
        Array to reduce
    skipna : bool, default True
        Whether to skip NA/null values
    initial_value:
        Initial_value for each reduction. Should be 0 or None.
        If None, we find the first_non_null value before commencing the reduction

    Returns
    -------
    scalar
        Result of the reduction operation

    Notes
    -----
    This function is JIT-compiled with Numba for performance.
    """
    if not find_initial_value:
        initial_value = 0
        initial_loc = -1
        count = 0

    elif skipna:
        # find the initial non-null value to pass through the reduction
        initial_loc, initial_value = _get_first_non_null(arr)
        if initial_loc == -1:  # all null
            return arr[0], 0
        else:
            count = 1

    else:
        # start at the start since we either have all non-null values
        # or we return the null value
        initial_loc, initial_value = 0, arr[0]
        if is_null(initial_value):
            return initial_value, 0
        else:
            count = 1

    start = initial_loc + 1
    result = initial_value

    for x in arr[start:]:
        if is_null(x):
            if skipna:
                continue
            else:
                # here the count is the number elements before the first null which might be of use
                return x, count

        result = reduce_func(result, x)
        count += 1

    return result, count


@nb.njit(nogil=True, parallel=True)
def _nb_reduce_2d(
    reduce_func: Callable,
    arr_list: NumbaList[np.ndarray] | np.ndarray,  # type: ignore
    target=np.ndarray,
    skipna: bool = True,
    find_initial_value: bool = True,
) -> Tuple[float | int, int]:
    counts = np.zeros(len(arr_list), dtype=np.int64)
    for i in nb.prange(len(arr_list)):
        arr = arr_list[i]
        target[i], counts[i] = _nb_reduce_single_arr(
            reduce_func, arr, skipna=skipna, find_initial_value=find_initial_value
        )

    return target, counts
