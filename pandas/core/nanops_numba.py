from typing import Any, Callable, Tuple, TypeVar, Literal, Optional
from functools import wraps

import numpy as np
import numba as nb

from pandas import to_datetime, to_timedelta
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_integer_dtype,
    is_timedelta64_dtype,
    is_bool_dtype,
)

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


def get_reduction_out_type_for_op_and_type(
    dtype, op: Literal["count", "min", "max", "sum", "sum_square", "mean"]
):
    if op == "count":
        return np.int64
    elif op in ("min", "max"):
        return dtype
    elif op == "sum":
        return np.float64 if dtype.kind in "fmM" else np.int64
    elif op in ["mean", "sum_square"]:
        # always use floats for mean/var/std calculation to avoid overflow
        return np.float64
    else:
        raise ValueError(
            'op must be one of ["count", "min", "max", "sum", "sum_square"]'
        )


def _nullify_below_mincount(result, count, min_count):
    if is_integer_dtype(result):
        null = MIN_INT
    else:
        null = np.nan

    null_out = count < min_count

    if np.ndim(result) == 0:
        if null_out:
            result = null
    else:
        result[null_out] = null

    return result


def nb_reduce(
    op: Literal["count", "min", "max", "sum", "sum_square"],
    values: np.ndarray,
    axis: Optional[int] = None,
    skipna: bool = True,
    min_count: int = 0,
    mask: Optional[np.ndarray] = None,
    multi_threading: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a reduction operation to a numpy array using Numba-accelerated functions.

    Parameters
    ----------
    op : {"count", "min", "max", "sum", "sum_square"}
        The reduction operation to perform
    arr : np.ndarray
        Input array to reduce (1D or 2D)
    axis : int, optional
        Axis along which to perform the reduction. If None, reduces over all elements.
        For 2D arrays, axis=0 reduces along rows, axis=1 reduces along columns.
    skipna : bool, default True
        Whether to skip NA/null values during reduction
    multi_threading : bool, default True
        Whether to use parallel processing by splitting array into chunks (1D only)

    Returns
    -------
    tuple[float | int | np.ndarray, int | np.ndarray]
        Two-element tuple containing:
        - Reduction result (scalar for 1D or axis=None, array for 2D with specified axis)
        - Count of non-null values processed (scalar or array matching result shape)

    Notes
    -----
    This function provides high-performance reduction operations by leveraging
    Numba's JIT compilation and optional parallel processing. For 1D arrays with
    multi_threading=True, the array is split into chunks processed in parallel.

    Supports arrays up to 2 dimensions:
    - 1D arrays: reduces to scalar
    - 2D arrays: reduces along specified axis or to scalar if axis=None

    The function handles null values according to the skipna parameter:
    - If skipna=True: null values are ignored in the reduction
    - If skipna=False: any null value causes early termination

    For integer arrays, MIN_INT is used as the null sentinel value.
    For float arrays, NaN is used as the null value.

    Examples
    --------
    >>> import numpy as np
    >>> # 1D array reduction
    >>> arr = np.array([1.0, 2.0, np.nan, 4.0])
    >>> result, count = nb_reduce("sum", arr, skipna=True)
    >>> result, count
    (7.0, 3)

    >>> # 2D array reduction along axis
    >>> arr_2d = np.array([[1.0, 2.0], [3.0, np.nan]])
    >>> result, count = nb_reduce("sum", arr_2d, axis=0, skipna=True)
    >>> result, count
    (array([4.0, 2.0]), array([2, 1]))
    """
    ndim = np.ndim(values)
    if not (axis is None or axis < ndim):
        raise ValueError(f"axis {axis} out-of-bounds for array of dimension {ndim}")

    if ndim == 1:
        if multi_threading:
            # TODO: be smarter about this choice. numba is handling the distribution of the compute
            # so don't need to worry about setting it too high
            max_n_threads = 6
            arr_list = NumbaList(np.array_split(values, max_n_threads))  # type: ignore
        else:
            arr_list = None

    elif ndim == 2:
        if axis is None:
            # pass 1-D back through nb_reduce for best performance
            return nb_reduce(
                op,
                values.ravel(),
                skipna=skipna,
                axis=0,
                multi_threading=multi_threading,
            )

        if axis == 1:
            arr_list = values
        else:
            arr_list = values.T
    else:
        raise ValueError("Only arrays of 1 or 2 dimensions are supported")

    reduce_op = "sum" if op == "mean" else op
    reduce_func = getattr(NumbaReductionOps, reduce_op)

    kwargs = {
        "reduce_func": reduce_func,
        "skipna": skipna,
        "find_initial_value": "sum" not in reduce_op,
    }

    if arr_list is not None:
        out_type = get_reduction_out_type_for_op_and_type(values.dtype, op)
        target = np.zeros(len(arr_list), dtype=out_type)  # type: ignore
        result, count = _nb_reduce_2d(
            target=target,
            arr_list=arr_list,
            **kwargs,
        )
        if ndim == 1 or axis is None:
            final_reduction = (
                NumbaReductionOps.sum if op == "sum_square" else reduce_func
            )
            result, _ = _nb_reduce_single_arr(final_reduction, result, skipna=skipna)
            count = count.sum()
    else:
        result, count = _nb_reduce_single_arr(arr=values, **kwargs)

    if skipna and min_count > 0:
        result = _nullify_below_mincount(result, count, min_count)

    result = np.asarray(result)  # convert scalars to zerodi array to simplify typing
    count = np.asarray(count)

    return result, count
