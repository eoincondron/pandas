# Module for implementing nanops in numba

import numpy as np
import numba as nb

from numba.core.extending import overload


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
