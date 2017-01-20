import numpy as np
from functools import wraps


def memo(func):
    cache = {}

    @wraps(func)
    def wrap(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]

    return wrap


def mask_random_entries(array, n):
    """takes 1-dimensional array returns same sized array with some
    entries randomly replaced with NaN for float arrays or with -1 for
    integer and boolean arrays
    """
    indices = np.random.choice(range(len(array)), n)
    array = np.copy(array)
    if np.issubdtype(array.dtype, np.integer):
        array[indices] = -1
    elif np.issubdtype(array.dtype, np.floating):
        array[indices] = np.NaN
    else:
        raise TypeError('only floaty and integer-y types '
                        + 'are supported but given %s' % array.dtype)
    return array


def missing_mask(array):
    """takes an array with missing values and returns boolean mask
    indicating which entries are missing

    missing values must be marked with np.NaN for float
    arrays and with -1 for bool or integer arrays - the
    same convention used in 'mask_random_entries'
    """
    if np.issubdtype(array.dtype, np.floating):
        return np.isnan(array)
    else:
        return array == -1


def mask_missing(values, fill_value=0):
    mask = missing_mask(values)
    new_values = np.array([v if not m else fill_value
                           for v, m in zip(values, mask)])

    return np.ma.masked_array(new_values, mask, fill_value=fill_value)


def missing_indices(values):
    return np.where(missing_mask(values))[0]


def is_floaty(array):
    return np.issubdtype(array.dtype, np.floating)


def test_missing_mask_returns_indices_of_missing_elements():
    array = np.array([-1, 2, 0, 3, -1])
    expected_result = np.array([True, False, False, False, True])
    assert np.alltrue(missing_mask(array) == expected_result)
