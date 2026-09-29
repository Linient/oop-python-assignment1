"""
metrics.py - error measures used to judge how good a forecast is.

Both Mini-Project 1 (population) and Mini-Project 5 (taxi) need to compare
forecasts against real values, so the functions live here once and both
projects import them. Writing them once avoids duplicated code.

Every function takes two lists/arrays of the same length:
    actual    - the values that really happened
    predicted - the values our model guessed
"""

# numpy is the main Python library for working with arrays (lists of numbers)
# and doing maths on a whole array at once. "np" is the usual short name.
import numpy as np


def _to_arrays(actual, predicted) -> tuple[np.ndarray, np.ndarray]:
    """Turn both inputs into NumPy arrays and check they can be compared.

    The underscore at the start of the name is a Python convention meaning
    "this is a helper for this file only - other files should not call it".
    """
    # np.asarray converts a list (or an array) into a NumPy array of decimals.
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    # We can only compare value-by-value if both have the same number of values.
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")
    if len(actual) == 0:
        raise ValueError("cannot measure error on empty input")
    return actual, predicted


def mae(actual, predicted) -> float:
    """Mean Absolute Error: on average, how far off are we (in the data's units)?

    Example: MAE = 12 for population in thousands means "about 12,000 people off".
    """
    actual, predicted = _to_arrays(actual, predicted)
    errors = actual - predicted              # one error per value (can be + or -)
    return float(np.mean(np.abs(errors)))    # abs removes the sign, mean averages


def rmse(actual, predicted) -> float:
    """Root Mean Squared Error: like MAE but punishes big mistakes more.

    Squaring makes a 10-unit miss count 100, but a 2-unit miss only 4.
    Taking the square root at the end brings the answer back to the data's units.
    """
    actual, predicted = _to_arrays(actual, predicted)
    squared_errors = (actual - predicted) ** 2   # ** means "to the power of"
    return float(np.sqrt(np.mean(squared_errors)))


def mape(actual, predicted) -> float:
    """Mean Absolute Percentage Error: average error as a % of the real value.

    Useful for comparing districts of different sizes, because 5% means the
    same thing for Kampala and for Gulu. It breaks if an actual value is 0
    (we would divide by zero), so we refuse that case.
    """
    actual, predicted = _to_arrays(actual, predicted)
    if np.any(actual == 0):
        raise ValueError("MAPE is undefined when an actual value is 0")
    percentage_errors = np.abs((actual - predicted) / actual) * 100
    return float(np.mean(percentage_errors))
