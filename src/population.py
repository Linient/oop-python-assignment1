"""
population.py - Mini-Project 1: UBOS District Population Forecaster.

This file holds the reusable logic (classes and functions). The notebook
project1_population.ipynb imports from here and does the analysis and plots.

Main pieces:
    DistrictPopulation      - one district's years + population numbers
    Forecaster              - an ABSTRACT parent class (a template) for models
    LinearTrendForecaster   - child class: fits a straight line
    CAGRForecaster          - child class: grows by a constant % every year
    FibonacciRatioForecaster- child class: the old "Fibonacci" method
    evaluate_models()       - trains every model, tests it, builds a score table
    bootstrap_intervals()   - extension: 95% prediction intervals

A note on TYPE HINTS (the ": float" and "-> float" you will see):
    def cagr(self) -> float:
The "-> float" part just documents "this function gives back a decimal number".
Python does not enforce it; it is there so a reader knows what to expect.
"""

# ---------------------------------------------------------------------------
# Imports - other code we borrow
# ---------------------------------------------------------------------------

# statistics is built into Python. It has mean, median, variance, stdev etc.
# The assignment asks us to use it AND numpy, then compare the two.
import statistics

# abc = "Abstract Base Classes". ABC lets us make a parent class that cannot be
# used directly; @abstractmethod marks methods every child class MUST write.
from abc import ABC, abstractmethod

# numpy: fast maths on arrays of numbers (see metrics.py for more).
import numpy as np

# pandas gives us the DataFrame - a table with named rows and columns.
# We use it only to print nice comparison tables.
import pandas as pd

# Our own error measures, written once in src/metrics.py.
from src.metrics import mae, mape, rmse


# ===========================================================================
# 1. The data class
# ===========================================================================

class DistrictPopulation:
    """Population estimates for one district.

    Attributes (the data each object carries):
        name        - district name, e.g. "Kampala"
        years       - NumPy array of years, e.g. [2015, 2016, ...]
        populations - NumPy array of population in THOUSANDS of people
    """

    def __init__(self, name: str, years, populations) -> None:
        """Store the data after checking it makes sense (validation)."""
        # Convert whatever we were given (usually lists) into NumPy arrays.
        years = np.asarray(years, dtype=int)
        populations = np.asarray(populations, dtype=float)

        # --- Validation: stop early with a clear message if data is wrong ---
        if len(years) != len(populations):
            raise ValueError(
                f"{name}: {len(years)} years but {len(populations)} population values"
            )
        if len(years) == 0:
            raise ValueError(f"{name}: no data given")
        if np.any(populations < 0):
            raise ValueError(f"{name}: population cannot be negative")

        self.name = name
        self.years = years
        self.populations = populations

    # --- "Dunder" (double-underscore) methods -----------------------------
    # Python calls these automatically in special situations.

    def __repr__(self) -> str:
        """What Python shows when you print the object or look at it in Jupyter."""
        return (
            f"DistrictPopulation('{self.name}', {self.years[0]}-{self.years[-1]}, "
            f"latest={self.populations[-1]:.0f}k)"
        )

    def __len__(self) -> int:
        """Lets us write len(district) to get the number of years of data."""
        return len(self.years)

    # --- Descriptive statistics -------------------------------------------

    def stats_with_statistics_module(self) -> dict:
        """Mean, median, variance and standard deviation using `statistics`.

        IMPORTANT: statistics.variance and statistics.stdev treat the data as a
        SAMPLE, so they divide by (n - 1). See stats_with_numpy for the contrast.
        """
        values = list(self.populations)   # statistics works on plain lists
        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "variance": statistics.variance(values),   # divides by n - 1
            "std_dev": statistics.stdev(values),       # divides by n - 1
        }

    def stats_with_numpy(self, ddof: int = 0) -> dict:
        """The same four numbers using NumPy.

        ddof = "delta degrees of freedom". NumPy divides by (n - ddof).
            ddof=0 (NumPy's default) -> divide by n     -> POPULATION variance
            ddof=1                   -> divide by n - 1 -> SAMPLE variance,
                                        the same as the statistics module.
        """
        values = self.populations
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "variance": float(np.var(values, ddof=ddof)),
            "std_dev": float(np.std(values, ddof=ddof)),
        }

    # --- Growth -------------------------------------------------------------

    def growth_rates(self) -> np.ndarray:
        """Year-on-year growth in %, e.g. 4.2 means the population grew 4.2%.

        Formula: (this year - last year) / last year * 100
        The result has one value fewer than the data (the first year has no
        "last year" to compare with).
        """
        this_year = self.populations[1:]    # every value except the first
        last_year = self.populations[:-1]   # every value except the last
        return (this_year - last_year) / last_year * 100

    def cagr(self) -> float:
        """Compound Annual Growth Rate, as a decimal (0.04 means 4% a year).

        It is the single steady growth rate that would take us from the first
        value to the last value in the same number of years:
            CAGR = (last / first) ** (1 / number_of_years) - 1
        """
        first = self.populations[0]
        last = self.populations[-1]
        number_of_years = len(self) - 1     # 10 values span 9 years of growth
        if first == 0 or number_of_years == 0:
            raise ValueError("need a non-zero start value and at least 2 years")
        return float((last / first) ** (1 / number_of_years) - 1)

    def split(self, last_training_year: int) -> tuple:
        """Split the data into a TRAINING part and a TEST part.

        We fit models on the training years only, then check their guesses
        against the test years they have never seen. Returns four arrays:
            train_years, train_values, test_years, test_values
        """
        is_train = self.years <= last_training_year   # True/False per year
        return (
            self.years[is_train], self.populations[is_train],
            self.years[~is_train], self.populations[~is_train],   # ~ means "not"
        )


# ===========================================================================
# 2. The forecasting models (inheritance)
# ===========================================================================

class Forecaster(ABC):
    """Parent class that every forecasting model inherits from.

    It fixes the "shape" that all models share:
        fit(years, values)  - learn from past data
        predict(horizon)    - guess the next `horizon` years
    Because it inherits from ABC and has abstract methods, you cannot create a
    plain Forecaster() - only its children, which fill in the details.
    """

    name = "Base forecaster"   # a class attribute: children replace it

    def __init__(self) -> None:
        self.is_fitted = False                 # becomes True after fit()
        self.fitted_values = np.array([])      # the model's version of the past

    @abstractmethod
    def fit(self, years, values) -> "Forecaster":
        """Learn from the training data. Must be written by each child."""

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        """Return forecasts for the next `horizon` years. Written by each child."""

    def _check_fitted(self) -> None:
        """Shared helper: complain if predict() is called before fit()."""
        if not self.is_fitted:
            raise RuntimeError(f"{self.name}: call fit() before predict()")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(fitted={self.is_fitted})"


class LinearTrendForecaster(Forecaster):
    """Model (a): draw the best straight line through the data.

    population = slope * year + intercept
    np.polyfit(x, y, 1) finds the slope and intercept (the "1" means a
    degree-1 polynomial, i.e. a straight line).
    """

    name = "Linear trend"

    def fit(self, years, values) -> "LinearTrendForecaster":
        years = np.asarray(years, dtype=float)
        values = np.asarray(values, dtype=float)
        if len(values) < 2:
            raise ValueError("need at least 2 points to fit a line")
        self.slope, self.intercept = np.polyfit(years, values, 1)
        self.last_year = years[-1]
        self.fitted_values = self.slope * years + self.intercept
        self.is_fitted = True
        return self   # returning self lets us write model.fit(...).predict(...)

    def predict(self, horizon: int) -> np.ndarray:
        self._check_fitted()
        # np.arange(1, horizon + 1) gives [1, 2, ..., horizon] years ahead
        future_years = self.last_year + np.arange(1, horizon + 1)
        return self.slope * future_years + self.intercept


class CAGRForecaster(Forecaster):
    """Model (b): exponential growth at a constant % per year (the CAGR).

    next value = last value * (1 + growth_rate)
    so h years ahead = last value * (1 + growth_rate) ** h
    """

    name = "Exponential (CAGR)"

    def fit(self, years, values) -> "CAGRForecaster":
        values = np.asarray(values, dtype=float)
        if len(values) < 2:
            raise ValueError("need at least 2 points to compute a growth rate")
        if values[0] <= 0:
            raise ValueError("first value must be positive for CAGR")
        number_of_years = len(values) - 1
        self.growth_rate = (values[-1] / values[0]) ** (1 / number_of_years) - 1
        self.last_value = values[-1]
        # Fitted line: start at the first value and grow steadily.
        steps = np.arange(len(values))          # 0, 1, 2, ...
        self.fitted_values = values[0] * (1 + self.growth_rate) ** steps
        self.is_fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        self._check_fitted()
        steps_ahead = np.arange(1, horizon + 1)
        return self.last_value * (1 + self.growth_rate) ** steps_ahead


def fibonacci_ratios(count: int) -> np.ndarray:
    """Successive Fibonacci ratios F(n+1)/F(n): 1/1, 2/1, 3/2, 5/3, 8/5, ...

    The sequence goes 1, 1, 2, 3, 5, 8, ... (each number = the two before it
    added together). The ratios settle towards the golden ratio, about 1.618.
    """
    if count < 0:
        raise ValueError("count cannot be negative")
    ratios = []
    previous, current = 1, 1          # the first two Fibonacci numbers
    for _ in range(count):            # "_" means "we don't need the loop number"
        ratios.append(current / previous)
        previous, current = current, previous + current
    return np.array(ratios)


class FibonacciRatioForecaster(Forecaster):
    """Model (c): the previous cohort's method.

    Start from the last known value and multiply by successive Fibonacci
    ratios: year 1 x 1.0, year 2 x 2.0, year 3 x 1.5, year 4 x 1.667 ...
    This is included so we can show, with numbers, why it is a poor model:
    it assumes the population grows about 60% every year.
    """

    name = "Fibonacci ratio"

    def fit(self, years, values) -> "FibonacciRatioForecaster":
        values = np.asarray(values, dtype=float)
        if len(values) < 1:
            raise ValueError("need at least 1 value")
        self.last_value = values[-1]
        # "Fitted" values: each year = previous actual year x a Fibonacci ratio.
        ratios = fibonacci_ratios(len(values) - 1)
        self.fitted_values = np.concatenate(([values[0]], values[:-1] * ratios))
        self.is_fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        self._check_fitted()
        # np.cumprod multiplies as it goes: [r1, r1*r2, r1*r2*r3, ...]
        return self.last_value * np.cumprod(fibonacci_ratios(horizon))


# ===========================================================================
# 3. Helper functions that use the classes
# ===========================================================================

def all_models() -> list:
    """A fresh (unfitted) copy of each of the three models."""
    return [LinearTrendForecaster(), CAGRForecaster(), FibonacciRatioForecaster()]


def evaluate_models(district: DistrictPopulation, last_training_year: int = 2021) -> pd.DataFrame:
    """Train every model on the early years, test on the later years.

    Returns a table with one row per model and columns MAE, RMSE, MAPE (%),
    sorted so the best model (lowest RMSE) is on top.
    """
    train_years, train_values, test_years, test_values = district.split(last_training_year)
    rows = []   # we will build a list of dictionaries, one per model
    for model in all_models():
        model.fit(train_years, train_values)
        guesses = model.predict(len(test_years))
        rows.append({
            "model": model.name,
            "MAE": mae(test_values, guesses),
            "RMSE": rmse(test_values, guesses),
            "MAPE (%)": mape(test_values, guesses),
        })
    table = pd.DataFrame(rows).set_index("model")
    return table.sort_values("RMSE")


def best_model_for(district: DistrictPopulation, last_training_year: int = 2021) -> Forecaster:
    """Pick the model with the lowest test RMSE and return a fresh copy of it."""
    table = evaluate_models(district, last_training_year)
    best_name = table.index[0]               # first row = lowest RMSE
    for model in all_models():
        if model.name == best_name:
            return model
    raise LookupError(best_name)             # should never happen


def extra_classrooms_needed(population_now_k: float, population_future_k: float,
                            school_age_share: float = 0.18,
                            pupils_per_classroom: int = 53) -> int:
    """How many NEW classrooms are needed for the population growth.

    Population is in thousands, so we multiply by 1,000 to get people.
    We round UP (np.ceil) because you cannot build 0.4 of a classroom -
    a partly-full extra class still needs a room. If the population shrinks,
    no new classrooms are needed, so the answer is never below 0.
    """
    extra_people = (population_future_k - population_now_k) * 1000
    extra_pupils = extra_people * school_age_share
    classrooms = int(np.ceil(extra_pupils / pupils_per_classroom))
    return max(classrooms, 0)


def bootstrap_intervals(model: Forecaster, actual_values, horizon: int,
                        n_resamples: int = 1000, seed: int = 42) -> tuple:
    """Extension: a 95% prediction interval using bootstrap of residuals.

    Idea in plain words:
      1. Residuals = actual - fitted: the model's past mistakes. We take away
         their average so the mistakes are centred on 0 (no built-in bias).
      2. Imagine a future where each year adds one past mistake, picked at
         random (with replacement), and the mistakes pile up year after year
         - so year 5 is less certain than year 1. Do this 1,000 times.
      3. The middle 95% of those 1,000 imagined futures is our interval.
    The model must already be fitted on `actual_values`.
    Returns (lower_bound_array, upper_bound_array).
    """
    residuals = np.asarray(actual_values, dtype=float) - model.fitted_values
    residuals = residuals - residuals.mean()
    forecast = model.predict(horizon)
    # default_rng(seed) makes a random-number generator. The fixed seed means
    # we get the SAME "random" numbers every run, so results are reproducible.
    rng = np.random.default_rng(seed)
    # Shape (n_resamples, horizon): 1,000 rows, one random residual per year.
    sampled = rng.choice(residuals, size=(n_resamples, horizon), replace=True)
    # np.cumsum along each row = running total, so mistakes accumulate.
    simulated_futures = forecast + np.cumsum(sampled, axis=1)
    # 2.5th and 97.5th percentiles of each column (axis=0 = down the rows).
    lower = np.percentile(simulated_futures, 2.5, axis=0)
    upper = np.percentile(simulated_futures, 97.5, axis=0)
    return lower, upper
