"""
taxi.py - Mini-Project 5: Taxi Route Revenue, Pricing & Fleet Planner.

Main pieces:
    Route                        - one route's daily passengers + fare
    market_equilibrium()         - solves the supply/demand system
    Forecaster (abstract parent) and its children:
        MovingAverageForecaster
        ExponentialSmoothingForecaster
        LinearTrendForecaster
        SeasonalNaiveForecaster (extension)
    walk_forward_mae()           - honest backtesting
    tune_alpha()                 - grid search for the smoothing constant
    vehicles_needed()            - fleet planning
    simulate_weekly_passengers() - extension: 60 days with a weekly pattern
"""

# statistics (built in): mean, variance, stdev.
import statistics

# abc: lets us write an abstract parent class (see population.py).
from abc import ABC, abstractmethod

# numpy: arrays and maths.
import numpy as np

# scipy.linalg.solve: solves linear systems A @ x = b.
from scipy.linalg import solve

# Our own error measure.
from src.metrics import mae


class Route:
    """A taxi route.

    Attributes:
        name       - e.g. "Kampala-Ntinda"
        passengers - NumPy array of passengers per day
        fare       - price of one trip in UGX
    """

    def __init__(self, name: str, passengers, fare: float) -> None:
        passengers = np.asarray(passengers, dtype=float)
        if len(passengers) == 0:
            raise ValueError(f"{name}: no passenger data")
        if np.any(passengers < 0):
            raise ValueError(f"{name}: passenger counts cannot be negative")
        if fare <= 0:
            raise ValueError(f"{name}: fare must be positive")
        self.name = name
        self.passengers = passengers
        self.fare = fare

    def __repr__(self) -> str:
        return f"Route('{self.name}', {len(self.passengers)} days, fare={self.fare:,.0f})"

    def daily_revenue(self) -> np.ndarray:
        """Revenue each day in UGX = passengers x fare."""
        return self.passengers * self.fare

    def total_revenue(self) -> float:
        """Revenue over all days, UGX."""
        return float(self.daily_revenue().sum())

    def describe(self) -> dict:
        """Mean, variance and standard deviation of daily passengers (statistics module)."""
        values = list(self.passengers)
        if len(values) < 2:
            raise ValueError("need at least 2 days for variance")
        return {
            "mean": statistics.mean(values),
            "variance": statistics.variance(values),
            "std_dev": statistics.stdev(values),
        }


def market_equilibrium(demand_intercept: float = 120, demand_slope: float = 0.02,
                       supply_intercept: float = 10, supply_slope: float = 0.03) -> tuple[float, float]:
    """Find the fare P and quantity Q where demand = supply.

        Demand: Q = 120 - 0.02P   ->   1Q + 0.02P = 120
        Supply: Q =  10 + 0.03P   ->   1Q - 0.03P = 10
    As a matrix system  [[1, 0.02], [1, -0.03]] @ [Q, P] = [120, 10].
    Returns (price_ugx, quantity).
    """
    A = np.array([[1, demand_slope],
                  [1, -supply_slope]])
    b = np.array([demand_intercept, supply_intercept])
    quantity, price = solve(A, b)
    return float(price), float(quantity)


# ===========================================================================
# Forecasters
# ===========================================================================

class Forecaster(ABC):
    """Parent class: every model learns from past days and predicts the next ones."""

    name = "Base"

    def __init__(self) -> None:
        self.history = np.array([])

    def fit(self, values) -> "Forecaster":
        """Remember the history. Children can do more work by overriding this."""
        values = np.asarray(values, dtype=float)
        if len(values) < self.minimum_history():
            raise ValueError(f"{self.name} needs at least {self.minimum_history()} values")
        self.history = values
        return self

    def minimum_history(self) -> int:
        """How many past days the model needs before it can predict."""
        return 1

    @abstractmethod
    def predict(self, horizon: int = 1) -> np.ndarray:
        """Forecast the next `horizon` days."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class MovingAverageForecaster(Forecaster):
    """(a) The original method: tomorrow = average of the last `window` days."""

    def __init__(self, window: int = 3) -> None:
        super().__init__()
        if window < 1:
            raise ValueError("window must be at least 1")
        self.window = window
        self.name = f"{window}-day moving average"

    def minimum_history(self) -> int:
        return self.window

    def predict(self, horizon: int = 1) -> np.ndarray:
        # A flat forecast: the same average for every future day.
        return np.full(horizon, self.history[-self.window:].mean())


class ExponentialSmoothingForecaster(Forecaster):
    """(b) Simple exponential smoothing.

    level = alpha x today + (1 - alpha) x previous level
    alpha near 1 -> follows the latest day closely (reacts fast)
    alpha near 0 -> changes slowly (smooths out noise)
    """

    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.name = f"Exp. smoothing (alpha={alpha:.2f})"

    def predict(self, horizon: int = 1) -> np.ndarray:
        level = self.history[0]                   # start at the first day
        for value in self.history[1:]:
            level = self.alpha * value + (1 - self.alpha) * level
        return np.full(horizon, level)


class LinearTrendForecaster(Forecaster):
    """(c) Fit a straight line through the days and extend it."""

    name = "Linear trend"

    def minimum_history(self) -> int:
        return 2

    def predict(self, horizon: int = 1) -> np.ndarray:
        days = np.arange(len(self.history))                  # 0, 1, 2, ...
        slope, intercept = np.polyfit(days, self.history, 1)
        future_days = np.arange(len(self.history), len(self.history) + horizon)
        return slope * future_days + intercept


class SeasonalNaiveForecaster(Forecaster):
    """Extension: tomorrow = the same weekday last week (7 days ago)."""

    def __init__(self, season_length: int = 7) -> None:
        super().__init__()
        self.season_length = season_length
        self.name = f"Seasonal naive ({season_length}-day)"

    def minimum_history(self) -> int:
        return self.season_length

    def predict(self, horizon: int = 1) -> np.ndarray:
        last_season = self.history[-self.season_length:]
        # Repeat last week's pattern as many times as needed, then cut to size.
        repeats = int(np.ceil(horizon / self.season_length))
        return np.tile(last_season, repeats)[:horizon]


# ===========================================================================
# Evaluation and planning
# ===========================================================================

def walk_forward_predictions(forecaster: Forecaster, values, first_day: int = 4) -> tuple:
    """Rolling-origin (walk-forward) backtest.

    To predict day 4 we fit on days 1-3 only; to predict day 5 we fit on
    days 1-4; ... and so on. The model never sees the day it is predicting,
    which is how it would be used in real life.
    Days are counted from 1. Returns (actual_array, predicted_array).
    """
    values = np.asarray(values, dtype=float)
    actual, predicted = [], []
    for day in range(first_day, len(values) + 1):
        past = values[:day - 1]                # everything before this day
        forecaster.fit(past)
        predicted.append(forecaster.predict(1)[0])
        actual.append(values[day - 1])
    return np.array(actual), np.array(predicted)


def walk_forward_mae(forecaster: Forecaster, values, first_day: int = 4) -> float:
    """MAE of the walk-forward backtest."""
    actual, predicted = walk_forward_predictions(forecaster, values, first_day)
    return mae(actual, predicted)


def tune_alpha(values, first_day: int = 4, alphas=None) -> tuple[float, float]:
    """Grid search: try many alphas, keep the one with the lowest backtest MAE.

    Returns (best_alpha, best_mae).
    """
    if alphas is None:
        alphas = np.round(np.arange(0.05, 1.0001, 0.05), 2)   # 0.05, 0.10, ..., 1.00
    best_alpha, best_error = None, float("inf")               # inf = "infinitely bad"
    for alpha in alphas:
        error = walk_forward_mae(ExponentialSmoothingForecaster(alpha), values, first_day)
        if error < best_error:
            best_alpha, best_error = float(alpha), error
    return best_alpha, best_error


def vehicles_needed(expected_passengers: float, trips_per_day: int = 8,
                    seats: int = 14, buffer: float = 0.15) -> int:
    """How many taxis to send out.

    One taxi carries at most trips_per_day x seats passengers a day
    (8 x 14 = 112). Add a 15% buffer for surprise demand, then round UP -
    half a taxi is not possible, and rounding down would leave people behind.
    At least one vehicle is always sent so the route stays served.
    """
    if expected_passengers < 0:
        raise ValueError("passengers cannot be negative")
    capacity_per_vehicle = trips_per_day * seats
    with_buffer = expected_passengers * (1 + buffer)
    return max(int(np.ceil(with_buffer / capacity_per_vehicle)), 1)


def simulate_weekly_passengers(days: int = 60, base: float = 50, seed: int = 7) -> np.ndarray:
    """Extension: fake daily passengers with a weekly pattern.

    Day 1 is a Monday. Fridays are busier (x1.4), Sundays quieter (x0.6),
    other days normal, plus random noise. Seeded for repeatability.
    """
    rng = np.random.default_rng(seed)
    #                 Mon  Tue  Wed  Thu  Fri  Sat  Sun
    weekday_factor = [1.0, 1.0, 1.0, 1.1, 1.4, 0.9, 0.6]
    pattern = np.array([weekday_factor[d % 7] for d in range(days)])
    noise = rng.normal(0, 3, days)
    return np.round(base * pattern + noise).clip(min=0)
