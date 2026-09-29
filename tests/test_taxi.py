"""Tests for src/taxi.py (Mini-Project 5). Run with: python -m pytest"""

import numpy as np
import pytest

from src.taxi import (
    ExponentialSmoothingForecaster,
    LinearTrendForecaster,
    MovingAverageForecaster,
    Route,
    SeasonalNaiveForecaster,
    market_equilibrium,
    tune_alpha,
    vehicles_needed,
    walk_forward_predictions,
)


def test_route_revenue():
    route = Route("Test", [10, 20], fare=2000)
    assert list(route.daily_revenue()) == [20_000, 40_000]
    assert route.total_revenue() == 60_000


def test_route_rejects_bad_input():                 # edge cases: empty / zero fare
    with pytest.raises(ValueError):
        Route("Empty", [], fare=2000)
    with pytest.raises(ValueError):
        Route("Free", [10], fare=0)


def test_equilibrium_by_hand():
    # 120 - 0.02P = 10 + 0.03P  ->  110 = 0.05P  ->  P = 2200, Q = 120 - 44 = 76
    price, quantity = market_equilibrium()
    assert price == pytest.approx(2200)
    assert quantity == pytest.approx(76)


def test_moving_average():
    model = MovingAverageForecaster(3).fit([10, 20, 30, 40])
    assert model.predict(1)[0] == pytest.approx(30)   # (20 + 30 + 40) / 3


def test_exponential_smoothing_alpha_one_is_last_value():
    model = ExponentialSmoothingForecaster(alpha=1.0).fit([5, 9, 7])
    assert model.predict(1)[0] == pytest.approx(7)


def test_linear_trend():
    assert LinearTrendForecaster().fit([2, 4, 6]).predict(1)[0] == pytest.approx(8)


def test_moving_average_needs_enough_history():     # edge case: too little data
    with pytest.raises(ValueError):
        MovingAverageForecaster(3).fit([1, 2])


def test_walk_forward_covers_days_4_to_10():
    actual, predicted = walk_forward_predictions(MovingAverageForecaster(3), range(1, 11))
    assert list(actual) == [4, 5, 6, 7, 8, 9, 10]
    assert predicted[0] == pytest.approx(2)           # mean of days 1-3


def test_tune_alpha_prefers_high_alpha_for_a_step():
    # A series that jumps and stays - reacting fast (high alpha) is best.
    alpha, _ = tune_alpha([10, 10, 10, 50, 50, 50, 50, 50, 50, 50])
    assert alpha >= 0.9


def test_vehicles_needed():
    assert vehicles_needed(100) == 2          # 115 / 112 = 1.03 -> round up to 2
    assert vehicles_needed(0) == 1            # edge case: always keep one vehicle


def test_seasonal_naive_repeats_last_week():
    history = np.arange(14)                   # 0..13
    assert list(SeasonalNaiveForecaster(7).fit(history).predict(3)) == [7, 8, 9]
