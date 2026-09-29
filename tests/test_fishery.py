"""Tests for src/fishery.py (Mini-Project 3). Run with: python -m pytest"""

import numpy as np
import pytest

from src.fishery import FishStock, PriceModel, RiskAssessor, fibonacci, weekly_revenue


def test_fibonacci_first_numbers():
    assert fibonacci(7) == [1, 1, 2, 3, 5, 8, 13]
    assert fibonacci(0) == []                        # edge case: zero


def test_one_logistic_step_by_hand():
    # N0 = 4000: growth = 0.4*4000*(1 - 0.4) = 960, harvest = 0.1*4000 = 400
    stock, harvest = FishStock(harvest_rate=0.1).simulate(weeks=1)
    assert stock[1] == pytest.approx(4000 + 960 - 400)
    assert harvest[0] == pytest.approx(400)


def test_stock_settles_at_theory():
    # Long-run stock should be K*(1 - h/r) = 10000*(1 - 0.1/0.4) = 7500
    model = FishStock(harvest_rate=0.1)
    stock, _ = model.simulate(weeks=200)
    assert stock[-1] == pytest.approx(model.long_run_stock(), rel=1e-3)


def test_msy_formula():
    assert FishStock().maximum_sustainable_yield() == pytest.approx(1000)   # 0.4*10000/4


def test_invalid_harvest_rate_rejected():            # edge case: invalid value
    with pytest.raises(ValueError):
        FishStock(harvest_rate=1.5)


def test_price_stays_in_bounds_and_is_reproducible():
    prices_a = PriceModel(seed=3).simulate(500)
    prices_b = PriceModel(seed=3).simulate(500)
    assert np.all((prices_a >= 9000) & (prices_a <= 16000))
    assert np.array_equal(prices_a, prices_b)        # same seed -> same prices


def test_closed_season_means_no_catch():
    _, harvest = FishStock().simulate(weeks=52, closed_weeks_per_year=8)
    assert np.all(harvest[:8] == 0)
    assert harvest[8] > 0


def test_revenue_and_risk_classes():
    assert weekly_revenue(2, 10_000) == 2 * 1000 * 10_000   # 2 t = 2,000 kg
    assessor = RiskAssessor()
    assert assessor.classify([100, 101, 99, 100]) == "Low"
    assert assessor.classify([50, 150, 60, 140]) == "High"


def test_value_at_risk_empty_input():                # edge case: empty input
    with pytest.raises(ValueError):
        RiskAssessor.value_at_risk([])
