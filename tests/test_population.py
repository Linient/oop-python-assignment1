"""
Tests for src/population.py (Mini-Project 1).

How pytest works: it finds every function whose name starts with "test_",
runs it, and reports PASS if nothing goes wrong. `assert something` fails the
test if `something` is False. Run all tests from the project folder with:
    python -m pytest
"""

# pytest is the testing tool. We use pytest.raises (check an error happens)
# and pytest.approx (compare decimals without tiny rounding problems).
import pytest

from src.population import (
    CAGRForecaster,
    DistrictPopulation,
    FibonacciRatioForecaster,
    LinearTrendForecaster,
    extra_classrooms_needed,
    fibonacci_ratios,
)

YEARS = list(range(2015, 2025))   # 2015, 2016, ..., 2024


def test_cagr_matches_hand_calculation():
    # Hand calculation: 100 -> 121 over 2 years is 10% a year (1.1 * 1.1 = 1.21).
    district = DistrictPopulation("Test", [2020, 2021, 2022], [100, 110, 121])
    assert district.cagr() == pytest.approx(0.10)


def test_len_and_growth_rates():
    district = DistrictPopulation("Test", [2020, 2021, 2022], [100, 110, 121])
    assert len(district) == 3
    assert district.growth_rates() == pytest.approx([10.0, 10.0])


def test_rejects_negative_population():            # edge case: invalid value
    with pytest.raises(ValueError):
        DistrictPopulation("Bad", [2020, 2021], [100, -5])


def test_rejects_unequal_lengths():                 # edge case: mismatched input
    with pytest.raises(ValueError):
        DistrictPopulation("Bad", [2020, 2021, 2022], [100, 110])


def test_rejects_empty_input():                     # edge case: empty input
    with pytest.raises(ValueError):
        DistrictPopulation("Empty", [], [])


def test_linear_trend_extends_a_perfect_line():
    # Data is exactly +10 per year, so the next two years must be 150 and 160.
    model = LinearTrendForecaster().fit([1, 2, 3, 4], [110, 120, 130, 140])
    assert model.predict(2) == pytest.approx([150, 160])


def test_cagr_forecaster_grows_at_constant_rate():
    model = CAGRForecaster().fit([1, 2, 3], [100, 110, 121])
    assert model.predict(1) == pytest.approx([133.1])   # 121 * 1.1


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        LinearTrendForecaster().predict(3)


def test_fibonacci_ratios_and_model():
    assert fibonacci_ratios(4) == pytest.approx([1, 2, 1.5, 5 / 3])
    model = FibonacciRatioForecaster().fit([1, 2], [50, 100])
    assert model.predict(2) == pytest.approx([100, 200])   # x1 then x2


def test_statistics_variance_equals_numpy_with_ddof_1():
    district = DistrictPopulation("Test", YEARS, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert district.stats_with_statistics_module()["variance"] == pytest.approx(
        district.stats_with_numpy(ddof=1)["variance"])


def test_classrooms_rounds_up_and_never_negative():
    # 1,000 more people * 18% = 180 pupils / 53 = 3.4 -> 4 classrooms
    assert extra_classrooms_needed(100, 101) == 4
    assert extra_classrooms_needed(100, 100) == 0        # edge case: zero growth
    assert extra_classrooms_needed(100, 90) == 0         # shrinking population
