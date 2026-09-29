"""Tests for src/microgrid.py (Mini-Project 2). Run with: python -m pytest"""

import numpy as np
import pytest

from src.microgrid import (
    HybridMicroGrid,
    MicroGrid,
    generate_demand_csv,
    load_demand_csv,
    read_positive_number,
)


def test_solve_day_matches_hand_calculation():
    # Pick x = 10, y = 20: then D1 = 3*10 + 2*20 = 70 and D2 = 4*10 + 20 = 60.
    x, y = MicroGrid().solve_day(70, 60)
    assert x == pytest.approx(10)
    assert y == pytest.approx(20)


def test_determinant_is_minus_five():
    # By hand: 3*1 - 2*4 = -5
    assert MicroGrid().determinant() == pytest.approx(-5)


def test_loop_and_vectorised_give_same_answers():
    grid = MicroGrid()
    demands = np.array([[70, 100, 120], [60, 90, 110]])
    assert np.allclose(grid.solve_days_loop(demands), grid.solve_days_vectorised(demands))


def test_negative_demand_rejected():                # edge case: invalid value
    with pytest.raises(ValueError):
        MicroGrid().solve_day(-1, 50)


def test_infeasible_day_detected_and_fixed_by_nnls():
    grid = MicroGrid()
    # D1 = 100, D2 = 30 gives y = (4*100 - 3*30)/5 = 62 but x = (2*30 - 100)/5 = -8
    usage = grid.solve_days_vectorised(np.array([[100], [30]]))
    assert grid.infeasible_days(usage)[0]
    fixed, _ = grid.solve_day_non_negative([100, 30])
    assert np.all(fixed >= 0)


def test_singular_system_rejected():
    # Second row is 2 x the first row -> determinant 0 -> no unique answer.
    grid = MicroGrid([[1, 2], [2, 4]])
    assert not grid.is_well_posed()
    with pytest.raises(np.linalg.LinAlgError):
        grid.solve_day(10, 20)


def test_read_positive_number_reprompts_until_valid():
    # A fake "user" who types: nothing, then "abc", then "-5", then "42".
    answers = iter(["", "abc", "-5", "42"])
    value = read_positive_number("D1: ", input_function=lambda prompt: next(answers))
    assert value == 42


def test_csv_round_trip(tmp_path):
    # tmp_path is a temporary folder pytest gives us, deleted afterwards.
    path = generate_demand_csv(tmp_path / "demand.csv", days=30, seed=1)
    demands = load_demand_csv(path)
    assert demands.shape == (2, 30)
    assert np.all(demands > 0)


def test_hybrid_grid_solves_three_sources():
    # x=10, y=20, z=5 -> D1 = 70, D2 = 60, D3 = 35
    solar, battery, diesel = HybridMicroGrid().solve_day(70, 60, 35)
    assert (solar, battery, diesel) == pytest.approx((10, 20, 5))


def test_daily_cost():
    assert MicroGrid.daily_cost(10, 2) == 10 * 150 + 2 * 450
