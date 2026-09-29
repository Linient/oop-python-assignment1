"""Tests for src/rainfall.py (Mini-Project 4). Run with: python -m pytest"""

import numpy as np
import pytest

from src.rainfall import (
    CropRule,
    Region,
    cosine_similarity,
    cosine_similarity_scipy,
    modality,
    rainy_seasons,
)

GULU = [8, 25, 75, 160, 190, 145, 170, 215, 175, 150, 60, 15]


def test_region_basic_numbers():
    region = Region("Gulu", GULU)
    assert region.annual_total() == sum(GULU)
    assert region.wettest_month() == "Aug"
    assert region.driest_month() == "Jan"


def test_region_needs_twelve_values():               # edge case: invalid input
    with pytest.raises(ValueError):
        Region("Short", [10, 20, 30])


def test_crop_rule_labels():
    maize = CropRule("maize", 100, 200)
    assert maize.classify(50) == "Drought risk"
    assert maize.classify(150) == "Good for maize"
    assert maize.classify(250) == "Waterlogging risk"
    assert maize.classify(100) == "Good for maize"   # edge case: exactly on the limit


def test_cosine_similarity_by_hand_and_against_scipy():
    # [1, 0] and [0, 1] point in completely different directions -> 0
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0)
    # A vector scaled up (3x) has the SAME shape -> similarity 1
    assert cosine_similarity([1, 2, 3], [3, 6, 9]) == pytest.approx(1)
    a, b = np.array(GULU), np.arange(1, 13)
    assert cosine_similarity(a, b) == pytest.approx(cosine_similarity_scipy(a, b))


def test_cosine_similarity_zero_vector():           # edge case: all zeros
    with pytest.raises(ValueError):
        cosine_similarity([0, 0, 0], [1, 2, 3])


def test_find_two_seasons():
    # A clearly two-humped year: wet in Apr and Oct.
    rain = [20, 40, 100, 200, 100, 30, 20, 40, 100, 200, 100, 30]
    assert rainy_seasons(Region("Two", rain)) == ["Apr", "Oct"]
    assert modality(Region("Two", rain)) == "bimodal"


def test_peak_in_december_is_found():
    # The wettest month is December - only found because we wrap the year.
    rain = [150, 100, 50, 20, 10, 10, 10, 10, 20, 50, 100, 200]
    assert rainy_seasons(Region("Dec", rain)) == ["Dec"]
