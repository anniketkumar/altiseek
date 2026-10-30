"""Independent basis and heading checks for the live flight interface."""

import math

import numpy as np
import pytest

from altiseek.geometry.frames import (
    enu_to_ned,
    enu_yaw_to_ned,
    ned_to_enu,
    ned_yaw_to_enu,
)


def test_cardinal_axes_and_altitude():
    assert np.array_equal(enu_to_ned(np.eye(3)), [[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    assert np.array_equal(ned_to_enu([10, 20, -30]), [20, 10, 30])


def test_roundtrip_preserves_distances_without_mutation():
    values = np.random.default_rng(17).normal(size=(2, 10, 3))
    saved = values.copy()
    converted = enu_to_ned(values)
    assert np.array_equal(ned_to_enu(converted), values)
    assert np.allclose(np.linalg.norm(converted, axis=-1), np.linalg.norm(values, axis=-1))
    assert np.array_equal(values, saved)


@pytest.mark.parametrize("yaw", [0, math.pi / 2, math.pi, -math.pi / 2, 7.3])
def test_heading_matches_transformed_direction(yaw):
    ned = enu_to_ned([math.cos(yaw), math.sin(yaw), 0])
    converted = enu_yaw_to_ned(yaw)
    assert math.cos(converted) == pytest.approx(ned[0])
    assert math.sin(converted) == pytest.approx(ned[1])
    roundtrip = ned_yaw_to_enu(converted)
    assert math.cos(roundtrip) == pytest.approx(math.cos(yaw))
    assert math.sin(roundtrip) == pytest.approx(math.sin(yaw))


@pytest.mark.parametrize("value", [1, [1, 2], [1, 2, float("nan")]])
def test_invalid_vectors_rejected(value):
    with pytest.raises(ValueError, match="finite vectors"):
        enu_to_ned(value)


def test_invalid_heading_rejected():
    with pytest.raises(ValueError, match="finite"):
        enu_yaw_to_ned(float("inf"))
