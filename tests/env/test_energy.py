"""The energy model, checked against the coverage costs of the CLAUDE.md §8/§14 worked example."""

import pytest

from altiseek.env.energy import EnergyModel
from altiseek.geometry.levels import altitude_for_footprint_m

from ..geometry.test_levels import WORKED

ENERGY = EnergyModel(
    horizontal_speed_mps=10.0,
    vertical_speed_mps=3.0,
    dwell_s=2.0,
    climb_multiplier=1.5,
    battery_s=600.0,
)


def test_moves_charge_climbing_extra():
    assert ENERGY.move(30.0, 40.0, 0.0) == pytest.approx((5.0, 5.0))
    assert ENERGY.move(0.0, 0.0, 9.0) == pytest.approx((3.0, 4.5))
    assert ENERGY.move(0.0, 0.0, -9.0) == pytest.approx((3.0, 3.0))
    assert ENERGY.observe() == (2.0, 2.0)


def test_full_coverage_costs_of_the_worked_example():
    def cost(index, start_m):
        level = WORKED.levels[index]
        altitude = altitude_for_footprint_m(WORKED.footprint_m(level), 80.0)
        return ENERGY.tour(WORKED.lawnmower(level), altitude, start_m)[1]

    # "full L1 coverage costs about 650 s (infeasible) and full L4 coverage under about 120 s"
    l1 = cost(0, altitude_for_footprint_m(25.0, 80.0))  # starting at L1
    l4 = cost(3, altitude_for_footprint_m(150.0, 80.0))  # starting at L4
    assert l1 == pytest.approx(645.5, abs=0.1)
    assert l1 > ENERGY.battery_s
    assert l4 == pytest.approx(78.0, abs=0.1)
    assert cost(3, 0.0) == pytest.approx(78.0 + 89.4 / 3 * 1.5, abs=0.1)  # including take-off
