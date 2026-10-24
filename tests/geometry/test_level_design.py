"""Level design on a synthetic recall curve."""

import math

import pytest

from altiseek.env.energy import EnergyModel
from altiseek.geometry.level_design import (
    candidates,
    choose_battery,
    footprint_options,
    recall_at,
    unmet_recall_rules,
)

# Recall falls from 0.95 at 2 cm/px to 0.05 at 30 cm/px (logistic in log-GSD).
CURVE = [
    (g, 0.05 + 0.9 / (1 + math.exp((math.log(g) - math.log(0.10)) * 3.0)))
    for g in (0.02, 0.03, 0.04, 0.05, 0.065, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25, 0.30)
]
ENERGY = EnergyModel(
    horizontal_speed_mps=10.0,
    vertical_speed_mps=3.0,
    dwell_s=2.0,
    climb_multiplier=1.5,
    battery_s=600.0,
)


def test_recall_interpolates_in_log_gsd_and_never_extrapolates():
    assert recall_at(0.10, CURVE) == pytest.approx(0.5)
    assert recall_at(0.02, CURVE) == pytest.approx(CURVE[0][1])
    assert recall_at(0.01, CURVE) is None
    assert recall_at(0.31, CURVE) is None


def test_footprint_options_tile_the_grid_exactly():
    options = footprint_options(60)
    assert {(5, 5), (10, 5), (30, 15)} <= set(options)
    assert (7, 7) not in options  # 60 - 7 = 53 is not a multiple of 7
    assert all((60 - n) % s == 0 for n, s in options)


def test_battery_choice():
    assert choose_battery(650.0, 80.0, 600.0, 0.25) == 600.0
    assert choose_battery(650.0, 200.0, 600.0, 0.25) is None  # 800 > 650
    assert choose_battery(500.0, 80.0, 600.0, 0.25) < 500.0  # just below the finest cost


def test_candidates_respect_the_design_rules():
    found = candidates(
        area_m=300.0,
        cell_options_m=[5.0, 10.0],
        level_counts=[3, 4, 5],
        curve=CURVE,
        resolution_px=640,
        hfov_deg=80.0,
        energy=ENERGY,
        finest_min=0.8,
        coarsest_max=0.3,
        coarse_share=0.25,
    )
    assert found
    for c in found[:20]:
        assert c.recall[0] >= 0.8
        assert c.recall[-1] <= 0.3
        assert all(b < a for a, b in zip(c.recall, c.recall[1:], strict=False))
        assert c.coverage_energy_s[0] > c.battery_s >= c.coverage_energy_s[-1] / 0.25
        assert 3 <= len(c.table.levels) <= 5
    steps = [c.largest_recall_step for c in found]
    assert steps == sorted(steps)


def test_a_flat_short_curve_names_both_unmet_rules():
    flat = [(0.02, 0.46), (0.05, 0.62), (0.08, 0.47), (0.13, 0.49)]
    rules = {"finest_min": 0.8, "coarsest_max": 0.3}
    assert not candidates(
        area_m=300.0,
        cell_options_m=[5.0],
        level_counts=[3],
        curve=flat,
        resolution_px=640,
        hfov_deg=80.0,
        energy=ENERGY,
        coarse_share=0.25,
        **rules,
    )
    finest, coarsest = unmet_recall_rules(flat, **rules)
    assert "0.62 (at 5 cm/px)" in finest
    assert "0.46 (at 2 cm/px)" in coarsest
    assert "ends at 13 cm/px" in coarsest
    assert unmet_recall_rules(CURVE, **rules) == []
