"""Footprints, level tables and lattices, checked against the worked example of CLAUDE.md §8."""

import itertools

import pytest

from altiseek.geometry.levels import (
    Level,
    LevelTable,
    altitude_for_footprint_m,
    footprint_m,
    sim_gsd_mpp,
)

# Area 300 m x 300 m, 5 m cells, theta = 80 deg, R = 640.
WORKED = LevelTable(
    cell_m=5.0,
    grid_cells=60,
    levels=(
        Level(name="L1", footprint_cells=5, stride_cells=5),
        Level(name="L2", footprint_cells=10, stride_cells=5),
        Level(name="L3", footprint_cells=20, stride_cells=10),
        Level(name="L4", footprint_cells=30, stride_cells=15),
    ),
)


@pytest.mark.parametrize(
    ("index", "altitude_m", "gsd_cm", "views"),
    [(0, 14.9, 3.9, 144), (1, 29.8, 7.8, 121), (2, 59.6, 15.6, 25), (3, 89.4, 23.4, 9)],
)
def test_worked_example(index, altitude_m, gsd_cm, views):
    level = WORKED.levels[index]
    width = WORKED.footprint_m(level)
    assert altitude_for_footprint_m(width, 80.0) == pytest.approx(altitude_m, abs=0.05)
    assert 100 * sim_gsd_mpp(width, 640) == pytest.approx(gsd_cm, abs=0.05)
    assert WORKED.views_per_world(level) == views == len(WORKED.lattice(level))


def test_footprint_and_altitude_are_inverse():
    assert footprint_m(altitude_for_footprint_m(50.0, 80.0), 80.0) == pytest.approx(50.0)


def test_lattice_keeps_footprints_inside_and_reaches_the_far_edge():
    level = WORKED.levels[2]
    cells = WORKED.lattice(level)
    assert min(c for cell in cells for c in cell) == 0
    assert max(c for cell in cells for c in cell) + level.footprint_cells == WORKED.grid_cells
    assert WORKED.centre_m(level, (0, 0)) == (50.0, 50.0)


def test_lawnmower_visits_each_view_once_moving_one_stride_at_a_time():
    level = WORKED.levels[1]
    path = WORKED.lawnmower(level)
    assert len(path) == len(set(path)) == 121
    stride_m = level.stride_cells * WORKED.cell_m
    steps = {
        round(abs(ax - bx) + abs(ay - by), 6) for (ax, ay), (bx, by) in itertools.pairwise(path)
    }
    assert steps == {stride_m}


@pytest.mark.parametrize(
    ("levels", "message"),
    [
        ((Level(name="L", footprint_cells=7, stride_cells=5),), "does not step"),
        ((Level(name="L", footprint_cells=61, stride_cells=1),), "footprint must be"),
        (
            (
                Level(name="A", footprint_cells=10, stride_cells=5),
                Level(name="B", footprint_cells=5, stride_cells=5),
            ),
            "finest",
        ),
    ],
)
def test_invalid_tables_are_rejected(levels, message):
    with pytest.raises(ValueError, match=message):
        LevelTable(cell_m=5.0, grid_cells=60, levels=levels)
