"""Pinhole geometry over a flat sea."""

import math

import pytest

from altiseek.geometry.camera import (
    focal_length_px,
    hfov_deg,
    horizon_dip_deg,
    horizon_offset_px,
    nadir_gsd_mpp,
    pixel_gsd_mpp,
)


def test_focal_length_and_hfov_are_inverse():
    f = focal_length_px(65.5, 3840)
    assert f == pytest.approx(3840 / (2 * math.tan(math.radians(32.75))))
    assert hfov_deg(f, 3840) == pytest.approx(65.5)


def test_worked_example_gsd_from_claude_md():
    # CLAUDE.md §8: theta = 80 deg, R = 640, footprint 25 m at about 14.9 m -> about 3.9 cm/px.
    f = focal_length_px(80.0, 640)
    altitude_m = 25.0 / (2 * math.tan(math.radians(40.0)))
    assert altitude_m == pytest.approx(14.9, abs=0.05)
    assert nadir_gsd_mpp(altitude_m, f) == pytest.approx(25.0 / 640)


@pytest.mark.parametrize(("x", "y"), [(0, 0), (1500, -900), (-1900, 1000)])
def test_nadir_gsd_is_uniform(x, y):
    across, along = pixel_gsd_mpp(x, y, 90.0, 50.0, 3000.0)
    assert across == pytest.approx(50.0 / 3000.0)
    assert along == pytest.approx(50.0 / 3000.0)


def test_tilted_gsd_matches_finite_differences():
    # Ground point of a pixel for a camera at height h pitched down by a, x right, y down.
    h, a, f = 40.0, math.radians(60.0), 3000.0

    def ground(x, y):
        depth = y * math.cos(a) + f * math.sin(a)
        return h * x / depth, h * (f * math.cos(a) - y * math.sin(a)) / depth

    x, y = 700.0, -400.0
    across, along = pixel_gsd_mpp(x, y, 60.0, h, f)
    assert across == pytest.approx(math.dist(ground(x + 0.5, y), ground(x - 0.5, y)), rel=1e-6)
    assert along == pytest.approx(math.dist(ground(x, y + 0.5), ground(x, y - 0.5)), rel=1e-6)


def test_gsd_coarsens_towards_the_horizon():
    top = pixel_gsd_mpp(0, -1080, 60.0, 50.0, 2985.0)
    bottom = pixel_gsd_mpp(0, 1080, 60.0, 50.0, 2985.0)
    assert top[0] > bottom[0]
    assert top[1] > bottom[1]


def test_pixel_above_the_horizon_is_an_error():
    with pytest.raises(ValueError, match="does not see the water"):
        pixel_gsd_mpp(0, -2000, 10.0, 50.0, 3000.0)


def test_horizon_dip():
    assert horizon_dip_deg(0.0) == 0.0
    # Small-angle form sqrt(2h/R) for h = 50 m: about 0.227 deg.
    assert horizon_dip_deg(50.0) == pytest.approx(
        math.degrees(math.sqrt(2 * 50 / 6_371_000)), rel=1e-4
    )


def test_horizon_offset():
    assert horizon_offset_px(0.0, 0.0, 3000.0) == 0.0
    assert horizon_offset_px(10.0, 0.0, 3000.0) == pytest.approx(
        3000.0 * math.tan(math.radians(10))
    )
    # The dip lowers the horizon, i.e. reduces its offset above the centre.
    assert horizon_offset_px(10.0, 100.0, 3000.0) < horizon_offset_px(10.0, 0.0, 3000.0)
