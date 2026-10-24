"""Pinhole geometry over a flat sea."""

import math

import numpy as np
import pytest

from altiseek.geometry.camera import (
    TiltedCamera,
    apply_homography,
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


TILTED = TiltedCamera.centred(3840, 2160, 3029.0, 55.0, 40.0)
NADIR = TiltedCamera.centred(3840, 2160, 3029.0, 90.0, 40.0)


def test_image_ground_round_trip():
    rng = np.random.default_rng(0)
    u, v = rng.uniform(0, 3839, 500), rng.uniform(0, 2159, 500)
    x, y, water = TILTED.image_to_ground(u, v)
    assert water.all()  # 55 deg pitch: the whole frame sees the sea
    u2, v2, ahead = TILTED.ground_to_image(x, y)
    assert ahead.all()
    assert np.allclose(u2, u, atol=1e-6)
    assert np.allclose(v2, v, atol=1e-6)


def test_nadir_projection_is_a_uniform_scaling_with_forward_up():
    u, v, ahead = NADIR.ground_to_image([0.0, 4.0], [0.0, 2.0])
    f_over_h = 3029.0 / 40.0
    assert ahead.all()
    assert np.allclose(u, [NADIR.cx_px, NADIR.cx_px + 4.0 * f_over_h])
    assert np.allclose(v, [NADIR.cy_px, NADIR.cy_px - 2.0 * f_over_h])
    assert np.allclose(NADIR.native_gsd_mpp([0, 3839], [0, 2159]), 40.0 / 3029.0)


def test_off_nadir_angles():
    assert TILTED.off_nadir_deg(TILTED.cx_px, TILTED.cy_px) == pytest.approx(35.0)
    corner = NADIR.off_nadir_deg(0, 0)
    assert corner == pytest.approx(math.degrees(math.atan(math.hypot(1919.5, 1079.5) / 3029.0)))
    sky = TiltedCamera.centred(3840, 2160, 3029.0, 10.0, 40.0)
    assert sky.off_nadir_deg(sky.cx_px, 0) > 90


def test_native_gsd_matches_the_scalar_formula_and_is_inf_at_the_sky():
    u, v = 3000.0, 400.0
    across, along = pixel_gsd_mpp(u - TILTED.cx_px, v - TILTED.cy_px, 55.0, 40.0, 3029.0)
    assert TILTED.native_gsd_mpp(u, v) == pytest.approx(max(across, along))
    sky = TiltedCamera.centred(3840, 2160, 3029.0, 10.0, 40.0)
    assert np.isinf(sky.native_gsd_mpp(sky.cx_px, 0))
    assert not sky.image_to_ground(sky.cx_px, 0)[2]


def test_downscaled_camera_matches_area_averaged_pixels():
    m = 3
    small = TILTED.downscaled(m)
    assert (small.width_px, small.height_px, small.focal_px) == (1280, 720, 3029.0 / 3)
    x, y = np.array([1.5, -7.0]), np.array([30.0, 55.0])
    u, v, _ = TILTED.ground_to_image(x, y)
    us, vs, _ = small.ground_to_image(x, y)
    assert np.allclose(us, (u + 0.5) / m - 0.5)
    assert np.allclose(vs, (v + 0.5) / m - 0.5)


def test_apply_homography_flags_points_behind():
    hom = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 1.0, -5.0]])
    _, _, positive = apply_homography(hom, [0.0, 0.0], [10.0, 1.0])
    assert positive.tolist() == [True, False]
