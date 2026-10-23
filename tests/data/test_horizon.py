"""Horizon detection on synthetic sky-over-sea images."""

import math

import numpy as np
import pytest

from altiseek.data.horizon import _dilate, detect_horizon, theil_sen

N_STRIPS = 32


def scene(row_px, roll_deg, height=540, width=960, seed=0):
    """Sky (bright) above a straight horizon through (centre column, row_px), sea (dark) below."""
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[0:height, 0:width]
    line = row_px + math.tan(math.radians(roll_deg)) * (cols - (width - 1) / 2)
    img = np.where(rows < line, 190.0, 70.0) + rng.normal(0, 6, (height, width))
    return np.clip(np.repeat(img[..., None], 3, axis=2), 0, 255).astype(np.uint8)


@pytest.mark.parametrize(("row", "roll"), [(120.0, 0.0), (300.0, 1.5), (60.0, -2.0)])
def test_finds_a_straight_horizon(row, roll):
    found = detect_horizon(scene(row, roll))
    assert found is not None
    assert found.row_px == pytest.approx(row, abs=2.0)
    assert found.roll_deg == pytest.approx(roll, abs=0.3)
    assert found.n_inliers == N_STRIPS
    assert found.inlier_fraction == 1.0


def test_blackened_region_on_the_horizon_is_ignored():
    img = scene(150.0, 0.5)
    img[100:200, 300:420] = 0  # a blackened ignore region straddling the horizon
    found = detect_horizon(img)
    assert found is not None
    assert found.row_px == pytest.approx(150.0, abs=2.0)
    assert found.inlier_fraction == 1.0
    assert found.n_inliers < N_STRIPS


def test_horizon_mostly_hidden_still_fits_on_the_visible_strips():
    img = scene(150.0, 0.0)
    img[110:200, 330:] = 0  # like ODv2 frame 11352: two thirds of the horizon blackened
    found = detect_horizon(img)
    assert found is not None
    assert found.row_px == pytest.approx(150.0, abs=2.0)
    assert found.inlier_fraction == 1.0
    assert 5 <= found.n_inliers <= 12


def test_jpeg_halo_around_a_wide_black_box_does_not_win():
    # Like ODv2 frame 12269: a box over most of the horizon, with a dark ringing band below it
    # that is brighter than the black threshold but darker than the sea.
    img = scene(150.0, 0.0)
    img[110:190, 330:] = 0
    img[190:194, 330:] = 40
    found = detect_horizon(img)
    assert found is not None
    assert found.row_px == pytest.approx(150.0, abs=2.0)


def test_dilate_grows_a_square_neighbourhood():
    mask = np.zeros((7, 7), dtype=bool)
    mask[3, 3] = True
    grown = _dilate(mask, 2)
    assert grown[1:6, 1:6].all()
    assert grown.sum() == 25
    assert (_dilate(mask, 0) == mask).all()


def test_bright_blob_in_the_sea_is_outvoted():
    img = scene(150.0, 0.5)
    img[350:380, 600:700] = 250  # a bright boat in the sea
    found = detect_horizon(img)
    assert found is not None
    assert found.row_px == pytest.approx(150.0, abs=2.0)
    assert found.inlier_fraction >= 0.8


def test_frame_without_horizon_has_few_inliers():
    rng = np.random.default_rng(1)
    noise = rng.normal(100, 6, (540, 960, 1)).repeat(3, axis=2).clip(0, 255).astype(np.uint8)
    found = detect_horizon(noise)
    assert found is not None
    assert found.inlier_fraction < 0.5


def test_fully_black_frame_has_no_horizon():
    assert detect_horizon(np.zeros((540, 960, 3), dtype=np.uint8)) is None


def test_theil_sen_ignores_outliers():
    x = np.arange(20.0)
    y = 3.0 + 0.5 * x
    y[[2, 7, 15]] += [40.0, -25.0, 60.0]
    intercept, slope = theil_sen(x, y)
    assert slope == pytest.approx(0.5)
    assert intercept == pytest.approx(3.0)


def test_theil_sen_needs_distinct_x():
    with pytest.raises(ValueError, match="distinct"):
        theil_sen(np.ones(4), np.arange(4.0))
