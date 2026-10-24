"""Rectification against a synthetic sea whose appearance is known everywhere on the ground."""

import math

import numpy as np
import pytest

from altiseek.geometry.camera import TiltedCamera
from altiseek.geometry.rectify import FILL, ViewWindow, project_boxes, rectify, valid_pixels
from altiseek.imaging.pipeline import CameraPipeline

OFF = CameraPipeline()


def texture(x_m, y_m):
    """A smooth ground pattern (periods of 7 m and 11 m), grey levels 68-188."""
    return 128 + 60 * np.sin(2 * np.pi * x_m / 7.0) * np.cos(2 * np.pi * y_m / 11.0)


def photograph(camera):
    """The frame a camera would take of the textured sea (sky pixels black)."""
    rows, cols = np.mgrid[0 : camera.height_px, 0 : camera.width_px].astype(np.float64)
    x, y, water = camera.image_to_ground(cols, rows)
    grey = np.where(water, texture(x, y), 0.0)
    return np.repeat(np.clip(np.round(grey), 0, 255)[..., None], 3, axis=2).astype(np.uint8)


def expected(window):
    col, row = np.meshgrid(np.arange(window.size_px), np.arange(window.size_px))
    g = window.gsd_mpp
    return texture(window.x0_m + (col + 0.5) * g, window.y0_m + window.side_m - (row + 0.5) * g)


# 960 x 540 frames with the mavic_4k focal length scaled to that width.
NADIR = TiltedCamera.centred(960, 540, 3029.0 / 4, 90.0, 30.0)
TILTED = TiltedCamera.centred(960, 540, 3029.0 / 4, 50.0, 30.0)


def test_nadir_view_reproduces_the_ground():
    window = ViewWindow(x0_m=-8.0, y0_m=-4.0, gsd_mpp=0.1, size_px=96)
    view = rectify(photograph(NADIR), NADIR, window, max_off_nadir_deg=60.0, pipeline=OFF)
    assert view.valid.all()
    assert view.downsample == 1  # native 4 cm/px is already coarser than the 2.5 cm grid
    err = np.abs(view.image[..., 0].astype(float) - expected(window))
    assert err.mean() < 1.5
    assert err.max() < 8


def test_tilted_view_reproduces_the_ground():
    # 50 deg pitch: the frame centre looks 40 deg off nadir, about 25 m ahead.
    window = ViewWindow(x0_m=-6.0, y0_m=20.0, gsd_mpp=0.15, size_px=80)
    view = rectify(photograph(TILTED), TILTED, window, max_off_nadir_deg=60.0, pipeline=OFF)
    assert view.valid.all()
    err = np.abs(view.image[..., 0].astype(float) - expected(window))
    assert err.mean() < 2.0
    assert err.max() < 12


def test_coarse_views_average_down_the_frame_first():
    window = ViewWindow(x0_m=-8.0, y0_m=-4.0, gsd_mpp=0.4, size_px=24)
    view = rectify(photograph(NADIR), NADIR, window, max_off_nadir_deg=60.0, pipeline=OFF)
    native = 30.0 / (3029.0 / 4)
    assert view.downsample == math.ceil(0.4 / (4 * native))
    assert np.abs(view.image[..., 0].astype(float) - expected(window)).mean() < 2.0


def test_pixels_outside_the_frame_or_beyond_the_off_nadir_limit_are_filled():
    half_out = ViewWindow(x0_m=-24.0, y0_m=-4.0, gsd_mpp=0.1, size_px=96)  # frame edge: -19 m
    view = rectify(photograph(NADIR), NADIR, half_out, max_off_nadir_deg=60.0, pipeline=OFF)
    assert view.valid.any()
    assert not view.valid.all()
    assert (view.image[~view.valid] == FILL).all()
    near = ViewWindow(x0_m=-4.0, y0_m=24.0, gsd_mpp=0.15, size_px=80)
    strict, _ = valid_pixels(TILTED, near, max_off_nadir_deg=30.0)
    assert not strict.any()  # the window is about 40 deg off nadir


def test_views_finer_than_the_frame_are_invalid():
    window = ViewWindow(x0_m=-2.0, y0_m=-2.0, gsd_mpp=0.02, size_px=64)  # native is 4 cm/px
    valid, finest = valid_pixels(NADIR, window, max_off_nadir_deg=60.0)
    assert not valid.any()
    assert math.isinf(finest)
    empty = rectify(photograph(NADIR), NADIR, window, max_off_nadir_deg=60.0, pipeline=OFF)
    assert (empty.image == FILL).all()


def test_frame_size_must_match_the_camera():
    with pytest.raises(ValueError, match="does not match"):
        rectify(
            np.zeros((10, 10, 3), np.uint8),
            NADIR,
            ViewWindow(0, 0, 0.1, 8),
            max_off_nadir_deg=60.0,
            pipeline=OFF,
        )


def test_project_boxes_nadir_and_sky():
    window = ViewWindow(x0_m=-8.0, y0_m=-4.0, gsd_mpp=0.1, size_px=96)
    gsd = 30.0 / (3029.0 / 4)
    # A box of 10 x 6 frame pixels whose top-left corner sits at the frame centre.
    box = np.array([[NADIR.cx_px + 0.5, NADIR.cy_px + 0.5, 10.0, 6.0]])
    out, water = project_boxes(box, NADIR, window)
    assert water.all()
    width, height = out[0, 2] - out[0, 0], out[0, 3] - out[0, 1]
    assert width == pytest.approx(10 * gsd / 0.1)
    assert height == pytest.approx(6 * gsd / 0.1)
    assert out[0, 0] == pytest.approx(8.0 / 0.1)  # ground x = 0 is 8 m from the left edge
    sky_camera = TiltedCamera.centred(960, 540, 3029.0 / 4, 5.0, 30.0)
    _, water = project_boxes(np.array([[480.0, 0.0, 4.0, 4.0]]), sky_camera, window)
    assert not water.any()
