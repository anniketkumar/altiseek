"""Actual-pose projection, rotations and conservative whole-cell visibility."""

import numpy as np
import pytest

from altiseek.deploy.projection import GroundCamera, quaternion_matrix
from altiseek.geometry.levels import altitude_for_footprint_m


def nadir(x=150, y=150):
    # Camera +X points down, +Y west, +Z north.
    return GroundCamera(
        np.array([x, y, altitude_for_footprint_m(50, 80)]),
        np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]),
        640,
        80,
    )


def test_north_up_footprint_and_round_trip():
    camera = nadir()
    expected = np.array([[125, 175], [175, 175], [175, 125], [125, 125]])
    assert camera.footprint() == pytest.approx(expected)
    pixels = np.array([[319.5, 319.5], [0, 0], [639, 639], [130.2, 477.1]])
    ground = camera.pixel_to_ground(pixels)
    restored, positive = camera.ground_to_pixel(ground)
    assert restored == pytest.approx(pixels)
    assert positive.all()
    assert ground[0] == pytest.approx([150, 150])


def test_actual_position_excludes_partial_cells():
    aligned = nadir().fully_visible_cells(120, 2.5)
    shifted = nadir(x=150.1).fully_visible_cells(120, 2.5)
    assert aligned.sum() == 20 * 20
    assert shifted.sum() == 19 * 20
    assert aligned[50, 50]
    assert not shifted[50, 50]
    assert not shifted[70, 50]
    assert np.array_equal(np.where(shifted)[0], np.where(aligned[51:70])[0] + 51)


def test_tilted_pose_round_trip_and_whole_cell_mask():
    base = nadir()
    angle = np.deg2rad(8)
    rotation = quaternion_matrix(np.array([np.cos(angle / 2), 0, np.sin(angle / 2), 0]))
    camera = GroundCamera(base.position_enu_m, rotation @ base.rotation, 640, 80)
    uv = np.array([[20, 30], [280, 470], [630, 610]])
    assert camera.ground_to_pixel(camera.pixel_to_ground(uv))[0] == pytest.approx(uv)
    assert not np.allclose(camera.footprint(), base.footprint())
    mask = camera.fully_visible_cells(120, 2.5)
    assert mask.any()
    assert not np.array_equal(mask, base.fully_visible_cells(120, 2.5))


def test_bad_pose_and_horizon_rejected():
    with pytest.raises(ValueError, match="unit"):
        quaternion_matrix(np.array([2, 0, 0, 0]))
    with pytest.raises(ValueError, match="proper"):
        GroundCamera(np.array([0, 0, 30]), -np.eye(3), 640, 80)
    horizontal = GroundCamera(np.array([0, 0, 30]), np.eye(3), 640, 80)
    with pytest.raises(ValueError, match="intersect"):
        horizontal.pixel_to_ground(np.array([319.5, 319.5]))
