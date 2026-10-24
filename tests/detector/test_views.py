"""View planning and labels on synthetic cameras."""

import itertools
from dataclasses import replace

import numpy as np
import pytest

from altiseek.data.frames import Box
from altiseek.detector.views import (
    CLASSES,
    CameraModel,
    Label,
    coverage,
    finest_usable_gsd_mpp,
    footprint_m,
    frame_camera,
    labels_for,
    object_centres_m,
    random_windows,
    rotate_quarter,
    tile_windows,
    yolo_lines,
)
from altiseek.geometry.camera import TiltedCamera
from altiseek.geometry.rectify import ViewWindow, valid_pixels

from ..data._frames import FRAME

NADIR = TiltedCamera.centred(3840, 2160, 3029.0, 90.0, 30.0)  # 1 cm/px, 38 m x 21 m footprint
TILTED = TiltedCamera.centred(3840, 2160, 3029.0, 45.0, 30.0)


def test_frame_camera_applies_the_pitch_bias():
    cam = frame_camera(
        replace(FRAME, gimbal_pitch_deg=50.0, altitude_m=20.0), CameraModel(3029.0, -0.4)
    )
    assert cam.pitch_deg == pytest.approx(49.6)
    assert (cam.altitude_m, cam.focal_px) == (20.0, 3029.0)
    with pytest.raises(ValueError, match="no pitch"):
        frame_camera(replace(FRAME, altitude_m=None), CameraModel(3029.0, 0.0))


def test_footprint_respects_the_gsd_limit():
    assert len(footprint_m(NADIR, 0.05, 60.0)) > 0
    assert len(footprint_m(NADIR, 0.005, 60.0)) == 0  # finer than the frame's 1 cm/px
    near = footprint_m(TILTED, 0.02, 60.0)  # far rows are coarser than 2 cm/px
    full = footprint_m(TILTED, 0.2, 60.0)
    assert near[:, 1].max() < full[:, 1].max()


def test_tiles_do_not_overlap_and_cover_the_footprint():
    tiles = tile_windows(NADIR, 0.05, 160, 60.0, min_coverage=0.5)  # 8 m tiles
    assert len(tiles) >= 8
    corners = sorted((t.x0_m, t.y0_m) for t in tiles)
    for (xa, ya), (xb, yb) in itertools.pairwise(corners):
        assert abs(xa - xb) >= 8.0 - 1e-9 or abs(ya - yb) >= 8.0 - 1e-9
    assert all(coverage(NADIR, t, 60.0) >= 0.5 for t in tiles)


def test_random_windows_are_seeded_and_object_centred():
    centres = np.array([[3.0, 2.0]])
    a = random_windows(NADIR, 0.05, 128, 60.0, 0.6, centres, 5, 1.0, np.random.default_rng(7))
    b = random_windows(NADIR, 0.05, 128, 60.0, 0.6, centres, 5, 1.0, np.random.default_rng(7))
    assert a == b
    assert len(a) == 5
    side = 0.05 * 128
    for w in a:  # object-centred windows contain the object
        assert w.x0_m <= 3.0 <= w.x0_m + side
        assert w.y0_m <= 2.0 <= w.y0_m + side


def test_labels_keep_valid_centres_and_drop_ignored_boxes():
    window = ViewWindow(-4.0, -4.0, 0.02, 400)  # 8 m around the nadir point
    valid, _ = valid_pixels(NADIR, window, 60.0)
    centre = (NADIR.cx_px + 0.5, NADIR.cy_px + 0.5)
    boxes = [
        Box(1, 0, "swimmer", centre[0] - 40, centre[1] - 20, 80, 40),  # at the nadir point
        Box(2, 0, "ignored", centre[0], centre[1], 10, 10),
        Box(3, 0, "boat", 0.0, 0.0, 50, 50),  # frame corner, far outside the window
    ]
    labels = labels_for(NADIR, window, boxes, valid)
    assert len(labels) == 1
    lab = labels[0]
    assert lab.class_id == CLASSES.index("swimmer")
    assert (lab.x_max - lab.x_min) == pytest.approx(80 * (30 / 3029) / 0.02, rel=1e-6)
    assert (lab.x_min + lab.x_max) / 2 == pytest.approx(200.0, abs=0.01)
    assert object_centres_m(NADIR, boxes[:1]) == pytest.approx(np.array([[0.0, 0.0]]), abs=1e-9)


def test_yolo_lines_are_normalised():
    line = yolo_lines([Label(0, 10.0, 20.0, 30.0, 60.0)], 100)
    assert line == ["0 0.200000 0.400000 0.200000 0.400000"]


def test_quarter_turns_move_labels_with_the_image():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[1:3, 6:9] = 255  # rows 1-2, columns 6-8: box x 6..9, y 1..3
    label = Label(0, 6.0, 1.0, 9.0, 3.0)
    for turns in range(4):
        rotated, (moved,) = rotate_quarter(image, [label], turns)
        rows, cols = np.nonzero(rotated[..., 0])
        assert (moved.x_min, moved.x_max) == (cols.min(), cols.max() + 1)
        assert (moved.y_min, moved.y_max) == (rows.min(), rows.max() + 1)


def test_finest_usable_gsd():
    assert finest_usable_gsd_mpp(NADIR, 60.0) == pytest.approx(30.0 / 3029.0)
    # 45 deg pitch: the bottom rows are about 25 deg off nadir, so 20 deg excludes the frame.
    assert finest_usable_gsd_mpp(TILTED, 20.0) == float("inf")
    assert finest_usable_gsd_mpp(TILTED, 60.0) > 30.0 / 3029.0
