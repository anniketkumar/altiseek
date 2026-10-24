"""Library extraction on a synthetic nadir frame: clean patches and object crops."""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from altiseek.data.frames import Box
from altiseek.data.splits import SplitRow
from altiseek.geometry.camera import TiltedCamera
from altiseek.world.config import BackgroundSettings, ObjectSettings
from altiseek.world.library import (
    Rect,
    background_patches,
    box_centres_m,
    box_footprints,
    object_crop,
    select_frames,
)

from ..data._frames import FRAME

NADIR = TiltedCamera.centred(3840, 2160, 3029.0, 90.0, 30.0)  # about 1 cm/px, 38 m x 21 m
WATER = (60, 110, 120)
OBJECTS = ObjectSettings(
    classes=("swimmer",), allowance_min_m=0.5, allowance_box_factor=0.5, ring_px=6
)
PATCHES = BackgroundSettings(
    patch_px=128, tile_px=512, tile_coverage=0.05, object_margin_m=2.0, black_margin_px=4
)
GEOMETRY = {"gsd_mpp": 0.05, "max_off_nadir_deg": 60.0, "supersample": 2}
NONE = np.zeros((0, 2))


def water(height, width, seed=0, spread=3.0):
    rng = np.random.default_rng(seed)
    image = rng.normal(WATER, spread, (height, width, 3))
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def box_at(u, v, size=50, category="swimmer", annotation_id=1):
    return Box(annotation_id, 0, category, u - size / 2, v - size / 2, size, size)


def test_rect_growth_overlap_and_containment():
    a = Rect(0, 0, 1, 1)
    assert a.intersects(Rect(0.5, 0.5, 2, 2))
    assert not a.intersects(Rect(1.5, 0, 2, 1))
    assert a.grown(1.0).intersects(Rect(1.5, 0, 2, 1))
    assert a.contains(np.array([0.5, 1.5]), np.array([0.5, 0.5])).tolist() == [True, False]


def test_patches_keep_clear_of_objects_and_blackened_pixels():
    frame = water(2160, 3840)
    frame[1000:1160, 2900:2960] = 0  # an ignore region about 10 m east of nadir
    box = box_at(1920, 1080)  # at nadir
    free = background_patches(frame, NADIR, [], **GEOMETRY, settings=PATCHES)
    clean = background_patches(frame, NADIR, [box], **GEOMETRY, settings=PATCHES)
    assert 0 < len(clean) < len(free)
    (foot,) = box_footprints(NADIR, [box])
    gx, gy, _ = NADIR.image_to_ground(np.array([2900.0, 2960.0]), np.array([1000.0, 1160.0]))
    blackened = Rect(gx.min(), gy.min(), gx.max(), gy.max()).grown(4 * 0.05)
    for _, rect, image in clean:
        assert image.shape == (128, 128, 3)
        assert not rect.intersects(foot.grown(PATCHES.object_margin_m))
        assert not rect.intersects(blackened)


def test_crop_holds_an_object_offset_from_its_box():
    frame = water(2160, 3840)
    cv2.circle(frame, (2000, 1100), 25, (230, 90, 40), -1)  # a 0.5 m object
    box = box_at(2030, 1100)  # its box drawn 30 px (0.3 m) to the east
    crop, reason = object_crop(frame, NADIR, box, NONE, **GEOMETRY, settings=OBJECTS)
    assert reason == "ok"
    alpha, image = crop.rgba[..., 3], crop.rgba[..., :3]
    orange = (image[..., 0] > 150) & (image[..., 2] < 90)
    assert orange.sum() > 50
    assert (alpha[orange] == 255).all()  # the whole object is pasted
    assert alpha[0, 0] == 0  # the window corners fade out
    n = crop.rgba.shape[0]
    assert crop.centre_px == pytest.approx(((n - 1) / 2, (n - 1) / 2), abs=0.6)
    assert crop.size_m == pytest.approx((50 * 30 / 3029, 50 * 30 / 3029), rel=1e-3)
    lab_l = cv2.cvtColor(np.float32([[WATER]]) / 255, cv2.COLOR_RGB2Lab)[0, 0, 0]
    assert crop.water.mean[0] == pytest.approx(float(lab_l), abs=1.0)


def test_crop_rejections():
    frame = water(2160, 3840)
    box = box_at(2030, 1100)
    (near,) = box_centres_m(NADIR, [box_at(2060, 1100)])  # 0.3 m away: inside the paste
    assert object_crop(frame, NADIR, box, near[None], **GEOMETRY, settings=OBJECTS) == (
        None,
        "neighbour",
    )
    (far,) = box_centres_m(NADIR, [box_at(2400, 1100)])
    assert object_crop(frame, NADIR, box, far[None], **GEOMETRY, settings=OBJECTS)[1] == "ok"
    frame[1080:1120, 2060:2100] = 0  # an ignore region inside the window
    assert object_crop(frame, NADIR, box, NONE, **GEOMETRY, settings=OBJECTS) == (
        None,
        "blackened",
    )
    edge = box_at(3830, 1100)  # the window runs off the frame
    assert object_crop(frame, NADIR, edge, NONE, **GEOMETRY, settings=OBJECTS) == (
        None,
        "outside the usable footprint",
    )


def test_frames_come_from_one_split_every_stride_per_video():
    frames = [
        replace(FRAME, image_id=i, source_folder=video, source_frame_no=i)
        for i, video in enumerate(["V1"] * 6 + ["V2"] * 3 + ["V3"] * 2)
    ]
    rows = {
        f.image_id: SplitRow(f.image_id, "s", "T" if f.source_folder == "V3" else "B")
        for f in frames
    }
    chosen = select_frames(frames, rows, "B", ["mavic_4k"], stride=2)
    assert [f.image_id for f in chosen] == [0, 2, 4, 6, 8]
    assert all(rows[f.image_id].split == "B" for f in chosen)
