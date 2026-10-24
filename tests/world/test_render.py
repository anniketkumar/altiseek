"""World composition and views on a tiny on-disk library and a 40 m world."""

import numpy as np
import pytest
from PIL import Image

from altiseek.geometry.levels import Level, LevelTable
from altiseek.geometry.raster import Raster
from altiseek.imaging.config import CameraSpec
from altiseek.imaging.pipeline import CameraPipeline
from altiseek.world.appearance import stats
from altiseek.world.config import VariantSettings
from altiseek.world.library import BackgroundRow, Library, ObjectRow
from altiseek.world.render import compose_world, feather, render_view, transformed
from altiseek.world.spec import Gaussian, Placement, Prior, WorldObject, WorldSpec

WATER = (90, 120, 110)
RED = (220, 30, 30)
TABLE = LevelTable(
    cell_m=2.5, grid_cells=16, levels=(Level(name="L1", footprint_cells=8, stride_cells=4),)
)
CAMERA = CameraSpec(resolution_px=128, hfov_deg=80.0, pipeline=CameraPipeline())
STILL = VariantSettings(
    count=1, shift_view_px=0.0, exposure=0.0, white_balance=0.0, glint=0.0, glint_threshold=200
)
LIVELY = VariantSettings(
    count=4, shift_view_px=0.75, exposure=0.05, white_balance=0.03, glint=0.2, glint_threshold=200
)


@pytest.fixture
def library(tmp_path):
    (tmp_path / "backgrounds").mkdir()
    (tmp_path / "objects").mkdir()
    patch = np.full((256, 256, 3), WATER, dtype=np.uint8)
    s = stats(patch)
    rows = []
    for i in range(4):
        Image.fromarray(patch).save(tmp_path / "backgrounds" / f"p{i}.png")
        rows.append(BackgroundRow(f"p{i}", i, "s", "V", "B", "-", *s.mean, *s.std, s.texture, 0))
    crop = np.zeros((60, 60, 4), dtype=np.uint8)
    crop[..., :3] = WATER
    crop[10:50, 10:50, 3] = 255  # opaque middle, transparent 10 px ring
    crop[5:55, 5:55, 3] = np.maximum(crop[5:55, 5:55, 3], 128)
    crop[25:35, 25:35, :3] = RED
    Image.fromarray(crop).save(tmp_path / "objects" / "c0.png")
    obj = ObjectRow(
        "c0", "swimmer", 0, 0, "s", "V", "B", "-", 60, 29.5, 29.5, 0.5, 0.5, *s.mean, *s.std
    )
    condition = {"count": 4} | dict(
        zip(
            ("l_mean", "a_mean", "b_mean", "l_std", "a_std", "b_std"),
            (*s.mean, *s.std),
            strict=True,
        )
    )
    return Library(
        path=tmp_path,
        library_id="lib",
        split="B",
        gsd_mpp=0.05,
        patch_px=256,
        ring_px=10,
        backgrounds=tuple(rows),
        objects=(obj,),
        conditions=(condition | {"texture": s.texture},),
    )


def world(*objects: WorldObject) -> WorldSpec:
    prior = Prior(
        components=(
            Gaussian(x_m=20, y_m=20, sigma_major_m=5, sigma_minor_m=5, angle_deg=0, weight=1),
        )
    )
    background = tuple(
        Placement(
            patch_id=f"p{(i + j) % 4}",
            x_m=5.0 + 10 * i,
            y_m=5.0 + 10 * j,
            turns=(i * j) % 4,
            flip=bool(i % 2),
        )
        for j in range(4)
        for i in range(4)
    )
    return WorldSpec(
        world_id="",
        seed=1,
        split="B",
        world_set="sensor-fit",
        family="f",
        library_id="lib",
        area_m=40.0,
        cell_m=2.5,
        gsd_mpp=0.05,
        condition=0,
        prior_given=prior,
        prior_true=prior,
        background=background,
        objects=objects,
    ).identified()


def swimmer(x, y, rotation=0.0):
    return WorldObject(
        category="swimmer",
        crop_id="c0",
        x_m=x,
        y_m=y,
        rotation_deg=rotation,
        scale=1.0,
        target=True,
    )


def test_feather_weights():
    w = feather(256, 56)
    assert w.shape == (256, 256)
    assert w[128, 128] == pytest.approx(1.0)
    assert 0 < w[0, 128] < 0.02
    assert w[:, 10] == pytest.approx(w[:, 245])


def test_a_uniform_world_composes_uniformly(library):
    composed = compose_world(world(), library)
    assert composed.shape == (800, 800, 3)
    assert np.abs(composed.astype(int) - WATER).max() <= 1


def test_objects_land_at_their_world_position(library):
    raster = Raster.covering(40.0, 40.0, 0.05)
    for rotation in (0.0, 90.0, 33.0):
        composed = compose_world(world(swimmer(12.0, 30.0, rotation)), library)
        col, row = raster.to_pixel(12.0, 30.0)
        assert tuple(composed[round(float(row)), round(float(col))]) == pytest.approx(RED, abs=6)
    turned, (cx, cy) = transformed(np.zeros((60, 60, 4), np.uint8), (29.5, 29.5), 45.0, 1.0)
    assert turned.shape[0] > 60
    assert (cx, cy) == pytest.approx(
        ((turned.shape[1] - 1) / 2, (turned.shape[0] - 1) / 2), abs=1.0
    )


def test_views_show_their_objects_and_truth(library):
    spec = world(swimmer(12.0, 30.0))
    composed = compose_world(spec, library)
    level = TABLE.levels[0]
    view, truth = render_view(
        spec,
        composed,
        table=TABLE,
        camera=CAMERA,
        level=level,
        cell=(0, 8),
        variant=0,
        settings=STILL,
    )
    assert view.shape == (128, 128, 3)
    (obj,) = truth
    assert obj.target
    assert (obj.col_px, obj.row_px) == pytest.approx(
        (12.0 / 20 * 128 - 0.5, 10.0 / 20 * 128 - 0.5), abs=0.01
    )
    assert view[round(obj.row_px), round(obj.col_px), 0] > 150  # red at the object
    _, empty = render_view(
        spec,
        composed,
        table=TABLE,
        camera=CAMERA,
        level=level,
        cell=(8, 0),
        variant=0,
        settings=STILL,
    )
    assert empty == ()


def test_rendering_is_deterministic_and_variants_differ(library):
    spec = world(swimmer(12.0, 30.0))
    composed = compose_world(spec, library)
    assert np.array_equal(composed, compose_world(spec, library))
    kwargs = {
        "table": TABLE,
        "camera": CAMERA,
        "level": TABLE.levels[0],
        "cell": (4, 4),
        "settings": LIVELY,
    }
    a, _ = render_view(spec, composed, variant=0, **kwargs)
    b, _ = render_view(spec, composed, variant=0, **kwargs)
    c, _ = render_view(spec, composed, variant=1, **kwargs)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_overlapping_views_agree_on_shared_ground(library):
    spec = world(swimmer(15.0, 10.0))
    composed = compose_world(spec, library)
    kwargs = {
        "table": TABLE,
        "camera": CAMERA,
        "level": TABLE.levels[0],
        "variant": 0,
        "settings": STILL,
    }
    west, _ = render_view(spec, composed, cell=(0, 0), **kwargs)
    east, _ = render_view(spec, composed, cell=(4, 0), **kwargs)
    assert np.array_equal(west[:, 64:], east[:, :64])


def test_a_world_needs_its_own_library(library):
    with pytest.raises(ValueError, match="needs library"):
        compose_world(world().model_copy(update={"library_id": "other"}), library)
