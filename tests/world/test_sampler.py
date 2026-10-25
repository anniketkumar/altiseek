"""World specs and the scenario sampler on a small in-memory library."""

import dataclasses
import itertools
import math
from pathlib import Path

import numpy as np
import pytest

from altiseek.geometry.levels import Level, LevelTable
from altiseek.world.config import FamilyGrid, PriorSettings, WorldConfig, WorldSet
from altiseek.world.library import BackgroundRow, Library, ObjectRow
from altiseek.world.sampler import draw_point, prior_grid, sample_world
from altiseek.world.spec import Gaussian, Prior, WorldSpec


def patch(i: int, condition: int) -> BackgroundRow:
    return BackgroundRow(f"p{i}", i, "s", "V", "B", "-", 50, 0, 5, 3, 1, 1, 2.0, condition)


def crop(i: int, category: str, size_px: int = 44) -> ObjectRow:
    c = size_px / 2 - 0.5
    return ObjectRow(
        f"c{i}", category, i, i, "s", "V", "B", "-", size_px, c, c, 0.5, 0.5, 50, 0, 5, 2, 1, 1
    )


LIBRARY = Library(
    path=Path("unused"),
    library_id="lib",
    split="B",
    gsd_mpp=0.05,
    patch_px=256,
    ring_px=6,
    backgrounds=tuple(patch(i, i % 2) for i in range(60)),
    objects=tuple(
        [crop(i, "swimmer") for i in range(10)] + [crop(100 + i, "boat", 200) for i in range(3)]
    ),
    conditions=({"count": 30}, {"count": 30}),
)
SET = WorldSet(name="sensor-fit", split="B", per_family=1, seed_base=0)
CFG = WorldConfig(
    levels=LevelTable(
        cell_m=2.5, grid_cells=120, levels=(Level(name="L1", footprint_cells=20, stride_cells=10),)
    ),
    libraries={"B": Path("unused")},
    prior=PriorSettings(mean_margin_m=50.0, aspect=(0.3, 1.0), mixture_share=0.5),
    grid=FamilyGrid(
        prior_sigma_m={"sharp": (20.0, 40.0), "diffuse": (60.0, 120.0)},
        targets={"single": (1, 1), "group": (2, 4)},
        distractors={"low": (0, 3), "high": (10, 20)},
    ),
    group_radius_m=6.0,
    lane_share=0.5,
    lane_half_width_m=15.0,
    conditions=(0, 1),
    min_condition_patches=20,
    patch_step_m=10.0,
    clearance_m=0.3,
    scale_jitter=0.05,
    world_sets=(SET,),
    out_dir=Path("unused"),
)


def test_prior_grid_is_a_distribution_elongated_along_its_angle():
    prior = Prior(
        components=(
            Gaussian(x_m=150, y_m=150, sigma_major_m=30, sigma_minor_m=10, angle_deg=0, weight=1),
        )
    )
    q = prior_grid(prior, 300.0, 2.5)
    assert q.shape == (120, 120)
    assert q.sum() == pytest.approx(1.0)
    ix, iy = np.unravel_index(q.argmax(), q.shape)
    assert abs(ix - 59.5) <= 1
    assert abs(iy - 59.5) <= 1
    cells = np.arange(120)
    var_x = (q.sum(axis=1) * (cells - 59.5) ** 2).sum()
    var_y = (q.sum(axis=0) * (cells - 59.5) ** 2).sum()
    assert var_x > 5 * var_y  # 30 m along x, 10 m along y


def test_draws_stay_in_the_truncation_square():
    prior = Prior(
        components=(
            Gaussian(x_m=10, y_m=10, sigma_major_m=50, sigma_minor_m=50, angle_deg=30, weight=1),
        )
    )
    rng = np.random.default_rng(1)
    points = [draw_point(prior, rng, 5.0, 295.0) for _ in range(200)]
    assert all(p is not None and 5.0 <= p[0] <= 295.0 and 5.0 <= p[1] <= 295.0 for p in points)


def test_worlds_are_seeded_and_content_addressed():
    a = sample_world(7, ("sharp", "group", "high"), SET, CFG, LIBRARY)
    assert a == sample_world(7, ("sharp", "group", "high"), SET, CFG, LIBRARY)
    assert a.world_id == a.content_id()
    assert len(a.world_id) == 16
    assert sample_world(8, ("sharp", "group", "high"), SET, CFG, LIBRARY).world_id != a.world_id
    moved = a.model_copy(update={"seed": 9})
    assert moved.content_id() != a.world_id


@pytest.mark.parametrize(
    "family", list(itertools.product(("sharp", "diffuse"), ("single", "group"), ("low", "high")))
)
def test_worlds_respect_the_family_and_the_layout_rules(family):
    for seed in range(5):
        w = sample_world(seed, family, SET, CFG, LIBRARY)
        lo, hi = CFG.grid.targets[family[1]]
        assert lo <= len(w.targets) <= hi
        assert len(w.objects) - len(w.targets) <= CFG.grid.distractors[family[2]][1]
        crops = {c.crop_id: c for c in LIBRARY.objects}
        radii = []
        for o in w.objects:
            c = crops[o.crop_id]
            g = LIBRARY.gsd_mpp * o.scale * math.sqrt(2)
            full, opaque = c.size_px / 2 * g, (c.size_px / 2 - LIBRARY.ring_px) * g
            assert full <= o.x_m <= 300 - full
            assert full <= o.y_m <= 300 - full
            assert (o.category == "swimmer") == o.target
            radii.append((o.x_m, o.y_m, opaque))
        for (x1, y1, r1), (x2, y2, r2) in itertools.combinations(radii, 2):
            assert math.hypot(x1 - x2, y1 - y2) >= r1 + r2 + CFG.clearance_m - 1e-9
        targets = [(t.x_m, t.y_m) for t in w.targets]
        for a, b in itertools.combinations(targets, 2):
            assert math.hypot(a[0] - b[0], a[1] - b[1]) <= 2 * CFG.group_radius_m
        patches = {p.patch_id: p for p in LIBRARY.backgrounds}
        assert all(patches[p.patch_id].condition == w.condition for p in w.background)
        assert len(w.background) == 30 * 30


def test_conditions_follow_patch_counts_and_the_allowed_list():
    library = dataclasses.replace(
        LIBRARY,
        backgrounds=tuple(patch(i, int(i >= 10)) for i in range(60)),
        conditions=({"count": 20}, {"count": 100}),
    )
    family = ("sharp", "single", "low")
    drawn = [sample_world(s, family, SET, CFG, library).condition for s in range(300)]
    assert 0.75 < np.mean(drawn) < 0.92  # 100 of 120 patches
    only = CFG.model_copy(update={"conditions": (0,)})
    assert {sample_world(s, family, SET, only, library).condition for s in range(20)} == {0}
    rare = CFG.model_copy(update={"min_condition_patches": 21})
    assert {sample_world(s, family, SET, rare, library).condition for s in range(20)} == {1}


def test_split_purity_is_enforced():
    with pytest.raises(ValueError, match="needs split"):
        sample_world(
            0, ("sharp", "single", "low"), SET.model_copy(update={"split": "T"}), CFG, LIBRARY
        )


def test_specs_round_trip_and_refuse_tampering(tmp_path):
    w = sample_world(3, ("diffuse", "single", "low"), SET, CFG, LIBRARY)
    w.save(tmp_path / "w.json")
    assert WorldSpec.load(tmp_path / "w.json") == w
    w.model_copy(update={"seed": 4}).save(tmp_path / "bad.json")
    with pytest.raises(ValueError, match="does not match"):
        WorldSpec.load(tmp_path / "bad.json")
