"""Cache contract on tiny rendered worlds; no local detector inference or downloads."""

from dataclasses import replace

import numpy as np
import pytest

from altiseek.cache.build import build_world, header_for, records
from altiseek.cache.config import CacheConfig
from altiseek.cache.storage import CachePool, WorldCache, draw_variant, view_keys
from altiseek.cache.validate import compare_detections, validate_world
from altiseek.config import load_config
from altiseek.detector.infer import Detection
from altiseek.world.config import RenderConfig
from tests.world.test_render import CAMERA, LIVELY, TABLE, library, swimmer, world

__all__ = ["library"]  # reuse the tiny on-disk fixture


class FakeDetector:
    detector_id = "fake"
    score_min = 0.05
    iou = 0.7
    max_det = 300

    def detect(self, views):
        # Outputs depend on the actual input pixels, catching render/variant drift.
        return [
            [
                Detection(0, 0.5 + float(v.mean()) / 1000, 40, 50, 50, 60),
                Detection(2, 0.01, 10, 10, 12, 12),
            ]
            for v in views
        ]


@pytest.fixture
def cfg(tmp_path):
    return CacheConfig(
        render=RenderConfig(camera=CAMERA, levels=TABLE, variants=LIVELY),
        detector_id="fake",
        checkpoint=tmp_path / "fake.pt",
        libraries={},
        library_ids={},
        world_set_ids={},
        batch_size=7,
        render_workers=2,
    )


@pytest.fixture
def built(library, cfg):
    spec = world(swimmer(12, 30))
    cache, _ = build_world(spec, library, cfg, FakeDetector())
    return spec, cache


def test_default_lattice_has_880_unique_views():
    cfg = load_config("configs/cache/default.yaml", CacheConfig)
    keys = view_keys(cfg.render)
    assert keys.shape == (880, 4)
    assert len(np.unique(keys, axis=0)) == 880
    assert [sum(keys[:, 0] == i) // 4 for i in range(5)] == [121, 49, 25, 16, 9]


def test_roundtrip_live_and_empty_truth(tmp_path, library, cfg, built):
    spec, cache = built
    path = tmp_path / "cache.npz"
    cache.save(path)
    loaded = WorldCache.load(path, expected=header_for(spec, cfg))
    assert np.array_equal(cache.detections, loaded.detections)
    assert loaded.lookup(0, 8, 0, 0)[1].size == 0
    assert len(loaded.detections) == len(loaded.views)  # low score removed
    indices = validate_world(loaded, spec, library, cfg, FakeDetector(), seed=42, samples=9)
    assert len(indices) == 9
    with pytest.raises(ValueError, match="read-only"):
        loaded.detections[0] = loaded.detections[0]


@pytest.mark.parametrize(
    ("field", "value"), [("detector_id", "wrong"), ("world_id", "wrong"), ("tau_min", 0.1)]
)
def test_loader_refuses_changed_identity(tmp_path, built, field, value):
    _, cache = built
    path = tmp_path / "cache.npz"
    cache.save(path)
    changed = cache.header.model_copy(update={field: value}).identified()
    assert changed.cache_id != cache.header.cache_id
    with pytest.raises(ValueError, match="header mismatch"):
        WorldCache.load(path, expected=changed)


def test_tampered_header_and_incomplete_lattice(built):
    _, cache = built
    args = [
        cache.header,
        cache.views,
        cache.offsets,
        cache.detections,
        cache.truth_offsets,
        cache.truth,
    ]
    args[0] = cache.header.model_copy(update={"cache_id": "bad"})
    with pytest.raises(ValueError, match="cache_id"):
        WorldCache(*args)
    args[0] = cache.header
    args[1] = cache.views[:-1]
    with pytest.raises(ValueError, match="lattice"):
        WorldCache(*args)
    args[1] = cache.views
    args[2] = cache.offsets.copy()
    args[2][-1] += 1
    with pytest.raises(ValueError, match="offsets"):
        WorldCache(*args)


def test_variants_are_seeded_and_independent_of_call_order():
    a = [draw_variant(17, i, 4) for i in range(100)]
    assert len(set(a)) == 4
    assert a == [draw_variant(17, i, 4) for i in reversed(range(100))][::-1]
    assert a != [draw_variant(18, i, 4) for i in range(100)]


def test_projection_matches_rendered_truth_including_shift(library, cfg):
    from altiseek.cache.build import render_key
    from altiseek.world.render import compose_world

    spec = world(swimmer(19.99, 20.01))
    composed = compose_world(spec, library)
    for v in range(4):
        key = np.asarray([0, 4, 4, v], dtype=np.int32)
        _, (t,) = render_key(spec, composed, cfg, key)
        # Truth uses pixel-centre coordinates; detector uses pixel-corner coordinates.
        d = Detection(0, 0.8, t.col_px, t.row_px, t.col_px + 1, t.row_px + 1)
        (row,) = records(spec, cfg, key, [d])
        assert row["x_m"] == pytest.approx(t.x_m, abs=1e-12)
        assert row["y_m"] == pytest.approx(t.y_m, abs=1e-12)
        assert (row["ix"], row["iy"]) == (7, 8)


def test_pool_evicts_and_refuses_other_detector(tmp_path, built):
    _, cache = built
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    cache.save(a)
    cache.save(b)
    pool = CachePool(capacity=1)
    first = pool.get(a, expected=cache.header)
    assert first is pool.get(a, expected=cache.header)
    pool.get(b, expected=cache.header)
    assert first is not pool.get(a, expected=cache.header)
    with pytest.raises(ValueError, match="header mismatch"):
        pool.get(a, expected=cache.header.model_copy(update={"detector_id": "bad"}).identified())


def test_live_check_detects_errors_and_tolerates_order(built):
    _, cache = built
    a = cache.detections[:2].copy()
    compare_detections(a, a[::-1])
    b = a.copy()
    b["score"] += 1e-5
    compare_detections(a, b)
    b["box_px"] += 1
    with pytest.raises(ValueError, match="mismatch"):
        compare_detections(a, b)
    with pytest.raises(ValueError, match="count mismatch"):
        compare_detections(a, a[:1])


def test_build_refuses_detector_and_world_mismatch(library, cfg):
    bad = FakeDetector()
    bad.detector_id = "wrong"
    with pytest.raises(ValueError, match="detector_id"):
        build_world(world(), library, cfg, bad)
    with pytest.raises(ValueError, match="world_id"):
        header_for(world().model_copy(update={"seed": 999}), cfg)


def test_empty_detections_are_complete(library, cfg):
    class EmptyDetector(FakeDetector):
        def detect(self, views):
            return [[] for _ in views]

    cache, timing = build_world(world(), library, cfg, EmptyDetector())
    assert len(cache.views) == 36
    assert cache.detections.size == cache.truth.size == 0
    assert timing["total_s"] > 0


def test_inference_and_render_settings_change_id(cfg):
    spec = world()
    baseline = header_for(spec, cfg)
    changed = cfg.model_copy(update={"inference": cfg.inference.model_copy(update={"iou": 0.6})})
    assert header_for(spec, changed).cache_id != baseline.cache_id
    changed = cfg.model_copy(
        update={
            "render": cfg.render.model_copy(
                update={"variants": LIVELY.model_copy(update={"count": 2})}
            )
        }
    )
    assert header_for(spec, changed).cache_id != baseline.cache_id


def test_validation_fails_for_changed_live_output(library, cfg, built):
    class ChangedDetector(FakeDetector):
        def detect(self, views):
            return [[replace(d, score=0.9) for d in row] for row in super().detect(views)]

    spec, cache = built
    with pytest.raises(ValueError, match="mismatch"):
        validate_world(cache, spec, library, cfg, ChangedDetector(), seed=7, samples=2)
