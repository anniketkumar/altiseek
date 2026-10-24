"""Lab statistics, colour transfer and appearance conditions on synthetic images."""

import numpy as np
import pytest

from altiseek.world.appearance import (
    FEATURES,
    Stats,
    fit_conditions,
    stats,
    to_lab,
    to_rgb,
    transfer,
)


def noise_image(mean_rgb, spread, seed=0, size=64):
    rng = np.random.default_rng(seed)
    image = np.asarray(mean_rgb, dtype=np.float64) + rng.normal(0, spread, (size, size, 3))
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def test_lab_round_trip():
    image = noise_image((60, 110, 120), 20)
    assert np.abs(to_rgb(to_lab(image)).astype(int) - image).max() <= 1


def test_uniform_patch_has_no_spread_or_texture():
    s = stats(np.full((32, 32, 3), (70, 120, 130), dtype=np.uint8))
    assert s.std == pytest.approx((0, 0, 0), abs=1e-3)  # float32 rounding
    assert s.texture == pytest.approx(0.0, abs=1e-3)
    assert len(s.features()) == len(FEATURES)


def test_transfer_moves_the_statistics():
    source = noise_image((50, 90, 100), 10, seed=1)
    target = stats(noise_image((90, 140, 150), 25, seed=2))
    moved = stats(transfer(source, stats(source), target))
    assert moved.mean == pytest.approx(target.mean, abs=0.5)
    assert moved.std == pytest.approx(target.std, rel=0.05)
    shifted = stats(transfer(source, stats(source), target, scale=False))
    assert shifted.mean == pytest.approx(target.mean, abs=0.5)
    assert shifted.std == pytest.approx(stats(source).std, rel=0.05)


def test_mask_restricts_the_statistics():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[:, 5:] = 200
    mask = np.zeros((10, 10), dtype=bool)
    mask[:, :5] = True
    assert stats(image, mask).mean[0] == pytest.approx(0.0, abs=1e-3)


def test_conditions_are_seeded_sorted_and_assign_to_the_nearest():
    rng = np.random.default_rng(3)
    bright = rng.normal([80, 0, 5, 4, 10], 1, (40, 5))
    dark = rng.normal([30, 0, -5, 2, 5], 1, (40, 5))
    features = np.vstack([bright, dark])
    a = fit_conditions(features, 2, seed=7)
    assert a == fit_conditions(features, 2, seed=7)
    labels = a.assign(features)
    assert set(labels[:40]) == {1}  # brighter L -> higher condition number
    assert set(labels[40:]) == {0}
    assert a.assign(np.array([[78.0, 0, 5, 4, 10]])).tolist() == [1]
    assert isinstance(Stats((1.0, 2.0, 3.0), (0.1, 0.2, 0.3), 4.0).features()[3], float)
