"""The shared resampling path: area averaging, then blur and JPEG."""

import numpy as np
import pytest

from altiseek.imaging.pipeline import CameraPipeline, area_resize, camera_pipeline, render_view

RNG = np.random.default_rng(0)
IMAGE = RNG.integers(0, 256, (120, 160, 3), dtype=np.uint8)
OFF = CameraPipeline()
ON = CameraPipeline(blur_sigma_px=0.8, jpeg_quality=85)


def test_integer_factor_is_the_block_mean():
    out = area_resize(IMAGE, 30, 40)
    expected = IMAGE.reshape(30, 4, 40, 4, 3).mean(axis=(1, 3))
    assert out.shape == (30, 40, 3)
    assert np.abs(out.astype(float) - expected).max() <= 1.0


def test_non_integer_factor_keeps_the_mean_and_flat_regions():
    flat = np.full((97, 131, 3), 77, dtype=np.uint8)
    assert (area_resize(flat, 40, 50) == 77).all()
    out = area_resize(IMAGE, 47, 61)
    assert abs(out.mean() - IMAGE.mean()) < 1.0


def test_same_size_is_a_copy_and_upsampling_is_refused():
    same = area_resize(IMAGE, 120, 160)
    assert (same == IMAGE).all()
    assert same is not IMAGE
    with pytest.raises(ValueError, match="only downsamples"):
        area_resize(IMAGE, 121, 160)


def test_area_averaging_suppresses_aliasing():
    # A one-pixel checkerboard averages to flat grey; point sampling would alias it.
    checker = (np.indices((128, 128)).sum(axis=0) % 2 * 255).astype(np.uint8)
    out = area_resize(np.repeat(checker[..., None], 3, axis=2), 32, 32)
    assert np.abs(out.astype(float) - 127.5).max() <= 1.0


def test_pipeline_off_is_identity_and_on_degrades():
    assert (camera_pipeline(IMAGE, OFF) == IMAGE).all()
    degraded = camera_pipeline(IMAGE, ON)
    assert degraded.shape == IMAGE.shape
    assert degraded.dtype == np.uint8
    assert degraded.astype(float).std() < IMAGE.astype(float).std()


def test_every_call_gives_identical_output():
    first = render_view(IMAGE, 60, 80, ON)
    second = render_view(IMAGE.copy(), 60, 80, ON)
    assert (first == second).all()
    assert (render_view(IMAGE, 60, 80, OFF) == area_resize(IMAGE, 60, 80)).all()
