"""Perceptual hashes, blackened-region statistics and the resumable scan."""

import io

import numpy as np
import pytest
from PIL import Image

from altiseek.data.image_stats import (
    _dct_matrix,
    analyse,
    hamming,
    hashes_to_uint64,
    load_thumbnails,
    masked_thumbnail,
    phash_hex,
    run_scan,
)


def sea(seed, height=540, width=960):
    """Smooth random luma field: a stand-in for water with some structure."""
    rng = np.random.default_rng(seed)
    coarse = rng.normal(110, 25, (9, 16))
    up = np.kron(coarse, np.ones((height // 9 + 1, width // 16 + 1)))[:height, :width]
    return np.clip(up + rng.normal(0, 3, (height, width)), 0, 255)


def jpeg(luma, quality=90):
    rgb = np.repeat(luma[..., None], 3, axis=2).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def thumb_hash(luma):
    return phash_hex(masked_thumbnail(luma.astype(np.float32), luma < 8))


def dist(a, b):
    return int(hamming(hashes_to_uint64([a]), hashes_to_uint64([b]))[0, 0])


def test_dct_basis_is_orthonormal():
    basis = _dct_matrix(32)
    assert np.allclose(basis @ basis.T, np.eye(32))


def test_hash_is_stable_under_noise_and_compression_but_not_across_scenes():
    base = sea(0)
    noisy = np.clip(base + np.random.default_rng(1).normal(0, 4, base.shape), 0, 255)
    assert dist(thumb_hash(base), thumb_hash(base)) == 0
    assert dist(thumb_hash(base), thumb_hash(noisy)) <= 4
    assert dist(analyse(1, jpeg(base))[0].phash, analyse(2, jpeg(base, quality=60))[0].phash) <= 4
    others = [dist(thumb_hash(base), thumb_hash(sea(s))) for s in range(2, 12)]
    assert min(others) >= 16


def test_blackening_barely_moves_the_hash():
    base = sea(3)
    masked = base.copy()
    masked[100:220, 500:760] = 0
    assert dist(thumb_hash(base), thumb_hash(masked)) <= 6


def test_hamming_counts_bits():
    a = hashes_to_uint64(["0000000000000000", "ffffffffffffffff"])
    b = hashes_to_uint64(["0000000000000001", "000000000000000f", "ffffffffffffffff"])
    assert hamming(a, b).tolist() == [[1, 4, 64], [63, 60, 0]]


def test_analyse_reports_size_and_black_blocks():
    luma = np.full((2160, 3840), 120.0)
    luma[0:64, 0:96] = 0  # 2 x 3 black blocks of 32 px
    stats, thumb = analyse(7, jpeg(luma))
    assert (stats.image_id, stats.width_px, stats.height_px) == (7, 3840, 2160)
    assert stats.black_blocks == 6
    assert stats.black_fraction == pytest.approx(64 * 96 / (2160 * 3840), rel=0.1)
    assert thumb.shape == (32, 32)
    assert thumb.dtype == np.float16
    clean, _ = analyse(8, jpeg(np.full((2160, 3840), 120.0)))
    assert (clean.black_blocks, clean.black_fraction) == (0, 0.0)


def test_scan_resumes_and_skips_finished_images(tmp_path):
    images = {i: jpeg(sea(i, 270, 480)) for i in range(7)}
    reads = []

    def read(image_id):
        reads.append(image_id)
        return images[image_id]

    assert run_scan([0, 1, 2], read, tmp_path, workers=2, chunk_size=2) == 3
    assert run_scan(list(range(7)), read, tmp_path, workers=2, chunk_size=2) == 4
    assert run_scan(list(range(7)), read, tmp_path, workers=2, chunk_size=2) == 0
    assert sorted(reads) == list(range(7))
    thumbs = load_thumbnails(tmp_path)
    assert sorted(thumbs) == list(range(7))
    assert len(list(tmp_path.glob("thumbs_*.npz"))) == 4  # 2 + 2 chunks
