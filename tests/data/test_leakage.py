"""Cross-split candidate search and the calibration pair sets."""

import numpy as np
import pytest

from altiseek.data.image_stats import hamming, hamming_aligned, hashes_to_uint64
from altiseek.data.leakage import (
    close_pairs,
    consecutive_pairs,
    correlation,
    cross_split_pairs,
    random_pairs,
)

RNG = np.random.default_rng(0)
THUMBS = {i: RNG.normal(size=(32, 32)).astype(np.float16) for i in range(6)}
# 0 and 3 are near-identical images in different splits; 1 and 4 share a hash by chance only.
THUMBS[3] = (THUMBS[0] + RNG.normal(scale=0.05, size=(32, 32))).astype(np.float16)
HASHES = {0: 0b1111, 1: 0xFF00, 2: 0xAAAA_0000, 3: 0b0111, 4: 0xFF01, 5: 0xFFFF_FFFF_0000_0000}
SPLIT = {0: "A", 1: "A", 2: "A", 3: "B", 4: "T", 5: "T"}


def test_close_pairs_scans_all_blocks():
    ids = np.arange(5, dtype=np.int64)
    hashes = hashes_to_uint64(["0", "1", "3", "ff", "0"])
    pairs = close_pairs(ids, hashes, ids[:1], hashes[:1], max_hamming=1, block=2)
    assert pairs == [(0, 0, 0), (1, 0, 1), (4, 0, 0)]


def test_cross_split_pairs_skip_same_split_and_confirm_by_correlation():
    pairs = cross_split_pairs(SPLIT, HASHES, THUMBS, max_hamming=1)
    found = {(p.image_a, p.image_b): p for p in pairs}
    assert set(found) == {(0, 3), (1, 4)}
    assert found[0, 3].correlation > 0.99
    assert abs(found[1, 4].correlation) < 0.3


def test_correlation():
    a = THUMBS[0]
    assert correlation(a, a) == pytest.approx(1.0, abs=1e-6)
    assert correlation(a, -a) == pytest.approx(-1.0, abs=1e-6)
    assert correlation(a, np.ones_like(a)) == 0.0


def test_consecutive_and_random_pairs():
    pairs = consecutive_pairs({"v": [0, 3, 1]}, HASHES, THUMBS)
    assert [(p.image_a, p.image_b, p.hamming) for p in pairs] == [(0, 3, 1), (3, 1, 11)]
    sample = random_pairs([0, 1], [4, 5], 10, seed=3, hash_of=HASHES, thumb_of=THUMBS)
    assert len(sample) == 10
    assert all(p.image_a in (0, 1) and p.image_b in (4, 5) for p in sample)
    assert random_pairs([0], [4], 3, 1, HASHES, THUMBS) == random_pairs(
        [0], [4], 3, 1, HASHES, THUMBS
    )


def test_aligned_and_pairwise_hamming_agree():
    a = hashes_to_uint64(["0123456789abcdef", "ffffffffffffffff"])
    b = hashes_to_uint64(["0123456789abcdee", "0000000000000000"])
    assert hamming_aligned(a, b).tolist() == [1, 64]
    assert np.diag(hamming(a, b)).tolist() == [1, 64]
