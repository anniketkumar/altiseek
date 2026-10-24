"""Cross-split candidate search and the calibration pair sets."""

from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest

from altiseek.data.image_stats import hamming, hamming_aligned, hashes_to_uint64
from altiseek.data.leakage import (
    Pairs,
    Thumbnails,
    choose_thresholds,
    close_pairs,
    consecutive_pairs,
    cross_split_pairs,
    geometric_inliers,
    random_pairs,
    video_runs,
)

from ._frames import FRAME, T0

RNG = np.random.default_rng(0)
RAW = {i: RNG.normal(size=(32, 32)).astype(np.float16) for i in range(6)}
# 0 and 3 are near-identical images in different splits; 1 and 4 share a hash by chance only.
RAW[3] = (RAW[0] + RNG.normal(scale=0.05, size=(32, 32))).astype(np.float16)
THUMBS = Thumbnails(RAW)
HASHES = {0: 0b1111, 1: 0xFF00, 2: 0xAAAA_0000, 3: 0b0111, 4: 0xFF01, 5: 0xFFFF_FFFF_0000_0000}
SPLIT = {0: "A", 1: "A", 2: "A", 3: "B", 4: "T", 5: "T"}


def ids(*values):
    return np.array(values, dtype=np.int64)


def test_close_pairs_scans_all_blocks():
    all_ids = np.arange(5, dtype=np.int64)
    hashes = hashes_to_uint64(["0", "1", "3", "ff", "0"])
    a, b, d = close_pairs(all_ids, hashes, all_ids[:1], hashes[:1], max_hamming=1, block=2)
    assert list(zip(a.tolist(), b.tolist(), d.tolist(), strict=True)) == [
        (0, 0, 0),
        (1, 0, 1),
        (4, 0, 0),
    ]


def test_cross_split_pairs_skip_same_split_and_confirm_by_correlation():
    pairs = cross_split_pairs(SPLIT, HASHES, THUMBS, max_hamming=1)
    found = dict(
        zip(zip(pairs.a.tolist(), pairs.b.tolist(), strict=True), pairs.correlation, strict=True)
    )
    assert set(found) == {(0, 3), (1, 4)}
    assert found[0, 3] > 0.99
    assert abs(found[1, 4]) < 0.3
    assert len(pairs.where(pairs.correlation > 0.9)) == 1


def test_thumbnail_correlation():
    r = THUMBS.correlation(ids(0, 0, 0), ids(0, 3, 1))
    assert r[0] == pytest.approx(1.0, abs=1e-5)
    assert r[1] > 0.99
    flat = Thumbnails({7: np.ones((32, 32)), 8: RAW[0]})
    assert flat.correlation(ids(7), ids(8))[0] == 0.0


def test_consecutive_and_random_pairs():
    pairs = consecutive_pairs({"v": [0, 3, 1]}, HASHES, THUMBS)
    assert pairs.a.tolist() == [0, 3]
    assert pairs.b.tolist() == [3, 1]
    assert pairs.hamming.tolist() == [1, 11]
    sample = random_pairs([0, 1], [4, 5], 10, seed=3, hash_of=HASHES, thumbs=THUMBS)
    assert len(sample) == 10
    assert set(sample.a.tolist()) <= {0, 1}
    assert set(sample.b.tolist()) <= {4, 5}
    again = random_pairs([0, 1], [4, 5], 10, seed=3, hash_of=HASHES, thumbs=THUMBS)
    assert (sample.a == again.a).all()
    assert (sample.b == again.b).all()


def test_video_runs_split_on_time_gaps_and_skip_non_video_frames():
    def at(image_id, seconds, folder="V", **changes):
        return replace(
            FRAME,
            image_id=image_id,
            source_folder=folder,
            timestamp=T0 + timedelta(seconds=seconds),
            **changes,
        )

    frames = [
        at(3, 1.0),
        at(1, 0.0),
        at(2, 0.5),
        at(4, 5.0),
        at(5, 5.5),
        at(6, 9.0),  # alone: no run
        at(7, 0.2, folder="W"),
        at(8, 0.6, folder="W", timestamp_source="meta"),  # not a video timestamp
    ]
    assert video_runs(frames, max_gap_s=1.0) == {"mavic_4k/V/1": [1, 2, 3], "mavic_4k/V/4": [4, 5]}


def test_aligned_and_pairwise_hamming_agree():
    a = hashes_to_uint64(["0123456789abcdef", "ffffffffffffffff"])
    b = hashes_to_uint64(["0123456789abcdee", "0000000000000000"])
    assert hamming_aligned(a, b).tolist() == [1, 64]
    assert np.diag(hamming(a, b)).tolist() == [1, 64]


def test_choose_thresholds_maximises_recall_without_false_alarms():
    def pairs(h, r):
        h, r = np.asarray(h), np.asarray(r, dtype=np.float32)
        return Pairs(np.zeros(len(h), np.int64), np.zeros(len(h), np.int64), h, r)

    positives = pairs([0, 2, 4, 6, 9], [0.999, 0.99, 0.97, 0.95, 0.90])
    negatives = pairs([5, 8, 3], [0.96, 0.93, 0.80])
    t = choose_thresholds(
        positives, negatives, max_hamming_limit=12, correlations=[0.9, 0.95, 0.97, 0.99]
    )
    # Distance 6 needs r > 0.96 (a negative sits at 5 bits, r 0.96): the best rule is h<=4, r>=0.97.
    assert (t.max_hamming, t.min_correlation, t.false_alarms) == (4, 0.97, 0)
    assert t.recall == pytest.approx(3 / 5)
    with pytest.raises(ValueError, match="no grid point"):
        choose_thresholds(positives, pairs([0], [1.0]), 4, [0.99])


def test_geometric_inliers_separate_duplicates_from_look_alikes():
    rng = np.random.default_rng(5)
    base = (rng.random((540, 960)) * 255).astype(np.uint8)
    base = np.repeat(base[..., None], 3, axis=2)
    shifted = np.roll(base, (6, 9), axis=(0, 1))  # the same scene, slightly moved
    other = np.repeat((rng.random((540, 960)) * 255).astype(np.uint8)[..., None], 3, axis=2)
    same = geometric_inliers(base, shifted, width_px=960)
    different = geometric_inliers(base, other, width_px=960)
    assert same > 200
    assert different < 20
    assert geometric_inliers(base, shifted, width_px=960) == same  # deterministic
