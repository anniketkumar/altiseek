"""Cross-split near-duplicate search: the leakage check (CLAUDE.md §7 step 7).

A pair of images in different splits is a candidate when their perceptual hashes differ by at
most ``max_hamming`` bits, and a near-duplicate when, in addition, the Pearson correlation of
their masked 32 x 32 luma thumbnails is at least ``min_correlation``. Open-water scenes are
similar to each other, so the hash alone over-reports; the thumbnail correlation confirms. The
check passes when no cross-split pair is a near-duplicate.

Both thresholds are calibrated on pairs whose status is known: consecutive frames of one video
(about 0.5 s apart, near-duplicates by construction) and frames from different campaigns (a lake
in 2020 and the North Sea in 2021, never duplicates).
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from altiseek.data.frames import Frame
from altiseek.data.image_stats import hamming, hamming_aligned


@dataclass(frozen=True)
class Pair:
    """Two compared images: hash distance in bits, thumbnail correlation in [-1, 1]."""

    image_a: int
    image_b: int
    hamming: int
    correlation: float


def close_pairs(
    ids_a: NDArray[np.int64],
    hashes_a: NDArray[np.uint64],
    ids_b: NDArray[np.int64],
    hashes_b: NDArray[np.uint64],
    max_hamming: int,
    block: int = 512,
) -> list[tuple[int, int, int]]:
    """``(id_a, id_b, distance)`` for every pair across the two sets within ``max_hamming``.

    The distance matrix is computed ``block`` rows at a time to bound memory.
    """
    out = []
    for start in range(0, len(ids_a), block):
        dist = hamming(hashes_a[start : start + block], hashes_b)
        rows, cols = np.nonzero(dist <= max_hamming)
        out.extend(
            (int(ids_a[start + r]), int(ids_b[c]), int(dist[r, c]))
            for r, c in zip(rows, cols, strict=True)
        )
    return out


def correlation(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    """Pearson correlation of two thumbnails (0 if either is constant)."""
    x = a.astype(np.float64).ravel()
    y = b.astype(np.float64).ravel()
    x, y = x - x.mean(), y - y.mean()
    norm = float(np.sqrt((x @ x) * (y @ y)))
    return float(x @ y) / norm if norm > 0 else 0.0


def cross_split_pairs(
    split_of: Mapping[int, str],
    hash_of: Mapping[int, int],
    thumb_of: Mapping[int, NDArray[np.floating]],
    max_hamming: int,
) -> list[Pair]:
    """Every candidate pair between images of different splits, with its thumbnail correlation."""
    names = sorted(set(split_of.values()))
    members = {
        k: np.array(sorted(i for i, s in split_of.items() if s == k), dtype=np.int64) for k in names
    }
    hashes = {k: hash_array(ids, hash_of) for k, ids in members.items()}
    pairs = []
    for x, first in enumerate(names):
        for second in names[x + 1 :]:
            close = close_pairs(
                members[first], hashes[first], members[second], hashes[second], max_hamming
            )
            pairs.extend(Pair(a, b, d, correlation(thumb_of[a], thumb_of[b])) for a, b, d in close)
    return pairs


def consecutive_pairs(
    videos: Mapping[str, Sequence[int]],
    hash_of: Mapping[int, int],
    thumb_of: Mapping[int, NDArray[np.floating]],
) -> list[Pair]:
    """Each frame paired with the next one of its video (ids in time order): known duplicates."""
    a = [x for ids in videos.values() for x in ids[:-1]]
    b = [y for ids in videos.values() for y in ids[1:]]
    return _pairs(a, b, hash_of, thumb_of)


def random_pairs(
    ids_a: Sequence[int],
    ids_b: Sequence[int],
    n: int,
    seed: int,
    hash_of: Mapping[int, int],
    thumb_of: Mapping[int, NDArray[np.floating]],
) -> list[Pair]:
    """``n`` random pairs drawn across two sets; known non-duplicates if the sets cannot share
    content (e.g. different campaigns)."""
    rng = np.random.default_rng(seed)
    a = rng.choice(np.asarray(ids_a, dtype=np.int64), size=n).tolist()
    b = rng.choice(np.asarray(ids_b, dtype=np.int64), size=n).tolist()
    return _pairs(a, b, hash_of, thumb_of)


def hash_array(
    ids: Sequence[int] | NDArray[np.int64], hash_of: Mapping[int, int]
) -> NDArray[np.uint64]:
    """The hashes of ``ids`` as a uint64 array, in order."""
    return np.array([hash_of[int(i)] for i in ids], dtype=np.uint64)


def video_runs(frames: Sequence[Frame], max_gap_s: float) -> dict[str, list[int]]:
    """Runs of video frames at most ``max_gap_s`` apart, per video; ids in time order.

    Only frames with an image-level (video) timestamp take part. Runs of one frame are dropped.
    """
    by_video: defaultdict[str, list[tuple[float, int]]] = defaultdict(list)
    for f in frames:
        if f.timestamp is not None and f.timestamp_source == "image":
            by_video[f"{f.camera_group}/{f.source_folder}"].append(
                (f.timestamp.timestamp(), f.image_id)
            )
    runs: dict[str, list[int]] = {}
    for video in sorted(by_video):
        members = sorted(by_video[video])
        run = [members[0]]
        for item in [*members[1:], (np.inf, -1)]:
            if item[0] - run[-1][0] <= max_gap_s:
                run.append(item)
                continue
            if len(run) > 1:
                runs[f"{video}/{run[0][1]}"] = [i for _, i in run]
            run = [item]
    return runs


def _pairs(
    a: Sequence[int],
    b: Sequence[int],
    hash_of: Mapping[int, int],
    thumb_of: Mapping[int, NDArray[np.floating]],
) -> list[Pair]:
    if not a:
        return []
    dist = hamming_aligned(hash_array(a, hash_of), hash_array(b, hash_of))
    return [
        Pair(i, j, int(d), correlation(thumb_of[i], thumb_of[j]))
        for i, j, d in zip(a, b, dist, strict=True)
    ]
