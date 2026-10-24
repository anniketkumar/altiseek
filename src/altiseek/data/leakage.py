"""Cross-split near-duplicate search: the leakage check (CLAUDE.md §7 step 7).

A pair of images in different splits is a candidate when their perceptual hashes differ by at
most ``max_hamming`` bits, and a near-duplicate when, in addition, the Pearson correlation of
their masked 32 x 32 luma thumbnails is at least ``min_correlation``. Open-water scenes are
similar to each other, so the hash alone over-reports; the thumbnail correlation confirms. The
check passes when no cross-split pair is a near-duplicate.

Both thresholds are calibrated on pairs whose status is known: consecutive frames of one video
(about 0.5 s apart, near-duplicates by construction) and frames from different campaigns (a lake
in 2020 and the North Sea in 2021, never duplicates). Pairs are held as arrays throughout:
the searches cover tens of millions of pairs.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from altiseek.data.frames import Frame
from altiseek.data.image_stats import hamming, hamming_aligned


@dataclass(frozen=True)
class Pairs:
    """Compared image pairs as parallel arrays: ids, hash distance (bits), correlation."""

    a: NDArray[np.int64]
    b: NDArray[np.int64]
    hamming: NDArray[np.int64]
    correlation: NDArray[np.float32]

    def __len__(self) -> int:
        return len(self.a)

    def where(self, mask: NDArray[np.bool_]) -> "Pairs":
        return Pairs(self.a[mask], self.b[mask], self.hamming[mask], self.correlation[mask])


class Thumbnails:
    """Thumbnails as zero-mean, unit-norm rows, so that a correlation is one dot product."""

    def __init__(self, thumb_of: Mapping[int, NDArray[np.floating]]) -> None:
        self.ids = np.array(sorted(thumb_of), dtype=np.int64)
        rows = np.stack([np.asarray(thumb_of[int(i)], np.float32).ravel() for i in self.ids])
        rows -= rows.mean(axis=1, keepdims=True)
        norm = np.linalg.norm(rows, axis=1, keepdims=True)
        self.rows = np.divide(rows, norm, out=np.zeros_like(rows), where=norm > 0)

    def correlation(
        self, a: NDArray[np.int64], b: NDArray[np.int64], chunk: int = 1 << 16
    ) -> NDArray[np.float32]:
        """Pearson correlation of the thumbnails of each pair ``(a[i], b[i])`` (0 if constant)."""
        ia, ib = np.searchsorted(self.ids, a), np.searchsorted(self.ids, b)
        out = np.empty(len(a), dtype=np.float32)
        for s in range(0, len(a), chunk):
            out[s : s + chunk] = np.einsum(
                "ij,ij->i", self.rows[ia[s : s + chunk]], self.rows[ib[s : s + chunk]]
            )
        return out


def hash_array(
    ids: Sequence[int] | NDArray[np.int64], hash_of: Mapping[int, int]
) -> NDArray[np.uint64]:
    """The hashes of ``ids`` as a uint64 array, in order."""
    return np.array([hash_of[int(i)] for i in ids], dtype=np.uint64)


def close_pairs(
    ids_a: NDArray[np.int64],
    hashes_a: NDArray[np.uint64],
    ids_b: NDArray[np.int64],
    hashes_b: NDArray[np.uint64],
    max_hamming: int,
    block: int = 512,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Ids and distances of every pair across the two sets within ``max_hamming`` bits.

    The distance matrix is computed ``block`` rows at a time to bound memory.
    """
    a, b, d = [], [], []
    for start in range(0, len(ids_a), block):
        dist = hamming(hashes_a[start : start + block], hashes_b)
        rows, cols = np.nonzero(dist <= max_hamming)
        a.append(ids_a[start + rows])
        b.append(ids_b[cols])
        d.append(dist[rows, cols])
    if not a:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    return np.concatenate(a), np.concatenate(b), np.concatenate(d).astype(np.int64)


def cross_split_pairs(
    split_of: Mapping[int, str],
    hash_of: Mapping[int, int],
    thumbs: Thumbnails,
    max_hamming: int,
) -> Pairs:
    """Every candidate pair between images of different splits, with its thumbnail correlation."""
    names = sorted(set(split_of.values()))
    members = {
        k: np.array(sorted(i for i, s in split_of.items() if s == k), dtype=np.int64) for k in names
    }
    parts = []
    for x, first in enumerate(names):
        for second in names[x + 1 :]:
            parts.append(
                close_pairs(
                    members[first],
                    hash_array(members[first], hash_of),
                    members[second],
                    hash_array(members[second], hash_of),
                    max_hamming,
                )
            )
    a = np.concatenate([p[0] for p in parts]) if parts else np.zeros(0, np.int64)
    b = np.concatenate([p[1] for p in parts]) if parts else np.zeros(0, np.int64)
    d = np.concatenate([p[2] for p in parts]) if parts else np.zeros(0, np.int64)
    return Pairs(a, b, d, thumbs.correlation(a, b))


def consecutive_pairs(
    videos: Mapping[str, Sequence[int]], hash_of: Mapping[int, int], thumbs: Thumbnails
) -> Pairs:
    """Each frame paired with the next one of its video (ids in time order): known duplicates."""
    a = np.array([x for ids in videos.values() for x in ids[:-1]], dtype=np.int64)
    b = np.array([y for ids in videos.values() for y in ids[1:]], dtype=np.int64)
    return _pairs(a, b, hash_of, thumbs)


def random_pairs(
    ids_a: Sequence[int],
    ids_b: Sequence[int],
    n: int,
    seed: int,
    hash_of: Mapping[int, int],
    thumbs: Thumbnails,
) -> Pairs:
    """``n`` random pairs drawn across two sets; known non-duplicates if the sets cannot share
    content (e.g. different campaigns)."""
    rng = np.random.default_rng(seed)
    a = rng.choice(np.asarray(ids_a, dtype=np.int64), size=n)
    b = rng.choice(np.asarray(ids_b, dtype=np.int64), size=n)
    return _pairs(a, b, hash_of, thumbs)


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
    a: NDArray[np.int64], b: NDArray[np.int64], hash_of: Mapping[int, int], thumbs: Thumbnails
) -> Pairs:
    if len(a) == 0:
        empty = np.zeros(0, dtype=np.int64)
        return Pairs(empty, empty, empty, np.zeros(0, dtype=np.float32))
    d = hamming_aligned(hash_array(a, hash_of), hash_array(b, hash_of))
    return Pairs(a, b, d, thumbs.correlation(a, b))


@dataclass(frozen=True)
class Thresholds:
    """A near-duplicate rule and how it scored on the calibration pairs."""

    max_hamming: int
    min_correlation: float
    #: Share of known duplicates the rule flags.
    recall: float
    #: Known non-duplicates the rule flags (zero by construction).
    false_alarms: int


def choose_thresholds(
    positives: Pairs, negatives: Pairs, max_hamming_limit: int, correlations: Sequence[float]
) -> Thresholds:
    """The grid point with the highest recall on ``positives`` and no false alarm on
    ``negatives``. Ties go to the stricter rule (smaller distance, then higher correlation).

    The grid is whole bits up to ``max_hamming_limit`` by the given correlation levels; its
    coarseness keeps a margin from the negatives instead of fitting their boundary exactly.
    """
    best: Thresholds | None = None
    for h in range(max_hamming_limit + 1):
        for r in sorted(correlations, reverse=True):
            if ((negatives.hamming <= h) & (negatives.correlation >= r)).any():
                break  # lower correlations at this distance only flag more negatives
            recall = float(((positives.hamming <= h) & (positives.correlation >= r)).mean())
            if best is None or recall > best.recall + 1e-12:
                best = Thresholds(h, float(r), recall, 0)
    if best is None:
        raise ValueError("no grid point avoids every false alarm")
    return best


def geometric_inliers(
    rgb_a: NDArray[np.uint8], rgb_b: NDArray[np.uint8], width_px: int = 1280
) -> int:
    """RANSAC homography inliers among ORB matches between two images: near-duplicates (same
    scene, about the same viewpoint) share many; look-alike water shares few.

    Both images are area-resized to ``width_px`` wide and blackened regions are masked out.
    Matches pass Lowe's ratio test (0.75); the homography tolerates 5 px reprojection error.
    OpenCV's random generator is reseeded, so the count is deterministic.
    """
    import cv2

    def prepare(rgb: NDArray[np.uint8]) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        scale = width_px / rgb.shape[1]
        small = cv2.resize(
            rgb, (width_px, round(rgb.shape[0] * scale)), interpolation=cv2.INTER_AREA
        )
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        return gray, np.where(gray >= 8, 255, 0).astype(np.uint8)

    (ga, ma), (gb, mb) = prepare(rgb_a), prepare(rgb_b)
    orb = cv2.ORB_create(nfeatures=3000)
    ka, da = orb.detectAndCompute(ga, ma)
    kb, db = orb.detectAndCompute(gb, mb)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return 0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 8:
        return 0
    src = np.float32([ka[m.queryIdx].pt for m in good])
    dst = np.float32([kb[m.trainIdx].pt for m in good])
    cv2.setRNGSeed(0)
    _, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return 0 if inliers is None else int(inliers.sum())
