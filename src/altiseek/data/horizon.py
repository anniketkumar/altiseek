"""Sea-horizon detection in oblique frames, used to calibrate focal lengths (CLAUDE.md §7 step 4).

Bright sky over darker sea gives the strongest downward brightness drop in each vertical strip
of the image; the horizon is the line that most strips agree on. SeaDronesSee blackens ignored
regions in the pixels, and their edges (with JPEG ringing around them) would otherwise produce
stronger drops than the horizon, so near-black pixels and a halo around them are left out of
the strip profiles. Shores, clouds and boats still make individual strips disagree; the caller
rejects frames with too few agreeing strips or a tilted line.
"""

import itertools
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

LUMA = np.array([0.299, 0.587, 0.114])
#: Pixels darker than this (0-255 luma) are treated as blackened, never as sky or sea.
BLACK_LUMA = 8.0


@dataclass(frozen=True)
class Horizon:
    """A detected horizon in full-resolution image pixels (row 0 at the top)."""

    #: Row of the line at the centre column.
    row_px: float
    #: Tilt of the line; positive when it descends to the right.
    roll_deg: float
    #: Strips that agree with the line, out of those whose stretch of it is not blackened.
    n_inliers: int
    inlier_fraction: float


def detect_horizon(
    rgb: NDArray[np.uint8],
    *,
    block_px: int = 4,
    n_strips: int = 32,
    inlier_tol_px: float = 8.0,
    halo_px: int = 8,
) -> Horizon | None:
    """Fit the sky-to-sea edge in an ``H x W x 3`` RGB image; None if no line can be fit.

    The image is block-averaged by ``block_px``. Blocks with a blackened pixel, or within
    ``halo_px`` of one, are masked. The central 80% of the width is split into ``n_strips``
    vertical strips, and each strip votes for the row with the steepest brightness drop in its
    masked mean profile. Of all lines through two votes, the one with the most votes within
    ``inlier_tol_px`` is kept and refined by least squares on those votes. Strips where the
    line runs behind a masked region cannot see the horizon and count neither for nor
    against it.
    """
    height, width = rgb.shape[0] // block_px, rgb.shape[1] // block_px
    luma = rgb[: height * block_px, : width * block_px] @ LUMA
    blocks = luma.reshape(height, block_px, width, block_px)
    small = blocks.mean(axis=(1, 3))
    valid = ~_dilate((blocks < BLACK_LUMA).any(axis=(1, 3)), -(-halo_px // block_px))

    edges = np.linspace(0.1 * width, 0.9 * width, n_strips + 1).astype(int)
    xs = (edges[:-1] + edges[1:] - 1) / 2.0
    ys = np.full(n_strips, np.nan)
    kernel = np.ones(5) / 5.0
    rows = np.arange(height)
    for i, (lo, hi) in enumerate(itertools.pairwise(edges)):
        strip, ok = small[:, lo:hi], valid[:, lo:hi]
        count = ok.sum(axis=1)
        good = count >= 0.5 * (hi - lo)
        if good.sum() < 2:
            continue
        mean = np.where(ok, strip, 0.0).sum(axis=1) / np.maximum(count, 1)
        profile = np.convolve(np.interp(rows, rows[good], mean[good]), kernel, mode="same")
        drop = profile[4:] - profile[:-4]  # drop[r] spans rows r..r+4, centred on r+2
        # drop[r] reads unsmoothed rows r-2..r+6; only fully unmasked windows may vote, so a
        # strip whose horizon is hidden cannot vote for the ramp interpolated across the mask.
        unmasked = np.convolve(good.astype(int), np.ones(9, dtype=int), mode="valid") == 9
        r = np.arange(3, drop.size - 3)
        candidates = np.where(unmasked[r - 2], drop[r], np.inf)
        if np.isfinite(candidates).any():
            ys[i] = r[np.argmin(candidates)] + 2

    voted = np.flatnonzero(~np.isnan(ys))
    if voted.size < 2:
        return None
    tol = inlier_tol_px / block_px
    inliers = _consensus(xs[voted], ys[voted], tol)
    slope, intercept = np.polyfit(xs[voted][inliers], ys[voted][inliers], 1)
    line = intercept + slope * xs
    visible = np.zeros(n_strips, dtype=bool)
    visible[voted] = _line_visible(valid, edges, line, margin=max(1, round(tol)))[voted]
    agree = np.zeros(n_strips, dtype=bool)
    agree[voted] = np.abs(ys[voted] - line[voted]) <= tol
    agree &= visible
    centre = (rgb.shape[1] / block_px - 1) / 2.0
    return Horizon(
        row_px=float((intercept + slope * centre + 0.5) * block_px - 0.5),
        roll_deg=float(np.degrees(np.arctan(slope))),
        n_inliers=int(agree.sum()),
        inlier_fraction=float(agree.sum() / visible.sum()) if visible.any() else 0.0,
    )


def theil_sen(x: NDArray[np.floating], y: NDArray[np.floating]) -> tuple[float, float]:
    """Robust line ``y = intercept + slope * x``: median pairwise slope, then median intercept."""
    i, j = np.triu_indices(len(x), k=1)
    dx = x[j] - x[i]
    keep = dx != 0
    if not keep.any():
        raise ValueError("Theil-Sen needs at least two distinct x values")
    slope = float(np.median((y[j] - y[i])[keep] / dx[keep]))
    return float(np.median(y - slope * x)), slope


def _consensus(x: NDArray[np.floating], y: NDArray[np.floating], tol: float) -> NDArray[np.bool_]:
    """Inlier mask of the line through two points that the most points lie within ``tol`` of.

    Exhaustive over point pairs (RANSAC without sampling), so it tolerates more than half of
    the points being outliers. Ties go to the smaller residual sum, then the first pair.
    """
    i, j = np.triu_indices(len(x), k=1)
    slope = (y[j] - y[i]) / (x[j] - x[i])
    intercept = y[i] - slope * x[i]
    residual = np.abs(y[None, :] - (intercept[:, None] + slope[:, None] * x[None, :]))
    close = residual <= tol
    count = close.sum(axis=1)
    spread = np.where(close, residual, 0.0).sum(axis=1)
    best = np.lexsort((spread, -count))[0]
    return close[best]


def _dilate(mask: NDArray[np.bool_], radius: int) -> NDArray[np.bool_]:
    """Grow a 2-D boolean mask by ``radius`` cells in every direction (square neighbourhood)."""
    rows = mask.copy()
    for shift in range(1, radius + 1):
        rows[shift:] |= mask[:-shift]
        rows[:-shift] |= mask[shift:]
    out = rows.copy()
    for shift in range(1, radius + 1):
        out[:, shift:] |= rows[:, :-shift]
        out[:, :-shift] |= rows[:, shift:]
    return out


def _line_visible(
    valid: NDArray[np.bool_], edges: NDArray[np.int_], line: NDArray[np.floating], margin: int
) -> NDArray[np.bool_]:
    """Per strip: are most blocks within ``margin`` rows of the line unmasked?"""
    out = np.zeros(len(line), dtype=bool)
    for i, (lo, hi) in enumerate(itertools.pairwise(edges)):
        row = round(float(line[i]))
        top, bottom = max(row - margin, 0), min(row + margin + 1, valid.shape[0])
        out[i] = bottom > top and valid[top:bottom, lo:hi].mean() >= 0.5
    return out
