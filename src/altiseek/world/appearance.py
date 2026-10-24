"""Appearance statistics, colour transfer and appearance conditions (CLAUDE.md §11).

Colours are compared in CIE Lab, OpenCV's float convention: L in 0-100, a and b roughly -127 to
127. Matching the mean, and optionally the spread, per Lab channel is the colour transfer of
Reinhard et al. (2001). Texture energy is the mean Sobel gradient magnitude of L. Appearance
conditions are k-means clusters of standardised patch statistics, numbered dark to bright.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

#: Patch statistics used to cluster appearance conditions, in this order.
FEATURES = ("l_mean", "a_mean", "b_mean", "l_std", "texture")
_EPS = 1e-6


def to_lab(rgb: NDArray[np.uint8]) -> NDArray[np.float32]:
    """An ``H x W x 3`` uint8 RGB image in float Lab."""
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)


def to_rgb(lab: NDArray[np.floating]) -> NDArray[np.uint8]:
    """Float Lab back to uint8 RGB, rounded and clipped."""
    rgb = cv2.cvtColor(np.ascontiguousarray(lab, dtype=np.float32), cv2.COLOR_Lab2RGB)
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


@dataclass(frozen=True)
class Stats:
    """Lab mean and standard deviation per channel, and texture energy."""

    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    texture: float

    def features(self) -> tuple[float, ...]:
        """The clustering features, in the order of :data:`FEATURES`."""
        return (*self.mean, self.std[0], self.texture)


def stats(rgb: NDArray[np.uint8], mask: NDArray[np.bool_] | None = None) -> Stats:
    """Statistics of ``rgb``, over the pixels of ``mask`` if given."""
    lab = to_lab(rgb)
    gx = cv2.Sobel(lab[..., 0], cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lab[..., 0], cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)
    pixels, grads = (lab[mask], grad[mask]) if mask is not None else (lab.reshape(-1, 3), grad)
    mean, std = pixels.mean(axis=0), pixels.std(axis=0)
    return Stats(
        (float(mean[0]), float(mean[1]), float(mean[2])),
        (float(std[0]), float(std[1]), float(std[2])),
        float(grads.mean()),
    )


def transfer(
    rgb: NDArray[np.uint8],
    source: Stats,
    target: Stats,
    *,
    scale: bool = True,
) -> NDArray[np.uint8]:
    """Move ``rgb``'s Lab statistics from ``source`` to ``target``: shift the mean and, with
    ``scale``, stretch the spread per channel."""
    lab = to_lab(rgb)
    src_mean, dst_mean = np.array(source.mean), np.array(target.mean)
    if scale:
        ratio = np.array(target.std) / np.maximum(np.array(source.std), _EPS)
        lab = (lab - src_mean) * ratio + dst_mean
    else:
        lab = lab + (dst_mean - src_mean)
    return to_rgb(lab)


@dataclass(frozen=True)
class Conditions:
    """Appearance conditions: k-means centroids in standardised feature units, dark to bright."""

    centre: tuple[float, ...]
    scale: tuple[float, ...]
    centroids: tuple[tuple[float, ...], ...]

    def assign(self, features: NDArray[np.floating]) -> NDArray[np.int64]:
        """The nearest condition of each row of ``features`` (``n x len(FEATURES)``)."""
        z = (np.asarray(features, dtype=np.float64) - self.centre) / self.scale
        d2 = ((z[:, None, :] - np.asarray(self.centroids)[None]) ** 2).sum(axis=2)
        return d2.argmin(axis=1)


def fit_conditions(features: Sequence[Sequence[float]], k: int, seed: int) -> Conditions:
    """k-means (k-means++ seeding, 10 restarts, seeded) of standardised ``features``."""
    x = np.asarray(features, dtype=np.float64)
    centre, scale = x.mean(axis=0), np.maximum(x.std(axis=0), _EPS)
    z = ((x - centre) / scale).astype(np.float32)
    cv2.setRNGSeed(seed)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 1e-6)
    _, _, centroids = cv2.kmeans(z, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    centroids = centroids[np.argsort(centroids[:, 0])]  # feature 0 is L mean: dark to bright
    return Conditions(
        tuple(float(v) for v in centre),
        tuple(float(v) for v in scale),
        tuple(tuple(float(v) for v in c) for c in centroids),
    )
