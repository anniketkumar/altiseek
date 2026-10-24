"""Detector views: nadir-equivalent square crops of real frames at chosen GSDs (CLAUDE.md §9).

Each view is a :class:`~altiseek.geometry.rectify.ViewWindow` on the water below one frame's
camera, rendered by :func:`~altiseek.geometry.rectify.rectify`. Training views are placed at
random, with a share centred on objects; evaluation views tile the frame's usable footprint
without overlap, so every object is counted once. A window is used only if enough of it has valid
source pixels (``min_coverage``); the rest of the view is fill. Both choose, per GSD bin, among
the frames that can serve it (:func:`can_serve`), so coarse bins are not starved (D23). Labels are
the annotated boxes projected into the view, kept when the box centre falls on a valid pixel.
"""

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from altiseek.data.frames import Box, Frame
from altiseek.geometry.camera import TiltedCamera
from altiseek.geometry.rectify import ViewWindow, project_boxes, source_valid

#: Detector classes, in label-id order. ODv2's ``ignored`` has no boxes and is not a class.
CLASSES = ("swimmer", "boat", "jetski", "life_saving_appliances", "buoy")


@dataclass(frozen=True)
class CameraModel:
    """A fitted camera group model (``data/processed/camera_models.json``, D12)."""

    focal_px: float
    pitch_bias_deg: float


@dataclass(frozen=True)
class Label:
    """A box in a view: class id and pixel-corner coordinates, clipped to the view."""

    class_id: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float


def frame_camera(frame: Frame, model: CameraModel) -> TiltedCamera:
    """The camera of a frame: logged pitch corrected by the fitted bias."""
    if frame.gimbal_pitch_deg is None or frame.altitude_m is None:
        raise ValueError(f"image {frame.image_id} has no pitch or altitude")
    return TiltedCamera.centred(
        frame.width_px,
        frame.height_px,
        model.focal_px,
        frame.gimbal_pitch_deg + model.pitch_bias_deg,
        frame.altitude_m,
    )


def footprint_m(
    camera: TiltedCamera, gsd_mpp: float, max_off_nadir_deg: float, step_px: int = 32
) -> NDArray[np.float64]:
    """Ground points (``n x 2``, metres) of a coarse grid of frame pixels usable at ``gsd_mpp``."""
    rows, cols = np.mgrid[0 : camera.height_px : step_px, 0 : camera.width_px : step_px]
    u, v = cols.ravel().astype(np.float64), rows.ravel().astype(np.float64)
    ok = source_valid(camera, u, v, gsd_mpp, max_off_nadir_deg)
    x, y, _ = camera.image_to_ground(u[ok], v[ok])
    return np.stack([x, y], axis=1)


def coverage(
    camera: TiltedCamera, window: ViewWindow, max_off_nadir_deg: float, samples: int = 24
) -> float:
    """Share of the window whose source is valid, from a ``samples x samples`` grid of points."""
    t = (np.arange(samples) + 0.5) / samples * window.side_m
    gx, gy = np.meshgrid(window.x0_m + t, window.y0_m + t)
    u, v, ahead = camera.ground_to_image(gx.ravel(), gy.ravel())
    ok = ahead & source_valid(camera, u, v, window.gsd_mpp, max_off_nadir_deg)
    return float(ok.mean())


def _tiles(
    camera: TiltedCamera, gsd_mpp: float, size_px: int, max_off_nadir_deg: float
) -> list[ViewWindow]:
    """Non-overlapping windows over the bounding box of the frame's usable footprint."""
    points = footprint_m(camera, gsd_mpp, max_off_nadir_deg)
    if len(points) == 0:
        return []
    side = gsd_mpp * size_px
    (x_lo, y_lo), (x_hi, y_hi) = points.min(axis=0), points.max(axis=0)
    nx, ny = max(1, math.ceil((x_hi - x_lo) / side)), max(1, math.ceil((y_hi - y_lo) / side))
    # Centre the tiling on the footprint's bounding box.
    x0 = (x_lo + x_hi) / 2 - nx * side / 2
    y0 = (y_lo + y_hi) / 2 - ny * side / 2
    return [
        ViewWindow(x0 + i * side, y0 + j * side, gsd_mpp, size_px)
        for j in range(ny)
        for i in range(nx)
    ]


def tile_windows(
    camera: TiltedCamera,
    gsd_mpp: float,
    size_px: int,
    max_off_nadir_deg: float,
    min_coverage: float,
) -> list[ViewWindow]:
    """Non-overlapping windows over the frame's usable footprint, aligned to its bounding box,
    that reach ``min_coverage``."""
    return [
        w
        for w in _tiles(camera, gsd_mpp, size_px, max_off_nadir_deg)
        if coverage(camera, w, max_off_nadir_deg) >= min_coverage
    ]


def can_serve(
    camera: TiltedCamera,
    gsd_mpp: float,
    size_px: int,
    max_off_nadir_deg: float,
    min_coverage: float,
) -> bool:
    """Whether the frame has a view at ``gsd_mpp``: one of its tiles reaches ``min_coverage``."""
    return any(
        coverage(camera, w, max_off_nadir_deg) >= min_coverage
        for w in _tiles(camera, gsd_mpp, size_px, max_off_nadir_deg)
    )


def servable(
    camera: TiltedCamera,
    gsds_mpp: Sequence[float],
    size_px: int,
    max_off_nadir_deg: float,
    min_coverage: float,
) -> frozenset[int]:
    """Indices of the GSDs in ``gsds_mpp`` at which the frame has a view (:func:`can_serve`)."""
    return frozenset(
        k
        for k, gsd in enumerate(gsds_mpp)
        if can_serve(camera, gsd, size_px, max_off_nadir_deg, min_coverage)
    )


def allocate_views(
    eligible: Sequence[int], n_views: int, max_per_frame: int, rng: np.random.Generator
) -> dict[int, int]:
    """Views per frame for one GSD bin: ``n_views`` dealt round-robin over the ``eligible`` frame
    ids in a seeded random order, at most ``max_per_frame`` each, so a bin with few frames gets
    fewer views rather than many near-copies."""
    order = [eligible[k] for k in rng.permutation(len(eligible))]
    total = min(n_views, max_per_frame * len(order))
    return dict(Counter(order[k % len(order)] for k in range(total)))


def random_windows(
    camera: TiltedCamera,
    gsd_mpp: float,
    size_px: int,
    max_off_nadir_deg: float,
    min_coverage: float,
    object_centres_m: NDArray[np.float64],
    n_views: int,
    object_share: float,
    rng: np.random.Generator,
    attempts: int = 20,
) -> list[ViewWindow]:
    """Up to ``n_views`` random windows; a share ``object_share`` centred near an object."""
    points = footprint_m(camera, gsd_mpp, max_off_nadir_deg)
    if len(points) == 0:
        return []
    side = gsd_mpp * size_px
    windows = []
    for _ in range(n_views):
        for _ in range(attempts):
            if len(object_centres_m) and rng.random() < object_share:
                cx, cy = object_centres_m[rng.integers(len(object_centres_m))]
                cx, cy = cx + rng.uniform(-0.4, 0.4) * side, cy + rng.uniform(-0.4, 0.4) * side
            else:
                cx, cy = points[rng.integers(len(points))]
            window = ViewWindow(cx - side / 2, cy - side / 2, gsd_mpp, size_px)
            if coverage(camera, window, max_off_nadir_deg) >= min_coverage:
                windows.append(window)
                break
    return windows


def object_centres_m(camera: TiltedCamera, boxes: Sequence[Box]) -> NDArray[np.float64]:
    """Ground positions (``n x 2``) of box centres that see the water."""
    if not boxes:
        return np.zeros((0, 2))
    u = np.array([b.x_px + b.w_px / 2 - 0.5 for b in boxes])
    v = np.array([b.y_px + b.h_px / 2 - 0.5 for b in boxes])
    x, y, water = camera.image_to_ground(u, v)
    return np.stack([x[water], y[water]], axis=1)


def labels_for(
    camera: TiltedCamera,
    window: ViewWindow,
    boxes: Sequence[Box],
    valid: NDArray[np.bool_],
    min_side_px: float = 1.0,
) -> list[Label]:
    """Boxes of ``boxes`` that belong in the view: centre on a valid pixel, and at least
    ``min_side_px`` wide and high after clipping. Unknown categories (``ignored``) are skipped."""
    kept = [b for b in boxes if b.category in CLASSES]
    if not kept:
        return []
    xywh = np.array([[b.x_px, b.y_px, b.w_px, b.h_px] for b in kept])
    projected, water = project_boxes(xywh, camera, window)
    n = window.size_px
    out = []
    for box, (x0, y0, x1, y1), ok in zip(kept, projected, water, strict=True):
        col, row = math.floor((x0 + x1) / 2), math.floor((y0 + y1) / 2)
        if not ok or not (0 <= col < n and 0 <= row < n) or not valid[row, col]:
            continue
        cx0, cy0, cx1, cy1 = max(x0, 0.0), max(y0, 0.0), min(x1, float(n)), min(y1, float(n))
        if cx1 - cx0 < min_side_px or cy1 - cy0 < min_side_px:
            continue
        out.append(Label(CLASSES.index(box.category), cx0, cy0, cx1, cy1))
    return out


def yolo_lines(labels: Sequence[Label], size_px: int) -> list[str]:
    """Labels as YOLO text lines: ``class cx cy w h``, normalised to the view size."""
    return [
        f"{lab.class_id} {(lab.x_min + lab.x_max) / 2 / size_px:.6f} "
        f"{(lab.y_min + lab.y_max) / 2 / size_px:.6f} {(lab.x_max - lab.x_min) / size_px:.6f} "
        f"{(lab.y_max - lab.y_min) / size_px:.6f}"
        for lab in labels
    ]


def rotate_quarter(
    image: NDArray[np.uint8], labels: Sequence[Label], turns: int
) -> tuple[NDArray[np.uint8], list[Label]]:
    """Rotate a square view and its labels by ``turns`` quarter turns counter-clockwise."""
    n = image.shape[0]
    out = [*labels]
    for _ in range(turns % 4):
        # Counter-clockwise: a point (x, y) moves to (y, n - x) in pixel-corner coordinates.
        out = [Label(b.class_id, b.y_min, n - b.x_max, b.y_max, n - b.x_min) for b in out]
    return np.ascontiguousarray(np.rot90(image, k=turns % 4)), out
