"""World renderer (CLAUDE.md §11): a world composed at g0 from its spec and library, and the
simulated camera's view of it at any level, lattice cell and variant.

**Composition** (:func:`compose_world`) is deterministic.
1. Background patches are turned and mirrored as placed and colour-harmonised to the world's
   appearance condition (Reinhard transfer in Lab towards the condition's mean statistics). They
   are blended with feathered weights, which fall linearly to zero across each patch's overlap
   with its neighbours. Overlapping patches are averaged by weight.
2. Objects: each crop is rotated and scaled about its object position, and its colours are
   shifted so its water ring matches the mean Lab of the background beneath it. It is then
   alpha-composited.

The world is composed in horizontal bands, so the float accumulators stay small. One
composition serves every view of the world.

**Views** (:func:`render_view`): the level's footprint at the lattice cell is cut from the
composed world, shifted by a seeded number of whole g0 pixels below one view pixel. It is
photometrically jittered (exposure, white balance, glint), then finished by
``altiseek.imaging.pipeline.render_view``, the same path every real evaluation view took.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from altiseek.data.horizon import LUMA
from altiseek.geometry.levels import Level, LevelTable
from altiseek.geometry.raster import Raster
from altiseek.imaging import pipeline
from altiseek.imaging.config import CameraSpec
from altiseek.world.appearance import Stats, stats, transfer
from altiseek.world.config import VariantSettings
from altiseek.world.library import BackgroundRow, Library, ObjectRow
from altiseek.world.spec import WorldSpec


def _png(path: object) -> NDArray[np.uint8]:
    with Image.open(path) as image:
        return np.asarray(image)


def feather(size_px: int, overlap_px: int) -> NDArray[np.float32]:
    """Separable weights of a square patch: 1 inside, falling linearly to 0 over ``overlap_px``
    at each edge (never quite 0, so a lone patch at the world's edge keeps its colour)."""
    i = np.arange(size_px, dtype=np.float32)
    ramp = np.clip(np.minimum(i + 0.5, size_px - i - 0.5) / max(overlap_px, 1), 0.0, 1.0)
    return np.outer(ramp, ramp)


def _row_stats(row: BackgroundRow) -> Stats:
    return Stats(
        (row.l_mean, row.a_mean, row.b_mean), (row.l_std, row.a_std, row.b_std), row.texture
    )


def _condition_stats(entry: dict[str, float]) -> Stats:
    return Stats(
        (entry["l_mean"], entry["a_mean"], entry["b_mean"]),
        (entry["l_std"], entry["a_std"], entry["b_std"]),
        entry["texture"],
    )


def _water_stats(row: ObjectRow) -> Stats:
    return Stats(
        (row.water_l, row.water_a, row.water_b),
        (row.water_l_std, row.water_a_std, row.water_b_std),
        0.0,
    )


def transformed(
    rgba: NDArray[np.uint8], centre_px: tuple[float, float], angle_deg: float, scale: float
) -> tuple[NDArray[np.uint8], tuple[float, float]]:
    """``rgba`` rotated anticlockwise by ``angle_deg`` and scaled about ``centre_px`` (column,
    row) into a canvas that holds all of it; returns the canvas and where the centre went."""
    h, w = rgba.shape[:2]
    m = cv2.getRotationMatrix2D(centre_px, angle_deg, scale)
    corners = np.array([[0.0, 0.0], [w, 0.0], [0.0, h], [w, h]])
    moved = corners @ m[:, :2].T + m[:, 2]
    lo, hi = np.floor(moved.min(axis=0)), np.ceil(moved.max(axis=0))
    m[:, 2] -= lo
    size = (int(hi[0] - lo[0]), int(hi[1] - lo[1]))
    canvas = cv2.warpAffine(
        rgba,
        m,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    cx, cy = m @ np.array([centre_px[0], centre_px[1], 1.0])
    return canvas, (float(cx), float(cy))


def compose_world(spec: WorldSpec, library: Library, band_px: int = 1024) -> NDArray[np.uint8]:
    """The world at g0: ``H x W x 3`` uint8, row 0 at the north edge (``geometry.raster``)."""
    if library.library_id != spec.library_id:
        raise ValueError(
            f"world {spec.world_id} needs library {spec.library_id}, not {library.library_id}"
        )
    raster = Raster.covering(spec.area_m, spec.area_m, spec.gsd_mpp)
    height, width = raster.height_px, raster.width_px
    p = library.patch_px
    xs = sorted({pl.x_m for pl in spec.background})
    step_px = round((xs[1] - xs[0]) / spec.gsd_mpp) if len(xs) > 1 else p
    weight = feather(p, p - step_px)
    target = _condition_stats(library.conditions[spec.condition])
    rows = {r.patch_id: r for r in library.backgrounds}
    harmonised: dict[str, NDArray[np.uint8]] = {}
    spans = []
    for pl in spec.background:
        col, row = raster.to_pixel(pl.x_m, pl.y_m)
        c0, r0 = round(float(col) + 0.5 - p / 2), round(float(row) + 0.5 - p / 2)
        spans.append((r0, c0, pl))
    composed = np.empty((height, width, 3), dtype=np.uint8)
    for b0 in range(0, height, band_px):
        b1 = min(height, b0 + band_px)
        acc = np.zeros((b1 - b0, width, 3), dtype=np.float32)
        total = np.zeros((b1 - b0, width), dtype=np.float32)
        for r0, c0, pl in spans:
            rr0, rr1 = max(r0, b0), min(r0 + p, b1)
            cc0, cc1 = max(c0, 0), min(c0 + p, width)
            if rr0 >= rr1 or cc0 >= cc1:
                continue
            if pl.patch_id not in harmonised:
                harmonised[pl.patch_id] = transfer(
                    _png(library.patch_path(pl.patch_id)), _row_stats(rows[pl.patch_id]), target
                )
            image = np.rot90(harmonised[pl.patch_id], k=pl.turns)
            if pl.flip:
                image = image[:, ::-1]
            sub = (slice(rr0 - r0, rr1 - r0), slice(cc0 - c0, cc1 - c0))
            w = weight[sub]
            acc[rr0 - b0 : rr1 - b0, cc0:cc1] += image[sub].astype(np.float32) * w[..., None]
            total[rr0 - b0 : rr1 - b0, cc0:cc1] += w
        composed[b0:b1] = np.clip(np.rint(acc / total[..., None]), 0, 255).astype(np.uint8)

    crops = {c.crop_id: c for c in library.objects}
    for obj in spec.objects:
        row_info = crops[obj.crop_id]
        rgba, (cx, cy) = transformed(
            _png(library.crop_path(obj.crop_id)),
            (row_info.centre_col, row_info.centre_row),
            obj.rotation_deg,
            obj.scale,
        )
        col, row = raster.to_pixel(obj.x_m, obj.y_m)
        c0, r0 = round(float(col) - cx), round(float(row) - cy)
        h, w = rgba.shape[:2]
        rr0, rr1, cc0, cc1 = max(r0, 0), min(r0 + h, height), max(c0, 0), min(c0 + w, width)
        if rr0 >= rr1 or cc0 >= cc1:
            continue
        piece = rgba[rr0 - r0 : rr1 - r0, cc0 - c0 : cc1 - c0]
        under = composed[rr0:rr1, cc0:cc1]
        alpha = piece[..., 3:].astype(np.float32) / 255.0
        ring = (piece[..., 3] > 0) & (piece[..., 3] < 255)
        if ring.any():
            local = stats(under, ring)
            colour = transfer(piece[..., :3], _water_stats(row_info), local, scale=False)
        else:
            colour = piece[..., :3]
        blended = colour.astype(np.float32) * alpha + under.astype(np.float32) * (1.0 - alpha)
        composed[rr0:rr1, cc0:cc1] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    return composed


@dataclass(frozen=True)
class ViewObject:
    """An object in a view: its class, whether it is a target, and where it is on the ground
    and in the view (column, row; pixel centres at integers)."""

    category: str
    target: bool
    x_m: float
    y_m: float
    col_px: float
    row_px: float


def jittered(
    region: NDArray[np.uint8], rng: np.random.Generator, settings: VariantSettings
) -> NDArray[np.uint8]:
    """Exposure and white-balance gains, then highlights above the glint threshold brightened or
    dimmed."""
    gain = (1.0 + rng.uniform(-settings.exposure, settings.exposure)) * (
        1.0 + rng.uniform(-settings.white_balance, settings.white_balance, 3)
    )
    glint = rng.uniform(-settings.glint, settings.glint)
    out = region.astype(np.float32) * gain.astype(np.float32)
    luma = out @ LUMA.astype(np.float32)
    out += (np.maximum(luma - settings.glint_threshold, 0.0) * glint)[..., None]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def render_view(
    spec: WorldSpec,
    composed: NDArray[np.uint8],
    *,
    table: LevelTable,
    camera: CameraSpec,
    level: Level,
    cell: tuple[int, int],
    variant: int,
    settings: VariantSettings,
) -> tuple[NDArray[np.uint8], tuple[ViewObject, ...]]:
    """The camera's view of the footprint of ``level`` whose south-west cell is ``cell``
    (``(ix, iy)``), and the objects in it."""
    raster = Raster(spec.gsd_mpp, composed.shape[1], composed.shape[0])
    row0, col0, n = raster.window(
        cell[0] * table.cell_m, cell[1] * table.cell_m, table.footprint_m(level)
    )
    size = camera.resolution_px
    rng = np.random.default_rng([spec.seed, table.levels.index(level), cell[0], cell[1], variant])
    reach = math.floor(n / size * settings.shift_view_px)
    dc, dr = (int(v) for v in rng.integers(-reach, reach + 1, 2)) if reach else (0, 0)
    col0 = min(max(col0 + dc, 0), raster.width_px - n)
    row0 = min(max(row0 + dr, 0), raster.height_px - n)
    region = jittered(composed[row0 : row0 + n, col0 : col0 + n], rng, settings)
    image = pipeline.render_view(region, size, size, camera.pipeline)
    truth = []
    for obj in spec.objects:
        col, row = raster.to_pixel(obj.x_m, obj.y_m)
        vc = (float(col) - col0 + 0.5) * size / n - 0.5
        vr = (float(row) - row0 + 0.5) * size / n - 0.5
        if -0.5 <= vc < size - 0.5 and -0.5 <= vr < size - 0.5:
            truth.append(ViewObject(obj.category, obj.target, obj.x_m, obj.y_m, vc, vr))
    return image, tuple(truth)
