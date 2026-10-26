"""Bounded CPU rendering and batched detector inference, one complete world at a time."""

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from altiseek.cache.config import CacheConfig
from altiseek.cache.storage import DETECTION_DTYPE, TRUTH_DTYPE, CacheHeader, WorldCache, view_keys
from altiseek.detector.infer import Detection
from altiseek.world.library import Library
from altiseek.world.render import ViewObject, compose_world, render_view, view_window
from altiseek.world.spec import WorldSpec


class BatchDetector(Protocol):
    detector_id: str
    score_min: float
    iou: float
    max_det: int

    def detect(self, views: NDArray[np.uint8]) -> list[list[Detection]]: ...


def header_for(spec: WorldSpec, cfg: CacheConfig) -> CacheHeader:
    if spec.world_id != spec.content_id():
        raise ValueError("world_id does not match content")
    if spec.area_m != cfg.render.levels.area_m or spec.cell_m != cfg.render.levels.cell_m:
        raise ValueError("world and level geometry mismatch")
    return CacheHeader(
        world_id=spec.world_id,
        detector_id=cfg.detector_id,
        render=cfg.render,
        tau_min=cfg.tau_min,
        inference=cfg.inference,
    ).identified()


def check_detector(detector: BatchDetector, cfg: CacheConfig) -> None:
    if detector.detector_id != cfg.detector_id:
        raise ValueError("detector_id mismatch")
    if (
        detector.score_min != cfg.tau_min
        or detector.iou != cfg.inference.iou
        or detector.max_det != cfg.inference.max_det
    ):
        raise ValueError("detector inference settings mismatch")


def render_key(
    spec: WorldSpec,
    composed: NDArray[np.uint8],
    cfg: CacheConfig,
    key: NDArray[np.int32],
) -> tuple[NDArray[np.uint8], tuple[ViewObject, ...]]:
    li, ix, iy, variant = (int(k) for k in key)
    return render_view(
        spec,
        composed,
        table=cfg.render.levels,
        camera=cfg.render.camera,
        level=cfg.render.levels.levels[li],
        cell=(ix, iy),
        variant=variant,
        settings=cfg.render.variants,
    )


def records(
    spec: WorldSpec,
    cfg: CacheConfig,
    key: NDArray[np.int32],
    detections: list[Detection],
) -> np.ndarray:
    """Project pixel-corner boxes through the shifted footprint into world metres/cells."""
    li, ix, iy, variant = (int(k) for k in key)
    table, camera = cfg.render.levels, cfg.render.camera
    row0, col0, n, _ = view_window(
        spec, table, table.levels[li], (ix, iy), variant, cfg.render.variants, camera.resolution_px
    )
    rows = []
    for d in detections:
        if d.score < cfg.tau_min:
            continue
        col = (d.x_min + d.x_max) / 2
        row = (d.y_min + d.y_max) / 2
        x = (col0 + col * n / camera.resolution_px) * spec.gsd_mpp
        y = spec.area_m - (row0 + row * n / camera.resolution_px) * spec.gsd_mpp
        rows.append(
            (
                d.class_id,
                d.score,
                (d.x_min, d.y_min, d.x_max, d.y_max),
                x,
                y,
                int(np.floor(x / spec.cell_m)),
                int(np.floor(y / spec.cell_m)),
            )
        )
    return np.asarray(rows, dtype=DETECTION_DTYPE)


def truth_records(spec: WorldSpec, truth: tuple[ViewObject, ...]) -> np.ndarray:
    return np.asarray(
        [
            (
                o.x_m,
                o.y_m,
                int(np.floor(o.x_m / spec.cell_m)),
                int(np.floor(o.y_m / spec.cell_m)),
                o.col_px,
                o.row_px,
            )
            for o in truth
            if o.target
        ],
        dtype=TRUTH_DTYPE,
    )


def build_world(
    spec: WorldSpec,
    library: Library,
    cfg: CacheConfig,
    detector: BatchDetector,
) -> tuple[WorldCache, dict[str, float]]:
    """Compose once; bound temporary images to one batch and active render workers."""
    check_detector(detector, cfg)
    header = header_for(spec, cfg)
    start = time.perf_counter()
    composed = compose_world(spec, library)
    compose_s = time.perf_counter() - start
    keys = view_keys(cfg.render)
    offsets, truth_offsets = [0], [0]
    detected, truths = [], []
    render_s, detect_s = 0.0, 0.0

    def render(key: NDArray[np.int32]) -> tuple[NDArray[np.uint8], tuple[ViewObject, ...]]:
        return render_key(spec, composed, cfg, key)

    with ThreadPoolExecutor(max_workers=cfg.render_workers) as pool:
        for begin in range(0, len(keys), cfg.batch_size):
            batch_keys = keys[begin : begin + cfg.batch_size]
            tick = time.perf_counter()
            batch = list(pool.map(render, batch_keys))
            images = np.stack([v[0] for v in batch])
            render_s += time.perf_counter() - tick
            tick = time.perf_counter()
            predictions = detector.detect(images)
            detect_s += time.perf_counter() - tick
            for key, (_, truth), prediction in zip(batch_keys, batch, predictions, strict=True):
                d, t = records(spec, cfg, key, prediction), truth_records(spec, truth)
                detected.append(d)
                truths.append(t)
                offsets.append(offsets[-1] + len(d))
                truth_offsets.append(truth_offsets[-1] + len(t))
    cache = WorldCache(
        header,
        keys,
        np.asarray(offsets, dtype=np.int64),
        np.concatenate(detected),
        np.asarray(truth_offsets, dtype=np.int64),
        np.concatenate(truths),
    )
    return cache, {
        "compose_s": compose_s,
        "render_s": render_s,
        "detect_s": detect_s,
        "total_s": time.perf_counter() - start,
    }
