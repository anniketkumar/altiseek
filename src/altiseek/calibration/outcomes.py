"""Sparse cell outcomes; millions of empty observations need only integer counts.

Occupancy uses full world truth, including a target just outside the shifted view
whose cell centre is still observed. Such an observation is a legitimate miss.
"""

from dataclasses import dataclass

import numpy as np

from altiseek.cache.storage import WorldCache
from altiseek.calibration.config import SensorConfig
from altiseek.world.render import view_window
from altiseek.world.spec import WorldSpec


def cell_outcomes(
    detections: np.ndarray, bounds: tuple[int, int, int, int]
) -> dict[tuple[int, int], float]:
    """Maximum raw swimmer score per observed centre cell; bounds are half-open."""
    x0, y0, x1, y1 = bounds
    result: dict[tuple[int, int], float] = {}
    for d in detections:
        x, y = int(d["ix"]), int(d["iy"])
        if d["class_id"] == 0 and x0 <= x < x1 and y0 <= y < y1:
            key = (x, y)
            result[key] = max(result.get(key, 0.0), float(d["score"]))
    return result


def observed_bounds(
    spec: WorldSpec, cache: WorldCache, key: np.ndarray
) -> tuple[int, int, int, int]:
    """Cells with centres in the actual shifted footprint, south-west coordinates."""
    li, ix, iy, variant = map(int, key)
    render = cache.header.render
    row, col, side, _ = view_window(
        spec,
        render.levels,
        render.levels.levels[li],
        (ix, iy),
        variant,
        render.variants,
        render.camera.resolution_px,
    )
    x0 = col * spec.gsd_mpp
    y0 = spec.area_m - (row + side) * spec.gsd_mpp
    width = side * spec.gsd_mpp
    # A centre at an inclusive lower edge is observed; upper edge is exclusive.
    return tuple(int(np.ceil(v / spec.cell_m - 0.5)) for v in (x0, y0, x0 + width, y0 + width))


@dataclass
class WorldOutcomes:
    world_id: str
    # [level, threshold (tau, tau_min), occupied/empty/true-positive/false-positive]
    counts: np.ndarray
    # [level, threshold, recovered-on-second/missed-first], all ordered variant pairs
    relook: np.ndarray
    levels: np.ndarray
    scores: np.ndarray
    labels: np.ndarray


def collect_world(spec: WorldSpec, cache: WorldCache, cfg: SensorConfig) -> WorldOutcomes:
    if cache.header.world_id != spec.world_id or cache.header.tau_min != cfg.tau_min:
        raise ValueError("world/score cutoff mismatch")
    table = cache.header.render.levels
    occupied = {(int(t.x_m // spec.cell_m), int(t.y_m // spec.cell_m)) for t in spec.targets}
    counts = np.zeros((len(table.levels), 2, 4), dtype=np.int64)
    relook = np.zeros((len(table.levels), 2, 2), dtype=np.int64)
    levels, scores, labels = [], [], []
    # The same occupied cell must be observed in both looks to enter the pair.
    groups: dict[tuple[int, int, int], list[dict[tuple[int, int], float]]] = {}
    for key in cache.views:
        li, ix, iy, variant = map(int, key)
        bounds = observed_bounds(spec, cache, key)
        x0, y0, x1, y1 = bounds
        truth = {p for p in occupied if x0 <= p[0] < x1 and y0 <= p[1] < y1}
        dets, _ = cache.lookup(li, ix, iy, variant)
        outcome = cell_outcomes(dets, bounds)
        groups.setdefault((li, ix, iy), []).append({p: outcome.get(p, 0.0) for p in truth})
        for cell, score in outcome.items():
            levels.append(li)
            scores.append(score)
            labels.append(cell in truth)
        for ti, threshold in enumerate((cfg.tau, cfg.tau_min)):
            positive = {p for p, s in outcome.items() if s >= threshold}
            counts[li, ti] += (
                len(truth),
                (x1 - x0) * (y1 - y0) - len(truth),
                len(positive & truth),
                len(positive - truth),
            )
    for (li, _, _), looks in groups.items():
        for a, first in enumerate(looks):
            for b, second in enumerate(looks):
                if a == b:
                    continue
                for cell in first.keys() & second.keys():
                    for ti, threshold in enumerate((cfg.tau, cfg.tau_min)):
                        if first[cell] < threshold:
                            relook[li, ti, 1] += 1
                            relook[li, ti, 0] += second[cell] >= threshold
    return WorldOutcomes(
        spec.world_id,
        counts,
        relook,
        np.asarray(levels, dtype=np.int32),
        np.asarray(scores, dtype=np.float64),
        np.asarray(labels, dtype=bool),
    )
