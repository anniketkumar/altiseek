"""Re-render seeded random views and compare live detections with cached records."""

import numpy as np

from altiseek.cache.build import (
    BatchDetector,
    check_detector,
    header_for,
    records,
    render_key,
    truth_records,
)
from altiseek.cache.config import CacheConfig
from altiseek.cache.storage import WorldCache
from altiseek.detector.infer import Detection
from altiseek.world.library import Library
from altiseek.world.render import compose_world
from altiseek.world.spec import WorldSpec


def compare_detections(
    cached: np.ndarray,
    live: np.ndarray,
    *,
    score_atol: float = 1e-4,
    box_atol_px: float = 0.05,
) -> None:
    """Class-preserving bijection under score/box tolerances, independent of output order."""
    if len(cached) != len(live):
        raise ValueError(f"detection count mismatch: {len(cached)} != {len(live)}")
    if not len(cached):
        return
    edges = (
        (cached["class_id"][:, None] == live["class_id"][None, :])
        & (np.abs(cached["score"][:, None] - live["score"][None, :]) <= score_atol)
        & (
            np.max(np.abs(cached["box_px"][:, None] - live["box_px"][None, :]), axis=2)
            <= box_atol_px
        )
    )
    matched: dict[int, int] = {}

    def augment(i: int, seen: set[int]) -> bool:
        for j in np.flatnonzero(edges[i]):
            j = int(j)
            if j not in seen:
                seen.add(j)
                if j not in matched or augment(matched[j], seen):
                    matched[j] = i
                    return True
        return False

    if not all(augment(i, set()) for i in range(len(cached))):
        raise ValueError("live detection class/score/box mismatch")


def validate_world(
    cache: WorldCache,
    spec: WorldSpec,
    library: Library,
    cfg: CacheConfig,
    detector: BatchDetector,
    *,
    seed: int,
    samples: int = 20,
    score_atol: float = 1e-4,
    box_atol_px: float = 0.05,
) -> list[int]:
    check_detector(detector, cfg)
    if cache.header != header_for(spec, cfg):
        raise ValueError("cache header mismatch")
    if samples < 1:
        raise ValueError("samples must be positive")
    if score_atol < 0 or box_atol_px < 0:
        raise ValueError("validation tolerances must be nonnegative")
    cache.validate()
    composed = compose_world(spec, library)
    indices = sorted(
        int(i)
        for i in np.random.default_rng(seed).choice(
            len(cache.views), min(samples, len(cache.views)), replace=False
        )
    )
    for begin in range(0, len(indices), cfg.batch_size):
        chosen = indices[begin : begin + cfg.batch_size]
        batch = [render_key(spec, composed, cfg, cache.views[i]) for i in chosen]
        predictions = detector.detect(np.stack([v[0] for v in batch]))
        for i, (_, truth), prediction in zip(chosen, batch, predictions, strict=True):
            key = cache.views[i]
            cached_d, cached_t = cache.lookup(*(int(k) for k in key))
            projected = records(
                spec,
                cfg,
                key,
                [
                    Detection(
                        int(d["class_id"]), float(d["score"]), *(float(x) for x in d["box_px"])
                    )
                    for d in cached_d
                ],
            )
            for field in ("x_m", "y_m", "ix", "iy"):
                if not np.allclose(cached_d[field], projected[field], rtol=0, atol=1e-7):
                    raise ValueError("cached detection world projection mismatch")
            live = records(spec, cfg, key, prediction)
            compare_detections(cached_d, live, score_atol=score_atol, box_atol_px=box_atol_px)
            if not np.array_equal(cached_t, truth_records(spec, truth)):
                raise ValueError("cached target truth mismatch")
    return indices
