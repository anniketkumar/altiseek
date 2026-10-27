"""Cell outcomes on real rectified frames, using a fixed camera-ground grid."""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from altiseek.calibration.config import SensorConfig
from altiseek.detector.infer import Detection
from altiseek.detector.views import Label
from altiseek.geometry.rectify import ViewWindow


def real_outcomes(
    window: ViewWindow,
    valid: np.ndarray,
    truth: Sequence[Label],
    detections: Sequence[Detection],
    cell_m: float,
    cfg: SensorConfig,
) -> dict[str, Any]:
    """Cell centres must lie on valid source pixels, excluding letterbox fill."""
    size, gsd = window.size_px, window.gsd_mpp
    if valid.shape != (size, size):
        raise ValueError("valid mask shape mismatch")
    x0, y0, x1, y1 = (
        math.ceil(v / cell_m - 0.5)
        for v in (
            window.x0_m,
            window.y0_m,
            window.x0_m + window.side_m,
            window.y0_m + window.side_m,
        )
    )
    observed = set()
    for iy in range(y0, y1):
        row = math.floor((window.y0_m + window.side_m - (iy + 0.5) * cell_m) / gsd)
        for ix in range(x0, x1):
            col = math.floor(((ix + 0.5) * cell_m - window.x0_m) / gsd)
            if 0 <= row < size and 0 <= col < size and valid[row, col]:
                observed.add((ix, iy))

    def cell(box: Label | Detection) -> tuple[int, int]:
        x = window.x0_m + (box.x_min + box.x_max) / 2 * gsd
        y = window.y0_m + window.side_m - (box.y_min + box.y_max) / 2 * gsd
        return math.floor(x / cell_m), math.floor(y / cell_m)

    occupied = {cell(t) for t in truth if t.class_id == 0} & observed
    outcomes: dict[tuple[int, int], float] = {}
    for d in detections:
        key = cell(d)
        if d.class_id == 0 and d.score >= cfg.tau_min and key in observed:
            outcomes[key] = max(outcomes.get(key, 0.0), d.score)
    ordered = sorted(outcomes)
    counts = []
    for threshold in (cfg.tau, cfg.tau_min):
        positive = {p for p, s in outcomes.items() if s >= threshold}
        counts.append(
            [
                len(occupied),
                len(observed - occupied),
                len(positive & occupied),
                len(positive - occupied),
            ]
        )
    return {
        "counts": counts,
        "scores": [outcomes[p] for p in ordered],
        "labels": [p in occupied for p in ordered],
    }
