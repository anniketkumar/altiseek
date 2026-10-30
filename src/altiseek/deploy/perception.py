"""Convert live detector boxes to the common evidence contract at actual pose."""

import numpy as np

from altiseek.cache.storage import DETECTION_DTYPE
from altiseek.deploy.projection import GroundCamera
from altiseek.detector.infer import Detection


def project_detections(
    detections: list[Detection], camera: GroundCamera, cell_m: float, tau_min: float
) -> np.ndarray:
    """Box coordinates are pixel boundaries; projection coordinates are centres.

    Subtract half a pixel before projection, as in the shared rectification path.
    Preserve raw scores and centre-cell matching. Visibility filtering belongs
    to the common belief updater, including the live whole-cell mask.
    """
    if not np.isfinite(cell_m) or cell_m <= 0 or not 0 < tau_min < 1:
        raise ValueError("Invalid cell size or detector cutoff")
    rows = []
    for d in detections:
        if d.score < tau_min:
            continue
        box = np.array([d.x_min, d.y_min, d.x_max, d.y_max])
        if (
            not np.isfinite(box).all()
            or not np.isfinite(d.score)
            or not 0 <= d.score <= 1
            or np.any(box[2:] < box[:2])
        ):
            raise ValueError("Invalid detector box/score")
        xy = camera.pixel_to_ground((box[:2] + box[2:]) / 2 - 0.5)
        ix, iy = np.floor(xy / cell_m).astype(int)
        rows.append((d.class_id, d.score, box, *xy, ix, iy))
    return np.asarray(rows, dtype=DETECTION_DTYPE)
