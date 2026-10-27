"""One-parameter temperature scaling and equal-width reliability summaries."""

from typing import Any

import numpy as np
from numpy.typing import NDArray


def calibrated(scores: NDArray[np.float64], temperature: float) -> NDArray[np.float64]:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("scores must be finite probabilities")
    s = np.clip(scores, 1e-12, 1 - 1e-12)
    z = (np.log(s) - np.log1p(-s)) / temperature
    return np.exp(-np.logaddexp(0, -z))


def fit_temperature(
    scores: NDArray[np.float64], labels: NDArray[np.bool_], minimum: float, maximum: float
) -> float:
    """BCE is convex in inverse temperature: bisect its monotone derivative."""
    if scores.size == 0 or scores.shape != labels.shape:
        raise ValueError("calibration needs paired, nonempty scores and labels")
    calibrated(scores, 1.0)  # validate the score domain
    s = np.clip(scores, 1e-12, 1 - 1e-12)
    logits = np.log(s) - np.log1p(-s)
    lo, hi = 1 / maximum, 1 / minimum
    for _ in range(80):
        mid = (lo + hi) / 2
        derivative = np.mean(logits * (calibrated(scores, 1 / mid) - labels))
        if derivative > 0:
            hi = mid
        else:
            lo = mid
    return 1 / ((lo + hi) / 2)


def reliability(
    scores: NDArray[np.float64], labels: NDArray[np.bool_], bins: int
) -> dict[str, Any]:
    """Empty bins have null means; ECE is weighted by detected-cell count."""
    calibrated(scores, 1.0)
    if scores.shape != labels.shape:
        raise ValueError("unpaired scores/labels")
    edges = np.linspace(0, 1, bins + 1)
    indices = np.minimum(np.searchsorted(edges, scores, side="right") - 1, bins - 1)
    rows, error = [], 0.0
    for i in range(bins):
        mask = indices == i
        count = int(mask.sum())
        confidence = float(scores[mask].mean()) if count else None
        accuracy = float(labels[mask].mean()) if count else None
        if count:
            error += count * abs(confidence - accuracy)
        rows.append({"count": count, "confidence": confidence, "accuracy": accuracy})
    s = np.clip(scores, 1e-12, 1 - 1e-12)
    return {
        "count": len(scores),
        "ece": error / len(scores) if len(scores) else None,
        "bce": float(-(labels * np.log(s) + (1 - labels) * np.log1p(-s)).mean())
        if len(scores)
        else None,
        "edges": edges.tolist(),
        "bins": rows,
    }
