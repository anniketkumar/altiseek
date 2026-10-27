"""Expected entropy reduction under each sensor's assumed observation model."""

import numpy as np

from altiseek.belief.state import entropy
from altiseek.calibration.model import SensorModel


def information(
    p: np.ndarray, model: SensorModel, level: int, variant: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return mutual information (nats) and expected true-positive mass per cell.

    V0 has no coherent score distribution: its planner uses the stated r0/f=0
    binary surrogate. V3 integrates its finite calibrated-score histogram exactly.
    """
    if variant == "V0":
        r, f = model.fit_config.naive_recall, 0.0
        occupied, empty = np.array([1 - r, r]), np.array([1 - f, f])
    else:
        lm = model.levels[level]
        rates = (
            model.global_binary if variant == "V1" else lm.binary if variant == "V2" else lm.soft
        )
        r, f = rates.recall, rates.false_alarm
        if variant == "V3":
            occupied = np.r_[1 - r, r * np.asarray(lm.occupied_hist)]
            empty = np.r_[1 - f, f * np.asarray(lm.empty_hist)]
        elif variant in ("V1", "V2"):
            occupied, empty = np.array([1 - r, r]), np.array([1 - f, f])
        else:
            raise ValueError("unknown sensor variant")
    expected = np.zeros_like(p)
    for a, b in zip(occupied, empty, strict=True):
        probability = p * a + (1 - p) * b
        posterior = np.divide(p * a, probability, out=np.zeros_like(p), where=probability > 0)
        expected += probability * entropy(posterior)
    return np.maximum(0, entropy(p) - expected), p * r
