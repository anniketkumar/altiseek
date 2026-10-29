"""Log-odds belief, first-look gating and verified-cell resolution; no hidden truth."""

import numpy as np

from altiseek.calibration.model import SensorModel
from altiseek.calibration.outcomes import cell_outcomes


def entropy(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-12, 1 - 1e-12)
    return -(q * np.log(q) + (1 - q) * np.log1p(-q))


class Belief:
    def __init__(
        self,
        prior: np.ndarray,
        count: float,
        levels: int,
        model: SensorModel,
        relook_rule: str = "first",
    ) -> None:
        if prior.ndim != 2 or (prior < 0).any() or not np.isclose(prior.sum(), 1) or count <= 0:
            raise ValueError("invalid prior/count")
        self.prior, self.model = prior.copy(), model
        if relook_rule not in ("first", "distinct_variant"):
            raise ValueError("unknown re-look rule")
        self.relook_rule = relook_rule
        self.limit = model.fit_config.log_odds_limit
        p = np.clip(
            -np.expm1(count * np.log1p(-np.minimum(prior, 1 - 1e-12))),
            1 / (1 + np.exp(self.limit)),
            1 - 1 / (1 + np.exp(self.limit)),
        )
        self.log_odds = np.log(p) - np.log1p(-p)
        self.observed = np.zeros((levels, *prior.shape), dtype=bool)
        self.used_variants = np.zeros((levels, *prior.shape), dtype=np.uint32)
        self.resolved = np.zeros(prior.shape, dtype=bool)

    @property
    def probabilities(self) -> np.ndarray:
        p = np.exp(-np.logaddexp(0, -self.log_odds))
        p[self.resolved] = 0
        return p

    def update(
        self,
        detections: np.ndarray,
        bounds: tuple[int, int, int, int],
        level: int,
        variant: int,
        sensor_variant: str,
        found_positions_m: np.ndarray,
        suppression_radius_m: float,
    ) -> int:
        if not 0 <= variant < 32:
            raise ValueError("variant outside bitset")
        if len(found_positions_m):
            points = np.column_stack((detections["x_m"], detections["y_m"]))
            distance2 = ((points[:, None] - found_positions_m[None, :]) ** 2).sum(axis=2)
            detections = detections[(distance2 > suppression_radius_m**2).all(axis=1)]
        x0, y0, x1, y1 = bounds
        region = np.s_[x0:x1, y0:y1]
        informative = ~self.observed[level][region] & ~self.resolved[region]
        if self.relook_rule == "distinct_variant":
            informative = (
                (self.used_variants[level][region] & np.uint32(1 << variant)) == 0
            ) & ~self.resolved[region]
        scores = np.zeros((x1 - x0, y1 - y0))
        for (ix, iy), score in cell_outcomes(detections, bounds).items():
            scores[ix - x0, iy - y0] = score
        if informative.any():
            odds = self.log_odds[region]
            p = np.exp(-np.logaddexp(0, -odds[informative]))
            posterior = self.model.posterior(p, level, scores[informative], sensor_variant)
            odds[informative] = np.clip(
                np.log(posterior) - np.log1p(-posterior), -self.limit, self.limit
            )
        self.observed[level][region] = True
        self.used_variants[level][region] |= np.uint32(1 << variant)
        return int(informative.sum())

    def resolve(self, cells: np.ndarray) -> None:
        if len(cells):
            self.resolved[cells[:, 0], cells[:, 1]] = True
            self.log_odds[cells[:, 0], cells[:, 1]] = -self.limit
