"""Random, oracle, sweeps and one/two-step information policies."""

from functools import lru_cache
from typing import Any

import numpy as np

from altiseek.belief.information import information
from altiseek.calibration.model import SensorModel
from altiseek.env.navigation import Navigation, Node


class Baseline:
    def __init__(
        self,
        name: str,
        navigation: Navigation,
        sensor: SensorModel,
        variant: str,
        *,
        seed: int,
        parameters: dict[str, Any] | None = None,
        oracle_cells: np.ndarray | None = None,
    ) -> None:
        if name not in ("random", "oracle", "B1", "B2", "B3", "B4", "B5"):
            raise ValueError("unknown baseline")
        if (name == "oracle") != (oracle_cells is not None):
            raise ValueError("only oracle may receive target cells")
        self.name, self.nav, self.sensor, self.variant = name, navigation, sensor, variant
        self.parameters = parameters or {}
        self.oracle_cells = oracle_cells
        self.rng = np.random.default_rng(seed)
        self.visited: set[int] = set()
        self.goal: int | None = None
        self.tour: list[int] | None = None
        self._conditional: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def _decode(
        self, observation: dict[str, np.ndarray]
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        count = len(self.nav.table.levels)
        maps = observation["map"][:, ::-1, :].transpose(0, 2, 1)
        level = int(np.argmax(observation["vector"][:count]))
        ix, iy = np.unravel_index(np.argmax(maps[count + 4]), maps[0].shape)
        half = self.nav.table.levels[level].footprint_cells // 2
        node = self.nav.index[Node(level, int(ix) - half, int(iy) - half)]
        return (
            node,
            maps[0].astype(float),
            maps[1],
            maps[2 : count + 2].astype(bool),
            maps[count + 2].astype(bool),
        )

    def act(self, observation: dict[str, np.ndarray], mask: np.ndarray) -> int:
        node, p, prior, observed, resolved = self._decode(observation)
        valid = np.flatnonzero(mask)
        if not len(valid):
            raise ValueError("no valid action")
        if self.name == "random":
            return int(self.rng.choice(valid))
        if self.name in ("B4", "B5"):
            return self._information_action(node, p, observed, resolved, valid)
        if self.name == "oracle":
            cells = self.oracle_cells
            cells = cells[~resolved[cells[:, 0], cells[:, 1]]]
            candidates = [
                i
                for i, n in enumerate(self.nav.nodes)
                if n.level == 0 and any(self._contains(i, cell) for cell in cells)
            ]
            if candidates:
                goal = min(candidates, key=lambda i: self.nav.distances[node, i])
                if goal != node:
                    return int(self.nav.first_actions[node, goal])
            return int(valid[0])
        level = int(
            self.parameters.get("coarse", 4)
            if self.name == "B3"
            else self.parameters.get("level", 0)
        )
        if self.nav.nodes[node].level == level:
            self.visited.add(node)
        if self.goal == node:
            self.goal = None
        # A new candidate interrupts a coarse sweep, but not an ongoing confirmation.
        if self.name == "B3" and (self.goal is None or self.nav.nodes[self.goal].level == level):
            confirm = int(self.parameters["confirm"])
            candidates = np.argwhere(
                (p > self.parameters["candidate"])
                & observed[level]
                & ~observed[confirm]
                & ~resolved
            )
            if len(candidates):
                cell = max(candidates, key=lambda xy: p[tuple(xy)])
                self.goal = self.nav.nearest(confirm, (cell + 0.5) * self.nav.table.cell_m)
        # Persist an in-flight goal so changing apparent distances cannot cause oscillation.
        if self.goal is not None and self.goal != node:
            return int(self.nav.first_actions[node, self.goal])
        if self.goal is None:
            remaining = [
                i
                for i, n in enumerate(self.nav.nodes)
                if n.level == level and i not in self.visited
            ]
            if remaining:
                if self.name == "B1":
                    if self.tour is None:
                        lvl = self.nav.table.levels[level]
                        self.tour = [
                            self.nav.nearest(level, np.array(pos))
                            for pos in self.nav.table.lawnmower(lvl)
                        ]
                        if (
                            self.nav.distances[node, self.tour[-1]]
                            < self.nav.distances[node, self.tour[0]]
                        ):
                            self.tour.reverse()
                    self.goal = next(i for i in self.tour if i not in self.visited)
                else:
                    ranked = sorted(remaining, key=lambda i: -self._sum(prior, i))
                    width = max(
                        1, int(np.ceil(len(ranked) * self.parameters.get("top_share", 0.25)))
                    )
                    self.goal = min(ranked[:width], key=lambda i: self.nav.distances[node, i])
        if self.goal is not None and self.goal != node:
            return int(self.nav.first_actions[node, self.goal])
        if self.nav.nodes[node].level != level:
            nearest = self.nav.nearest(level, self.nav.positions[node])
            return int(self.nav.first_actions[node, nearest])
        # A completed fixed-level sweep stays at that level; no hidden extra policy.
        horizontal = valid[valid < 4]
        return int(horizontal[0] if len(horizontal) else valid[0])

    def _contains(self, node: int, cell: np.ndarray) -> bool:
        x0, y0, x1, y1 = self.nav.bounds(node)
        return bool(x0 <= cell[0] < x1 and y0 <= cell[1] < y1)

    def _sum(self, values: np.ndarray, node: int) -> float:
        x0, y0, x1, y1 = self.nav.bounds(node)
        return float(values[x0:x1, y0:y1].sum())

    def _information_action(
        self,
        node: int,
        p: np.ndarray,
        observed: np.ndarray,
        resolved: np.ndarray,
        valid: np.ndarray,
    ) -> int:
        gains, finds = [], []
        for li in range(len(self.nav.table.levels)):
            gain, found = information(p, self.sensor, li, self.variant)
            mask = ~observed[li] & ~resolved
            gains.append(gain * mask)
            finds.append(found * mask)
        choices = []
        for action in valid:
            target = int(self.nav.next[node, action])
            li = self.nav.nodes[target].level
            gain, found = self._sum(gains[li], target), self._sum(finds[li], target)
            cost = self.nav.cost[node, action]
            score = (gain / cost, found / cost, -cost, -int(action))
            if self.name == "B5":
                for second, following in enumerate(self.nav.next[target]):
                    if following < 0:
                        continue
                    lj = self.nav.nodes[following].level
                    extra = self._sum(gains[lj], int(following))
                    a0, b0, a1, b1 = self.nav.bounds(target)
                    c0, d0, c1, d1 = self.nav.bounds(int(following))
                    x0, y0, x1, y1 = max(a0, c0), max(b0, d0), min(a1, c1), min(b1, d1)
                    if x0 < x1 and y0 < y1:
                        sl = np.s_[x0:x1, y0:y1]
                        both = ~observed[li][sl] & ~observed[lj][sl] & ~resolved[sl]
                        if li == lj:
                            extra -= float(gains[lj][sl][both].sum())
                        elif both.any():
                            grid, conditional = self._conditional_table(li, lj)
                            pp = np.clip(p[sl][both], 1e-12, 1 - 1e-12)
                            corrected = np.interp(np.log(pp) - np.log1p(-pp), grid, conditional)
                            extra += float((corrected - gains[lj][sl][both]).sum())
                    total_cost = cost + self.nav.cost[target, second]
                    candidate = (
                        (gain + max(0, extra)) / total_cost,
                        found / total_cost,
                        -total_cost,
                        -int(action),
                    )
                    score = max(score, candidate)
            choices.append((score, int(action)))
        return max(choices)[1]

    def _conditional_table(self, first: int, second: int) -> tuple[np.ndarray, np.ndarray]:
        key = (first, second)
        if key not in self._conditional:
            self._conditional[key] = conditional_table(
                self.sensor.model_dump_json(), self.variant, first, second
            )
        return self._conditional[key]


@lru_cache(maxsize=100)
def conditional_table(
    model_json: str, variant: str, first: int, second: int
) -> tuple[np.ndarray, np.ndarray]:
    """E[MI at second level | first outcome], tabulated on 2,049 log-odds values.

    The outcome sum is exact under the sensor's conditional-independence assumption;
    interpolation avoids repeating the histogram sum for every depth-two action.
    """
    model = SensorModel.model_validate_json(model_json)
    grid = np.linspace(-model.fit_config.log_odds_limit, model.fit_config.log_odds_limit, 2049)
    p = np.exp(-np.logaddexp(0, -grid))
    if variant == "V0":
        r, f = model.fit_config.naive_recall, 0.0
        a, b = np.array([1 - r, r]), np.array([1 - f, f])
    else:
        lm = model.levels[first]
        rates = (
            model.global_binary if variant == "V1" else lm.binary if variant == "V2" else lm.soft
        )
        r, f = rates.recall, rates.false_alarm
        a = (
            np.r_[1 - r, r * np.asarray(lm.occupied_hist)]
            if variant == "V3"
            else np.array([1 - r, r])
        )
        b = np.r_[1 - f, f * np.asarray(lm.empty_hist)] if variant == "V3" else np.array([1 - f, f])
    expected = np.zeros_like(p)
    for occupied, empty in zip(a, b, strict=True):
        probability = p * occupied + (1 - p) * empty
        posterior = np.divide(
            p * occupied, probability, out=np.zeros_like(p), where=probability > 0
        )
        expected += probability * information(posterior, model, second, variant)[0]
    return grid, expected
