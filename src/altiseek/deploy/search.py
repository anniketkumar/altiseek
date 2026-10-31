"""Live evidence adapter for B-only waypoint integration pilots.

Search observations, masks, verification and policy inputs remain shared. This
pilot retains the frozen ideal energy model; actual flight timing is logged by
the source. It must not be used for transfer scores until measured flight-energy
accounting and a transfer protocol are implemented and checked.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from altiseek.cache.config import CacheConfig
from altiseek.cache.storage import CacheHeader
from altiseek.calibration.model import SensorModel
from altiseek.deploy.meter import BudgetExhausted, FlightMeter
from altiseek.deploy.projection import GroundCamera
from altiseek.env.config import SearchConfig
from altiseek.env.search import SearchEnv
from altiseek.world.spec import WorldSpec


@dataclass(frozen=True)
class LiveView:
    camera: GroundCamera
    detections: np.ndarray


class LiveSearchPilot(SearchEnv):
    def __init__(
        self,
        world: WorldSpec,
        cache: CacheConfig,
        sensor: SensorModel,
        config: SearchConfig,
        source: Callable[[np.ndarray, int], LiveView],
        *,
        meter: FlightMeter | None = None,
    ) -> None:
        if world.split != "B":
            raise ValueError("Live integration pilot accepts B only")
        if config.relook_rule != "first" or config.declaration_mode != "auto":
            raise ValueError("Live pilot requires frozen first-look and automatic declaration")
        if config.observation_frame != "north_up":
            raise ValueError("Live pilot requires the frozen north-up observation")
        self.source = source
        self.meter = meter
        self.live_visible: np.ndarray | None = None
        super().__init__([world], cache, sensor, config)

    def _load_cache(self, expected: CacheHeader) -> None:
        # Parent reset still checks world/sensor contracts. Never read cached detections.
        return None

    def _observe(self) -> dict:
        node = self.nav.nodes[self.node]
        goal = np.r_[self.nav.positions[self.node], self.nav.altitudes[node.level]]
        view = self.source(goal, node.level)
        if self.meter is not None:
            self._account()
            if self.meter.snapshot()["exhausted"]:
                raise BudgetExhausted("Battery expired before evidence update")
        measured_level = int(np.argmin(abs(self.nav.altitudes - view.camera.position_enu_m[2])))
        if measured_level != node.level:
            raise ValueError(
                "Captured altitude belongs to a different level; flight did not settle"
            )
        visible = view.camera.fully_visible_cells(self.grid, self.table.cell_m)
        self.live_visible = visible.copy()
        self.observations += 1
        cells = np.argwhere(visible)
        bounds = (
            (0, 0, 0, 0)
            if not len(cells)
            else (
                int(cells[:, 0].min()),
                int(cells[:, 1].min()),
                int(cells[:, 0].max()) + 1,
                int(cells[:, 1].max()) + 1,
            )
        )
        return self._apply_evidence(view.detections, bounds, measured_level, 0, visible)

    def _account(self) -> None:
        state = self.meter.snapshot()
        self.energy_used_s, self.elapsed_s = state["energy_s"], state["elapsed_s"]
        self.path_m = state["path_m"]
        self.time_per_level = state["time_per_level"]

    def step(self, action: int) -> tuple:
        if self.meter is None:
            return super().step(action)
        self._account()
        before = self.energy_used_s
        try:
            result = super().step(action)
            # Parent's unaffordable-move branch fills the ideal budget. Live
            # reporting instead retains the energy actually consumed.
            self._account()
            if result[2] or result[3]:
                self.meter.stop()
            reward = self._reward(self.last_events, self.energy_used_s - before)
            return self._observation(), reward, result[2], result[3], self._info()
        except BudgetExhausted:
            self._account()
            self._ended = True
            self.last_events = {
                "found": 0,
                "false": 0,
                "informative_cells": 0,
                "battery_expired_in_flight": True,
            }
            reward = -(self.energy_used_s - before) / self.config.energy.battery_s
            return self._observation(), reward, False, True, self._info()

    def _observation(self) -> dict[str, np.ndarray]:
        result = super()._observation()
        if self.live_visible is not None:
            result["map"][self.level_count + 3] = self.live_visible.T[::-1]
        return result
