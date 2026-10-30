"""Cache-backed Gymnasium search; policies receive no unfound target locations."""

from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from altiseek.belief.state import Belief, entropy
from altiseek.cache.build import header_for
from altiseek.cache.config import CacheConfig
from altiseek.cache.storage import CacheHeader, CachePool, WorldCache, draw_variant
from altiseek.calibration.model import SensorModel
from altiseek.calibration.outcomes import observed_bounds
from altiseek.env.config import SearchConfig
from altiseek.env.navigation import Navigation
from altiseek.world.sampler import prior_grid
from altiseek.world.spec import Prior, WorldSpec


class SearchEnv(gym.Env):
    """Six masked moves. Map layout is C/north-to-south rows/west-to-east columns.

    Oracle mode is an explicitly privileged bound: it confirms targets whose cells
    are covered at L1 without detector misses. All travel/dwell costs still apply.
    Normal modes always declare from the same sensor posterior threshold.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    def __init__(
        self,
        worlds: list[WorldSpec],
        cache: CacheConfig,
        sensor: SensorModel,
        config: SearchConfig,
        *,
        pool: CachePool | None = None,
        navigation: Navigation | None = None,
        oracle: bool = False,
        prior_override: Prior | None = None,
    ) -> None:
        if not worlds or config.start_level >= len(cache.render.levels.levels):
            raise ValueError("empty worlds or invalid start level")
        if prior_override is not None and len(worlds) != 1:
            raise ValueError("a prior override requires exactly one physical world")
        self.prior_override = prior_override
        contract = header_for(worlds[0], cache).model_dump(
            mode="json", exclude={"world_id", "cache_id"}
        )
        if (
            sensor.detector_id != cache.detector_id
            or sensor.cache_contract != contract
            or sensor.sensor_model_id != sensor.content_id()
        ):
            raise ValueError("sensor/cache contract mismatch")
        if config.energy.battery_s <= config.energy.dwell_s:
            raise ValueError("battery must permit initial observation")
        self._worlds, self.cache_config, self.sensor, self.config = worlds, cache, sensor, config
        self.pool = pool or CachePool()
        self.nav = navigation or Navigation(
            cache.render.levels, cache.render.camera.hfov_deg, config.energy
        )
        if self.nav.table != cache.render.levels or self.nav.energy != config.energy:
            raise ValueError("navigation configuration mismatch")
        self.oracle = oracle
        self.table = self.nav.table
        self.level_count, self.grid = len(self.table.levels), self.table.grid_cells
        channels = self.level_count + 5 + 2 * config.coordinate_channels
        self.map_size = 2 * self.grid - 1 if config.observation_frame == "egocentric" else self.grid
        high = np.array([1.0] * (self.level_count + 3) + [config.max_targets], dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "map": spaces.Box(0, 1, (channels, self.map_size, self.map_size), dtype=np.float32),
                "vector": spaces.Box(np.zeros_like(high), high, dtype=np.float32),
            }
        )
        self.action_space = spaces.Discrete(7 if config.declaration_mode == "learned" else 6)
        coords = (np.arange(self.grid, dtype=np.float32) + 0.5) / self.grid
        self._coordinates = np.stack(np.meshgrid(coords, coords, indexing="ij"))
        self._ended = True

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        wi = int(options.get("world_index", self.np_random.integers(len(self._worlds))))
        if not 0 <= wi < len(self._worlds):
            raise ValueError("invalid world index")
        self.world = self._worlds[wi]
        if self.world.world_id != self.world.content_id():
            raise ValueError("invalid world content ID")
        expected = header_for(self.world, self.cache_config)
        if (
            expected.model_dump(mode="json", exclude={"world_id", "cache_id"})
            != self.sensor.cache_contract
        ):
            raise ValueError("world cache contract mismatch")
        self.cache = self._load_cache(expected)
        self._targets = np.array([(t.x_m, t.y_m) for t in self.world.targets], dtype=float)
        if not 0 < len(self._targets) <= self.config.max_targets:
            raise ValueError("target count outside configured bounds")
        self._target_cells = np.floor(self._targets / self.table.cell_m).astype(int)
        self.found = np.zeros(len(self._targets), dtype=bool)
        q = prior_grid(
            self.prior_override or self.world.prior_given, self.world.area_m, self.world.cell_m
        )
        count = len(self._targets) if self.config.known_count else self.config.assumed_count
        self.belief = Belief(q, count, self.level_count, self.sensor, self.config.relook_rule)
        self._static_map = np.zeros(
            (self.observation_space["map"].shape[0], self.grid, self.grid), dtype=np.float32
        )
        self._static_map[1] = q.T[::-1]
        if self.config.coordinate_channels:
            self._static_map[-2:] = self._coordinates.transpose(0, 2, 1)[:, ::-1, :]
        self.episode_seed = int(self.np_random.integers(2**31))
        # Random edge launch point, snapped to an admissible footprint centre.
        edge, along = (
            int(self.np_random.integers(4)),
            float(self.np_random.uniform(0, self.table.area_m)),
        )
        area = self.table.area_m
        position = ((0, along), (area, along), (along, 0), (along, area))[edge]
        self.node = int(
            options.get("start_node", self.nav.nearest(self.config.start_level, np.array(position)))
        )
        if not 0 <= self.node < len(self.nav.nodes):
            raise ValueError("invalid start node")
        self.steps, self.observations, self.false_declarations = 0, 0, 0
        self.elapsed_s = self.energy_used_s = self.config.energy.dwell_s
        self.path_m = 0.0
        self.time_per_level = np.zeros(self.level_count)
        self.time_per_level[self.nav.nodes[self.node].level] = self.elapsed_s
        self._ended = False
        self.last_events = self._observe()
        self.initial_reward = self._reward(self.last_events, self.elapsed_s)
        self._pending_termination = bool(self.found.all())
        self._ended = False
        return self._observation(), self._info()

    def _load_cache(self, expected: CacheHeader) -> WorldCache | None:
        """Live integration overrides only the observation source, not search state."""
        path = self.cache_config.out_dir / self.world.world_set / f"{self.world.world_id}.npz"
        return self.pool.get(Path(path), expected=expected)

    def action_masks(self) -> np.ndarray:
        movement = (self.nav.next[self.node] >= 0).copy()
        if self.config.declaration_mode == "learned":
            x0, y0, x1, y1 = self.footprint
            return np.r_[movement, (~self.belief.resolved[x0:x1, y0:y1]).any()]
        return movement

    def _reward(self, events: dict[str, Any], energy: float) -> float:
        return (
            events["found"]
            - self.config.false_declaration_cost * events["false"]
            - energy / self.config.energy.battery_s
        )

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._ended:
            raise RuntimeError("reset required after episode end")
        if not self.action_space.contains(action):
            raise ValueError("action outside action space")
        if self._pending_termination:
            # reset() has no terminal flag. Expose initial completion without a free view.
            self._pending_termination = False
            self._ended = True
            self.last_events = {"found": 0, "false": 0, "informative_cells": 0}
            return self._observation(), 0.0, True, False, self._info()
        self.steps += 1
        self.last_events = {"found": 0, "false": 0, "informative_cells": 0}
        if not self.action_masks()[action]:
            # Mask-ignoring callers get a zero-reward no-op, bounded by the step cap.
            truncated = self.steps >= self.config.max_steps
            self._ended = truncated
            reward = 0.0
            if self.config.shaping_weight:
                potential = -float(entropy(self.belief.probabilities).sum())
                following = 0.0 if truncated else potential
                reward = self.config.shaping_weight * (self.config.discount * following - potential)
            return self._observation(), reward, False, truncated, self._info()
        remaining = self.config.energy.battery_s - self.energy_used_s
        declaring = action == 6
        cost = self.config.energy.dwell_s if declaring else float(self.nav.cost[self.node, action])
        previous_potential = (
            -float(entropy(self.belief.probabilities).sum()) if self.config.shaping_weight else 0.0
        )
        if cost > remaining + 1e-9:
            # Insufficient energy for the complete move + dwell: end without observing.
            self.energy_used_s += remaining
            self.elapsed_s += remaining
            self.time_per_level[self.nav.nodes[self.node].level] += remaining
            self._ended = True
            reward = -remaining / self.config.energy.battery_s
            if self.config.shaping_weight:
                reward -= self.config.shaping_weight * previous_potential
            return self._observation(), reward, False, True, self._info()
        if declaring:
            self.energy_used_s += cost
            self.elapsed_s += cost
            self.time_per_level[self.nav.nodes[self.node].level] += cost
            x0, y0, x1, y1 = self.footprint
            probabilities = self.belief.probabilities[x0:x1, y0:y1].copy()
            probabilities[self.belief.resolved[x0:x1, y0:y1]] = -1
            cx, cy = np.unravel_index(np.argmax(probabilities), probabilities.shape)
            found, false = self._resolve(np.array([[cx + x0, cy + y0]]))
            self.last_events = {"found": found, "false": false, "informative_cells": 0}
            terminated = bool(self.found.all())
            truncated = (
                self.energy_used_s >= self.config.energy.battery_s
                or self.steps >= self.config.max_steps
            )
            self._ended = terminated or truncated
            reward = self._reward(self.last_events, cost)
            if self.config.shaping_weight:
                following = 0.0 if self._ended else -float(entropy(self.belief.probabilities).sum())
                reward += self.config.shaping_weight * (
                    self.config.discount * following - previous_potential
                )
            return self._observation(), reward, terminated, truncated, self._info()
        time = float(self.nav.time[self.node, action])
        self.path_m += float(self.nav.distance_m[self.node, action])
        self.node = int(self.nav.next[self.node, action])
        self.elapsed_s += time
        self.energy_used_s += cost
        self.time_per_level[self.nav.nodes[self.node].level] += time
        self.last_events = self._observe()
        terminated = bool(self.found.all())
        truncated = bool(
            self.energy_used_s >= self.config.energy.battery_s - 1e-9
            or self.steps >= self.config.max_steps
        )
        self._ended = terminated or truncated
        reward = self._reward(self.last_events, cost)
        if self.config.shaping_weight:
            potential = 0.0 if self._ended else -float(entropy(self.belief.probabilities).sum())
            reward += self.config.shaping_weight * (
                self.config.discount * potential - previous_potential
            )
        return self._observation(), reward, terminated, truncated, self._info()

    def _observe(self) -> dict[str, Any]:
        node = self.nav.nodes[self.node]
        variant = draw_variant(
            self.episode_seed, self.observations, self.cache.header.render.variants.count
        )
        self.observations += 1
        key = np.array([node.level, node.ix, node.iy, variant])
        self.footprint = observed_bounds(self.world, self.cache, key)
        detections, _ = self.cache.lookup(*map(int, key))
        return self._apply_evidence(detections, self.footprint, node.level, variant)

    def _apply_evidence(
        self,
        detections: np.ndarray,
        bounds: tuple[int, int, int, int],
        level: int,
        variant: int,
        visible_cells: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Shared cached/live evidence, declarations and verification semantics."""
        self.footprint = bounds
        informative = self.belief.update(
            detections,
            self.footprint,
            level,
            variant,
            self.config.sensor_variant,
            self._targets[self.found],
            self.config.suppression_radius_m,
            visible_cells=visible_cells,
        )
        if self.oracle:
            candidates = np.empty((0, 2), dtype=int)
            if level == 0:
                x0, y0, x1, y1 = self.footprint
                cells = self._target_cells[~self.found]
                candidates = np.unique(
                    cells[
                        (cells[:, 0] >= x0)
                        & (cells[:, 0] < x1)
                        & (cells[:, 1] >= y0)
                        & (cells[:, 1] < y1)
                    ],
                    axis=0,
                )
        elif self.config.declaration_mode == "auto":
            threshold = np.log(self.config.declare_threshold) - np.log1p(
                -self.config.declare_threshold
            )
            candidates = np.argwhere((self.belief.log_odds > threshold) & ~self.belief.resolved)
        else:
            candidates = np.empty((0, 2), dtype=int)
        new_found, false = self._resolve(candidates)
        return {
            "found": new_found,
            "false": false,
            "informative_cells": informative,
            "variant": variant,
        }

    def _resolve(self, candidates: np.ndarray) -> tuple[int, int]:
        """Resolve specified cells against truth; only verified outcomes reach observations."""
        new_found, false = 0, 0
        for cell in candidates:
            matches = (self._target_cells == cell).all(axis=1) & ~self.found
            count = int(matches.sum())
            self.found[matches] = True
            new_found += count
            false += count == 0
        self.belief.resolve(candidates)
        self.false_declarations += false
        return new_found, false

    def _observation(self) -> dict[str, np.ndarray]:
        tensor = self._static_map.copy()
        tensor[0] = self.belief.probabilities.T[::-1]
        tensor[2 : self.level_count + 2] = self.belief.observed.transpose(0, 2, 1)[:, ::-1, :]
        if self.config.relook_rule == "distinct_variant":
            count = self.cache.header.render.variants.count
            coverage = sum((self.belief.used_variants >> i) & 1 for i in range(count)) / count
            tensor[2 : self.level_count + 2] = coverage.transpose(0, 2, 1)[:, ::-1, :]
        tensor[self.level_count + 2] = self.belief.resolved.T[::-1]
        x0, y0, x1, y1 = self.footprint
        tensor[self.level_count + 3, self.grid - y1 : self.grid - y0, x0:x1] = 1
        ix, iy = np.minimum(
            (self.nav.positions[self.node] / self.table.cell_m).astype(int), self.grid - 1
        )
        tensor[self.level_count + 4, self.grid - iy - 1, ix] = 1
        one_hot = np.eye(self.level_count, dtype=np.float32)[self.nav.nodes[self.node].level]
        found_fraction = (
            self.found.sum() / len(self.found)
            if self.config.known_count
            else self.found.sum() / self.config.max_targets
        )
        vector = np.r_[
            one_hot,
            max(0, 1 - self.energy_used_s / self.config.energy.battery_s),
            min(1, self.elapsed_s / self.config.energy.battery_s),
            found_fraction,
            len(self.found) if self.config.known_count else 0,
        ].astype(np.float32)
        if self.config.observation_frame == "egocentric":
            # Retain the full world without wraparound or clipping. North remains up.
            centred = np.zeros(self.observation_space["map"].shape, dtype=np.float32)
            row, column = self.grid - iy - 1, ix
            r0, c0 = self.grid - 1 - row, self.grid - 1 - column
            centred[:, r0 : r0 + self.grid, c0 : c0 + self.grid] = tensor
            tensor = centred
        return {"map": tensor, "vector": vector}

    def _info(self) -> dict[str, Any]:
        return {
            "action_mask": self.action_masks(),
            "found": int(self.found.sum()),
            "false_declarations": self.false_declarations,
            "time_s": self.elapsed_s,
            "energy_s": self.energy_used_s,
            "path_m": self.path_m,
            "steps": self.steps,
            "terminated": bool(self.found.all()),
            "ended": self._ended or self._pending_termination,
            "events": self.last_events.copy(),
        }
