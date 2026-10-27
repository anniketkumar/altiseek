"""Episode records and hidden-truth diagnostics, kept outside policy observations."""

from typing import Any

import gymnasium as gym
import numpy as np

from altiseek.env.search import SearchEnv


class EpisodeLog(gym.Wrapper):
    def __init__(self, env: SearchEnv) -> None:
        super().__init__(env)
        self.search = env
        self.records: list[dict[str, Any]] = []

    def reset(self, **kwargs: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self.records = []
        self.false_clear_crossings = 0
        self.coarse_target_negative_updates = 0
        self.episode_reward = self.search.initial_reward
        self._record(None, self.search.initial_reward, info)
        return obs, info

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        env = self.search
        cells = env._target_cells
        before = np.exp(-np.logaddexp(0, -env.belief.log_odds[cells[:, 0], cells[:, 1]]))
        was_found = env.found.copy()
        old_observations = env.observations
        obs, reward, terminated, truncated, info = self.env.step(action)
        after = np.exp(-np.logaddexp(0, -env.belief.log_odds[cells[:, 0], cells[:, 1]]))
        li = env.nav.nodes[env.node].level
        if env.observations > old_observations and li == len(env.table.levels) - 1:
            negative = (after < before - 1e-12) & ~was_found & ~env.found
            self.coarse_target_negative_updates += int(negative.sum())
            self.false_clear_crossings += int((negative & (before >= 0.01) & (after < 0.01)).sum())
        self.episode_reward += reward
        self._record(action, reward, info)
        return obs, reward, terminated, truncated, info

    def _record(self, action: int | None, reward: float, info: dict[str, Any]) -> None:
        node = self.search.nav.nodes[self.search.node]
        self.records.append(
            {
                "action": action,
                "level": node.level,
                "ix": node.ix,
                "iy": node.iy,
                "reward": reward,
                **{
                    k: info[k]
                    for k in (
                        "found",
                        "false_declarations",
                        "time_s",
                        "energy_s",
                        "path_m",
                        "events",
                    )
                },
            }
        )

    def summary(self) -> dict[str, Any]:
        env = self.search
        budget, count = env.config.energy.battery_s, len(env.found)
        previous_t, previous_fraction, auc, first = 0.0, 0.0, 0.0, budget
        for row in self.records:
            time = min(budget, row["time_s"])
            auc += (time - previous_t) * previous_fraction
            previous_t, previous_fraction = time, row["found"] / count
            if row["found"]:
                first = min(first, time)
        auc += max(0, budget - previous_t) * previous_fraction
        return {
            "found_fraction": float(env.found.mean()),
            "success": bool(env.found.all()),
            "auc": auc / budget,
            "first_find_s": first,
            "false_declarations": env.false_declarations,
            "reward": self.episode_reward,
            "energy_s": env.energy_used_s,
            "elapsed_s": env.elapsed_s,
            "path_m": env.path_m,
            "steps": env.steps,
            "time_per_level_s": env.time_per_level.tolist(),
            "false_clear_crossings": self.false_clear_crossings,
            "coarse_target_negative_updates": self.coarse_target_negative_updates,
        }
