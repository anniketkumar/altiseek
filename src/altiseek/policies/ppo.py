"""CPU MaskablePPO integration; no privileged state reaches the network."""

from copy import deepcopy
from typing import Any, Literal

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from pydantic import Field, model_validator
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from altiseek.config import Config
from altiseek.env.search import SearchEnv


class PPOConfig(Config):
    seeds: tuple[int, ...] = (17, 29, 43)
    variants: tuple[Literal["V0", "V3"], ...] = ("V0", "V3")
    total_steps: int = Field(default=131072, ge=1)
    eval_every: int = Field(default=16384, ge=1)
    n_envs: int = Field(default=4, ge=1)
    n_steps: int = Field(default=128, ge=2)
    batch_size: int = Field(default=64, ge=2)
    epochs: int = Field(default=4, ge=1)
    threads: int = Field(default=1, ge=1)
    learning_rate: float = Field(default=0.0003, gt=0)
    gamma: float = Field(default=0.99, gt=0, le=1)
    gae_lambda: float = Field(default=0.95, gt=0, le=1)
    clip_range: float = Field(default=0.2, gt=0)
    entropy_coef: float = Field(default=0.01, ge=0)
    target_kl: float = Field(default=0.03, gt=0)
    vectorization: Literal["dummy", "subproc"] = "dummy"
    zero_belief: bool = False
    objective: Literal["legacy", "auc"] = "legacy"
    evaluation_actions: Literal["deterministic", "sampled"] = "deterministic"
    evaluation_repeats: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def valid_batches(self) -> "PPOConfig":
        rollout = self.n_steps * self.n_envs
        if rollout % self.batch_size or self.eval_every % rollout:
            raise ValueError("batch and evaluation sizes must divide complete rollouts")
        if self.total_steps % self.eval_every:
            raise ValueError("total steps must be a multiple of evaluation interval")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("duplicate training seeds")
        if self.objective == "auc" and self.gamma != 1:
            raise ValueError("AUC objective requires gamma=1")
        if self.evaluation_actions == "deterministic" and self.evaluation_repeats != 1:
            raise ValueError("deterministic evaluation needs only one repeat")
        return self


class SearchFeatures(BaseFeaturesExtractor):
    """Small spatial CNN plus vector; fixed log scaling exposes sparse p/q values.

    Scaling uses only public probabilities: log(clamp(p, exp(-20), 1))/20 + 1.
    A zero-belief ablation is applied AFTER scaling so the input is exactly zero.
    No learned normalizer or validation statistics enter preprocessing.
    """

    def __init__(self, observation_space: spaces.Dict, zero_belief: bool = False) -> None:
        super().__init__(observation_space, features_dim=128)
        self.zero_belief = zero_belief
        self.max_targets = float(observation_space["vector"].high[-1])
        channels = observation_space["map"].shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 16, 5, stride=3, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(32 * 4 * 4 + observation_space["vector"].shape[0], 128), nn.ReLU()
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        maps = observations["map"].clone()
        maps[:, :2] = torch.log(maps[:, :2].clamp(min=np.exp(-20), max=1)) / 20 + 1
        if self.zero_belief:
            maps[:, 0] = 0
        vector = observations["vector"].clone()
        vector[:, -1] /= self.max_targets  # fixed space bound; no fitted normalization
        return self.head(torch.cat((self.cnn(maps), vector), dim=1))


class FiniteSearch(gym.Wrapper):
    """Treat battery/cap endings as task terminals, preventing SB3 timeout bootstrap.

    The core Gymnasium API and evaluation stay unchanged. Initial rewards remain
    in evaluation logs; they precede every policy action and are not trainable.
    """

    def __init__(self, env: SearchEnv, objective: str = "legacy") -> None:
        super().__init__(env)
        if objective not in ("legacy", "auc"):
            raise ValueError("unknown objective")
        if objective == "auc" and (
            not env.unwrapped.config.known_count or env.unwrapped.config.shaping_weight
        ):
            raise ValueError("AUC training requires known K and no shaping")
        self.objective = objective

    def action_masks(self) -> np.ndarray:
        return self.env.unwrapped.action_masks()

    def snapshot(self) -> dict[str, Any]:
        env = self.env.unwrapped
        names = (
            "belief",
            "episode_seed",
            "node",
            "steps",
            "observations",
            "false_declarations",
            "elapsed_s",
            "energy_used_s",
            "path_m",
            "time_per_level",
            "_ended",
            "last_events",
            "initial_reward",
            "_pending_termination",
            "found",
            "footprint",
        )
        return deepcopy(
            {
                "world_index": env._worlds.index(env.world),
                "objective": self.objective,
                "rng": env.np_random.bit_generator.state,
                "values": {name: getattr(env, name) for name in names},
            }
        )

    def restore(self, state: dict[str, Any]) -> None:
        if state.get("objective", "legacy") != self.objective:
            raise ValueError("checkpoint objective mismatch")
        env = self.env.unwrapped
        env.reset(seed=0, options={"world_index": state["world_index"]})
        env.np_random.bit_generator.state = deepcopy(state["rng"])
        for name, value in deepcopy(state["values"]).items():
            setattr(env, name, value)

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if not self.action_masks()[action]:
            raise ValueError("training selected a masked action")
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.objective == "auc":
            core = self.env.unwrapped
            reward = (
                info["events"]["found"]
                / len(core.found)
                * max(0.0, 1 - info["time_s"] / core.config.energy.battery_s)
            )
        return obs, reward, terminated or truncated, False, info
