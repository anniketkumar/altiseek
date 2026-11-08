"""B-only public-observation demonstrations and CPU actor/critic pretraining (D66)."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import Field
from sb3_contrib import MaskablePPO

from altiseek.calibration.model import digest
from altiseek.config import Config
from altiseek.env.logging import EpisodeLog
from altiseek.env.search import SearchEnv
from altiseek.policies.baselines import Baseline
from altiseek.policies.ppo import FiniteSearch
from altiseek.registry.run import sha256_file
from altiseek.world.spec import WorldSpec

ARRAY_FIELDS = frozenset({"map", "vector", "mask", "action", "reward", "return"})


class ImitationConfig(Config):
    """New schema used only by D66; existing PPO config fields are unchanged."""

    seed: int = 17
    reset_seed_base: int = 20261108
    repeats: int = Field(default=4, ge=1)
    holdout_per_family: int = Field(default=2, ge=1)
    batch_size: int = Field(default=32, ge=1)
    max_epochs: int = Field(default=40, ge=1)
    patience: int = Field(default=5, ge=1)
    min_improvement: float = Field(default=0.0001, ge=0)
    learning_rate: float = Field(default=0.0003, gt=0)
    value_weight: float = Field(default=0.5, gt=0)
    threads: int = Field(default=2, ge=1)


def split_worlds(worlds: list[WorldSpec], holdout_per_family: int = 2) -> dict[str, list[str]]:
    if not worlds or any(w.split != "B" or w.world_set != "rl-train" for w in worlds):
        raise ValueError("Imitation requires rl-train B worlds only")
    if len({w.world_id for w in worlds}) != len(worlds):
        raise ValueError("Duplicate imitation world")
    held = []
    for family in sorted({w.family for w in worlds}):
        group = sorted(
            (w for w in worlds if w.family == family),
            key=lambda w: digest(["m12-bc-holdout-v1", w.world_id]),
        )
        if len(group) <= holdout_per_family:
            raise ValueError("Holdout leaves no training worlds")
        held.extend(w.world_id for w in group[:holdout_per_family])
    return {
        "train": sorted(w.world_id for w in worlds if w.world_id not in held),
        "holdout": sorted(held),
    }


def returns_to_go(rewards: np.ndarray) -> np.ndarray:
    if rewards.ndim != 1 or not np.isfinite(rewards).all():
        raise ValueError("Invalid demonstration rewards")
    return np.cumsum(rewards[::-1], dtype=np.float64)[::-1].astype(np.float32)


def collect_episode(core: SearchEnv, expert: Baseline, seed: int) -> tuple[dict, dict]:
    if any(w.split != "B" or w.world_set != "rl-train" for w in core._worlds):
        raise ValueError("Demonstrations require rl-train B only")
    if core.config.sensor_variant != "V3" or not core.config.known_count:
        raise ValueError("D66 demonstrations require V3 and known K")
    logger = EpisodeLog(core)
    env = FiniteSearch(logger, "auc")
    observation, info = env.reset(seed=seed)
    initial = (
        info["found"] / len(core.found) * max(0, 1 - info["time_s"] / core.config.energy.battery_s)
    )
    maps, vectors, masks, actions, rewards = [], [], [], [], []
    while not info["ended"]:
        if set(observation) != {"map", "vector"}:
            raise ValueError("Unexpected observation fields")
        mask = core.action_masks().copy()
        action = int(expert.act(observation, mask))
        if not mask[action]:
            raise ValueError("Expert chose a masked action")
        maps.append(observation["map"].copy())
        vectors.append(observation["vector"].copy())
        masks.append(mask)
        actions.append(action)
        observation, reward, _, _, info = env.step(action)
        rewards.append(reward)
    count = len(actions)
    reward_array = np.asarray(rewards, dtype=np.float32)
    arrays = {
        "map": np.asarray(maps, dtype=np.float32).reshape(
            (count, *core.observation_space["map"].shape)
        ),
        "vector": np.asarray(vectors, dtype=np.float32).reshape(
            (count, *core.observation_space["vector"].shape)
        ),
        "mask": np.asarray(masks, dtype=bool).reshape((count, core.action_space.n)),
        "action": np.asarray(actions, dtype=np.int64),
        "reward": reward_array,
        "return": returns_to_go(reward_array),
    }
    auc = logger.summary()["auc"]
    if not np.isclose(sum(rewards) + initial, auc, atol=1e-6, rtol=0):
        raise ValueError("Demonstration return does not reproduce D41 AUC")
    return arrays, {
        "transitions": count,
        "auc": auc,
        "initial_contribution": initial,
        "action_reward_sum": float(sum(rewards)),
    }


def load_shard(path: Path, checksum: str) -> dict[str, np.ndarray]:
    if sha256_file(path) != checksum:
        raise ValueError("Demonstration checksum mismatch")
    with np.load(path, allow_pickle=False) as stored:
        if set(stored.files) != ARRAY_FIELDS:
            raise ValueError("Prohibited or missing demonstration fields")
        arrays = {name: stored[name] for name in ARRAY_FIELDS}
    n = len(arrays["action"])
    if any(len(value) != n for value in arrays.values()):
        raise ValueError("Demonstration length mismatch")
    if arrays["mask"].dtype != np.bool_ or arrays["action"].dtype != np.int64:
        raise ValueError("Invalid demonstration mask/action types")
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Non-finite demonstration")
    if n and (
        arrays["action"].min() < 0
        or arrays["action"].max() >= arrays["mask"].shape[1]
        or not arrays["mask"][np.arange(n), arrays["action"]].all()
    ):
        raise ValueError("Masked demonstration label")
    if not np.allclose(arrays["return"], returns_to_go(arrays["reward"]), atol=1e-6, rtol=0):
        raise ValueError("Incorrect Monte-Carlo returns")
    return arrays


def batches(
    root: Path,
    episodes: list[dict],
    batch_size: int,
    rng: np.random.Generator | None,
) -> Iterator[dict[str, np.ndarray]]:
    order = np.arange(len(episodes)) if rng is None else rng.permutation(len(episodes))
    for index in order:
        episode = episodes[int(index)]
        if episode["split"] != "B" or episode["world_set"] != "rl-train":
            raise ValueError("Imitation shard outside B training scope")
        relative = Path(episode["file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid shard path")
        arrays = load_shard(root / relative, episode["sha256"])
        count = len(arrays["action"])
        indices = np.arange(count) if rng is None else rng.permutation(count)
        for start in range(0, count, batch_size):
            selected = indices[start : start + batch_size]
            yield {key: value[selected] for key, value in arrays.items()}


def imitation_loss(
    model: MaskablePPO, batch: dict, value_weight: float = 0.5
) -> tuple[torch.Tensor, dict]:
    if str(model.device) != "cpu":
        raise ValueError("Imitation must run on CPU")
    observation = {key: torch.as_tensor(batch[key], device="cpu") for key in ("map", "vector")}
    masks = torch.as_tensor(batch["mask"], dtype=torch.bool, device="cpu")
    actions = torch.as_tensor(batch["action"], dtype=torch.long, device="cpu")
    targets = torch.as_tensor(batch["return"], dtype=torch.float32, device="cpu")
    if not masks.gather(1, actions[:, None]).all():
        raise ValueError("Masked action cannot be an imitation label")
    values, log_probability, _ = model.policy.evaluate_actions(
        observation, actions, action_masks=masks
    )
    actor = -log_probability.mean()
    critic = (values.flatten() - targets).square().mean()
    loss = actor + value_weight * critic
    if not torch.isfinite(loss):
        raise ValueError("Non-finite imitation loss")
    with torch.no_grad():
        distribution = model.policy.get_distribution(observation, action_masks=masks)
        accuracy = (distribution.distribution.probs.argmax(dim=1) == actions).float().mean()
    return loss, {
        "loss": float(loss.detach()),
        "actor_ce": float(actor.detach()),
        "critic_mse": float(critic.detach()),
        "accuracy": float(accuracy),
        "transitions": len(actions),
    }


def space_contract(model: MaskablePPO) -> dict:
    return {
        "map_shape": list(model.observation_space["map"].shape),
        "vector_shape": list(model.observation_space["vector"].shape),
        "vector_low": model.observation_space["vector"].low.tolist(),
        "vector_high": model.observation_space["vector"].high.tolist(),
        "actions": int(model.action_space.n),
        "feature_class": type(model.policy.features_extractor).__name__,
        "zero_belief": model.policy.features_extractor.zero_belief,
    }


def save_weights(model: MaskablePPO, root: Path, metadata: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / "policy.tmp.pt"
    torch.save(model.policy.state_dict(), temporary)
    temporary.replace(root / "policy.pt")
    record = {**metadata, "space": space_contract(model), "sha256": sha256_file(root / "policy.pt")}
    temporary_json = root / "complete.tmp.json"
    temporary_json.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary_json.replace(root / "complete.json")


def load_weights(model: MaskablePPO, root: Path) -> dict:
    record = json.loads((root / "complete.json").read_text(encoding="utf-8"))
    if record["sha256"] != sha256_file(root / "policy.pt") or record["space"] != space_contract(
        model
    ):
        raise ValueError("Warm-start identity/observation/action contract mismatch")
    model.policy.load_state_dict(
        torch.load(root / "policy.pt", map_location="cpu", weights_only=True)
    )
    return record


def epoch_metrics(
    model: MaskablePPO,
    root: Path,
    episodes: list[dict],
    cfg: ImitationConfig,
    *,
    rng: np.random.Generator | None = None,
    train: bool = False,
) -> dict[str, Any]:
    model.policy.set_training_mode(train)
    totals = {"loss": 0.0, "actor_ce": 0.0, "critic_mse": 0.0, "accuracy": 0.0}
    count = 0
    with torch.set_grad_enabled(train):
        for batch in batches(root, episodes, cfg.batch_size, rng):
            loss, stats = imitation_loss(model, batch, cfg.value_weight)
            if train:
                model.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
                model.policy.optimizer.step()
            n = stats["transitions"]
            count += n
            for key in totals:
                totals[key] += n * stats[key]
    if count == 0:
        raise ValueError("No imitation transitions in partition")
    return {**{key: value / count for key, value in totals.items()}, "transitions": count}
