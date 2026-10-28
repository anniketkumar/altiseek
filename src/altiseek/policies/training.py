"""Verified training inputs, masked evaluation and complete rollout-boundary checkpoints."""

import json
import random
from itertools import product
from pathlib import Path
from typing import Any

import cloudpickle
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from altiseek.cache.build import header_for
from altiseek.cache.config import CacheConfig
from altiseek.cache.jobs import load_worlds, resume_metadata, write_json
from altiseek.cache.storage import CachePool
from altiseek.calibration.model import SensorModel
from altiseek.config import load_config
from altiseek.env.config import SearchConfig
from altiseek.env.logging import EpisodeLog
from altiseek.env.navigation import Navigation
from altiseek.env.search import SearchEnv
from altiseek.policies.ppo import FiniteSearch, PPOConfig, SearchFeatures
from altiseek.registry.run import sha256_file
from altiseek.world.spec import WorldSpec

SENSOR_PATH = Path("data/sensor_models/fcf8cdae58ca3947.json")
BASELINE_PATH = Path("docs/baseline-results.json")


def inputs() -> tuple[CacheConfig, SensorModel, dict[str, list[WorldSpec]], dict[str, Path]]:
    cache = load_config("configs/cache/default.yaml", CacheConfig)
    sensor = SensorModel.load(
        SENSOR_PATH,
        detector_id=cache.detector_id,
        fit_world_set_id=cache.world_set_ids["sensor-fit"],
    )
    worlds = {split: load_worlds(cache, split) for split in ("rl-train", "rl-val")}
    paths = {
        "sensor": SENSOR_PATH,
        "baselines": BASELINE_PATH,
        "selected": Path("configs/policy/selected.yaml"),
    }
    for split, specs in worlds.items():
        for world in specs:
            path = cache.out_dir / split / f"{world.world_id}.npz"
            if resume_metadata(path, world, cache) is None:
                raise ValueError("unverified training/validation cache")
            paths[world.world_id] = path
    return cache, sensor, worlds, paths


def vector_env(
    worlds: list[WorldSpec],
    cache: CacheConfig,
    sensor: SensorModel,
    search: SearchConfig,
    cfg: PPOConfig,
    seed: int,
) -> VecEnv:
    if any(w.world_set != "rl-train" for w in worlds):
        raise ValueError("PPO training must use rl-train only")

    def create() -> FiniteSearch:
        torch.set_num_threads(cfg.threads)
        return FiniteSearch(
            SearchEnv(worlds, cache, sensor, search, pool=CachePool(40)), objective=cfg.objective
        )

    cls = DummyVecEnv if cfg.vectorization == "dummy" else SubprocVecEnv
    env = cls([create for _ in range(cfg.n_envs)])
    env.seed(seed)
    return env


def make_model(env: VecEnv, cfg: PPOConfig, seed: int) -> MaskablePPO:
    torch.set_num_threads(cfg.threads)
    return MaskablePPO(
        "MultiInputPolicy",
        env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.entropy_coef,
        target_kl=cfg.target_kl,
        seed=seed,
        device="cpu",
        policy_kwargs={
            "features_extractor_class": SearchFeatures,
            "features_extractor_kwargs": {"zero_belief": cfg.zero_belief},
            "normalize_images": False,
            "net_arch": {"pi": [64], "vf": [64]},
        },
    )


def save_checkpoint(model: MaskablePPO, root: Path) -> dict[str, str]:
    """Commit marker written last; checkpoints are trusted local files, never downloads."""
    root.mkdir(parents=True, exist_ok=True)
    model.save(root / "model.zip")
    env = model.get_env()
    assert env is not None
    state = {
        "envs": env.env_method("snapshot"),
        "last_obs": model._last_obs,
        "last_episode_starts": model._last_episode_starts,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),  # noqa: NPY002 - SB3 shuffles with global NumPy RNG
        "python_rng": random.getstate(),
    }
    (root / "state.pkl").write_bytes(cloudpickle.dumps(state))
    hashes = {name: sha256_file(root / name) for name in ("model.zip", "state.pkl")}
    write_json(root / "complete.json", {"steps": model.num_timesteps, "hashes": hashes})
    return hashes


def load_checkpoint(root: Path, env: VecEnv) -> MaskablePPO:
    completion = json.loads((root / "complete.json").read_text(encoding="utf-8"))
    for name in ("model.zip", "state.pkl"):
        if sha256_file(root / name) != completion["hashes"][name]:
            raise ValueError("checkpoint checksum mismatch")
    model = MaskablePPO.load(root / "model.zip", env=env, device="cpu", force_reset=False)
    state = cloudpickle.loads((root / "state.pkl").read_bytes())
    if len(state["envs"]) != env.num_envs or model.num_timesteps != completion["steps"]:
        raise ValueError("checkpoint state mismatch")
    for index, item in enumerate(state["envs"]):
        env.env_method("restore", item, indices=index)
    model._last_obs = state["last_obs"]
    model._last_episode_starts = state["last_episode_starts"]
    # Restored environments already have seeded state; prevent VecEnv's next reset
    # from applying construction seeds after an automatic episode end.
    env._reset_seeds()
    torch.set_rng_state(state["torch_rng"])
    np.random.set_state(state["numpy_rng"])  # noqa: NPY002 - restore SB3 minibatch RNG
    random.setstate(state["python_rng"])
    return model


def evaluate(
    model: MaskablePPO,
    worlds: list[WorldSpec],
    cache: CacheConfig,
    sensor: SensorModel,
    search: SearchConfig,
    *,
    zero_belief: bool = False,
    action_rule: str = "deterministic",
    repeats: int = 1,
) -> dict[str, Any]:
    """Masked B validation with independent seeded action draws; never changes training RNGs."""
    if any(w.world_set != "rl-val" for w in worlds):
        raise ValueError("M8 evaluation must use rl-val only")
    if action_rule not in ("deterministic", "sampled") or repeats < 1:
        raise ValueError("invalid evaluation action rule/repeat count")
    if action_rule == "deterministic" and repeats != 1:
        raise ValueError("deterministic evaluation needs one repeat")
    nav = Navigation(cache.render.levels, cache.render.camera.hfov_deg, search.energy)
    pool = CachePool(40)
    rows = []
    extractor = model.policy.features_extractor
    model.policy.set_training_mode(False)
    old_zero = extractor.zero_belief
    extractor.zero_belief = zero_belief or old_zero
    try:
        for world, repeat in product(worlds, range(repeats)):
            core = SearchEnv([world], cache, sensor, search, pool=pool, navigation=nav)
            env = EpisodeLog(core)
            obs, info = env.reset(seed=20261027 + world.seed)
            rng = np.random.default_rng(20261028 + repeat * 1000000 + world.seed)
            opportunities, descents = 0, 0
            visits, informative = [], []
            while not info["ended"]:
                if action_rule == "deterministic":
                    action, _ = model.predict(
                        obs, deterministic=True, action_masks=info["action_mask"]
                    )
                    action = int(action)
                else:
                    tensors, _ = model.policy.obs_to_tensor(obs)
                    with torch.no_grad():
                        p = (
                            model.policy.get_distribution(tensors, action_masks=info["action_mask"])
                            .distribution.probs.cpu()
                            .numpy()[0]
                            .astype(float)
                        )
                    action = int(rng.choice(len(p), p=p / p.sum()))
                if not info["action_mask"][action]:
                    raise ValueError("evaluation selected a masked action")
                node = core.nav.nodes[core.node]
                if node.level >= 3 and "variant" in info["events"]:
                    detections, _ = core.cache.lookup(
                        node.level, node.ix, node.iy, info["events"]["variant"]
                    )
                    detected = bool(
                        np.any(
                            (detections["class_id"] == 0)
                            & (detections["score"] >= sensor.fit_config.tau)
                        )
                    )
                    opportunities += detected
                    descents += detected and action == 5
                obs, _, _, _, info = env.step(action)
                visits.append(core.node)
                informative.append(info["events"]["informative_cells"])
            rows.append(
                {
                    "world_id": world.world_id,
                    "repeat": repeat,
                    "family": world.family,
                    "cache_id": header_for(world, cache).cache_id,
                    "metrics": env.summary()
                    | {
                        "repeat_fraction": 1 - len(set(visits)) / len(visits) if visits else 0,
                        "zero_information_fraction": float(np.mean(np.array(informative) == 0))
                        if informative
                        else 0,
                    },
                    "coarse_cached_detection_opportunities": opportunities,
                    "immediate_descents_after_coarse_cached_detection": descents,
                    "records": env.records,
                }
            )
    finally:
        extractor.zero_belief = old_zero
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    ref = {
        (r["family"], r["method"]): r["auc"]
        for r in baseline["family_results"]
        if r["variant"] == search.sensor_variant
    }
    families = sorted({w.family for w in worlds})
    scores = []
    for family in families:
        den = ref[family, "oracle"] - ref[family, "random"]
        if den <= 0:
            raise ValueError("undefined family normalization")
        mean = np.mean([r["metrics"]["auc"] for r in rows if r["family"] == family])
        scores.append(float((mean - ref[family, "random"]) / den))
    return {
        "steps": model.num_timesteps,
        "action_rule": action_rule,
        "repeats": repeats,
        "zero_belief": zero_belief or old_zero,
        "mean_normalised_auc": float(np.mean(scores)),
        "mean_auc": float(np.mean([r["metrics"]["auc"] for r in rows])),
        "mean_found": float(np.mean([r["metrics"]["found_fraction"] for r in rows])),
        "episodes": rows,
    }
