"""Controlled four-world B training probe; preserves all M8 main runs and settings."""

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv

from altiseek.cache.jobs import write_json
from altiseek.cache.storage import CachePool
from altiseek.calibration.model import digest
from altiseek.config import load_config
from altiseek.env.config import SearchConfig
from altiseek.env.logging import EpisodeLog
from altiseek.env.navigation import Navigation
from altiseek.env.search import SearchEnv
from altiseek.policies.baselines import Baseline
from altiseek.policies.ppo import FiniteSearch, PPOConfig
from altiseek.policies.training import inputs, load_checkpoint, make_model, save_checkpoint
from altiseek.registry.run import sha256_file, write_metadata


class FixedLaunch(FiniteSearch):
    def __init__(self, env: SearchEnv, launch_seed: int) -> None:
        super().__init__(env)
        self.launch_seed = launch_seed

    def reset(self, **kwargs: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        kwargs["seed"] = self.launch_seed
        return self.env.reset(**kwargs)


def main() -> None:
    torch.set_num_threads(2)
    cfg = PPOConfig.model_validate(
        load_config("configs/policy/ppo.yaml", PPOConfig).model_dump()
        | {"total_steps": 32768, "eval_every": 8192}
    )
    search = load_config("configs/env/search.yaml", SearchConfig)
    cache, sensor, worlds, paths = inputs()
    # Predeclared selection by family and content ID, never by achieved returns.
    families = ("sharp-single-low", "diffuse-single-low", "sharp-group-low", "diffuse-group-low")
    selected = [
        min((w for w in worlds["rl-train"] if w.family == family), key=lambda w: w.world_id)
        for family in families
    ]
    nav = Navigation(cache.render.levels, cache.render.camera.hfov_deg, search.energy)
    pool = CachePool(4)
    launch_seeds = []
    for world in selected:
        env = SearchEnv([world], cache, sensor, search, navigation=nav, pool=pool)
        for seed in range(20261028, 20261128):
            _, info = env.reset(seed=seed)
            if not info["ended"]:
                launch_seeds.append(seed)
                break
        else:
            raise ValueError("no nontrivial initial state in declared seed range")
    plan = {
        "version": "m8-overfit-v1",
        "config": cfg.model_dump(mode="json"),
        "worlds": [w.world_id for w in selected],
        "launch_seeds": launch_seeds,
        "source_sha256": sha256_file(Path(__file__)),
        "sensor_model_id": sensor.sensor_model_id,
    }
    root = Path("data/ppo-diagnosis") / digest(plan)
    write_json(root / "plan.json", plan)
    print(f"Overfit probe {root}", flush=True)
    if (root / "complete.json").exists():
        print("Already complete", flush=True)
        return

    def factory(index: int) -> Any:
        def create() -> FixedLaunch:
            return FixedLaunch(
                SearchEnv([selected[index]], cache, sensor, search), launch_seeds[index]
            )

        return create

    vector = SubprocVecEnv([factory(i) for i in range(4)])
    vector.seed(17)
    checkpoints = sorted(root.glob("step-*/complete.json"))
    model = (
        load_checkpoint(checkpoints[-1].parent, vector)
        if checkpoints
        else make_model(vector, cfg, 17)
    )
    logger = configure(str(root / f"logs-{model.num_timesteps:09d}"), ["csv"])
    model.set_logger(logger)

    def evaluate(policy_name: str, repeat: int = 0) -> dict[str, Any]:
        results = []
        for i, world in enumerate(selected):
            env = EpisodeLog(SearchEnv([world], cache, sensor, search, navigation=nav, pool=pool))
            obs, info = env.reset(seed=launch_seeds[i])
            baseline = Baseline("B5", nav, sensor, "V3", seed=17) if policy_name == "B5" else None
            rng = np.random.default_rng(17 + repeat)
            while not info["ended"]:
                if baseline is not None:
                    action = baseline.act(obs, info["action_mask"])
                else:
                    tensors, _ = model.policy.obs_to_tensor(obs)
                    with torch.no_grad():
                        p = (
                            model.policy.get_distribution(tensors, action_masks=info["action_mask"])
                            .distribution.probs.cpu()
                            .numpy()[0]
                        )
                    action = (
                        int(p.argmax())
                        if policy_name == "deterministic"
                        else int(rng.choice(6, p=p / p.sum()))
                    )
                obs, _, _, _, info = env.step(action)
            results.append({"world_id": world.world_id, "metrics": env.summary()})
        return {
            "method": policy_name,
            "repeat": repeat,
            "steps": model.num_timesteps,
            "auc": float(np.mean([r["metrics"]["auc"] for r in results])),
            "reward": float(np.mean([r["metrics"]["reward"] for r in results])),
            "found": float(np.mean([r["metrics"]["found_fraction"] for r in results])),
            "episodes": results,
        }

    try:
        if not (root / "baseline.json").exists():
            write_json(root / "baseline.json", evaluate("B5"))
        while True:
            result_path = root / f"eval-{model.num_timesteps:09d}.json"
            if not result_path.exists():
                results = [
                    evaluate("deterministic"),
                    *[evaluate("stochastic", r) for r in range(3)],
                ]
                write_json(result_path, results)
                print(
                    f"step={model.num_timesteps} deterministic AUC={results[0]['auc']:.3f}; "
                    f"stochastic AUC={np.mean([r['auc'] for r in results[1:]]):.3f}",
                    flush=True,
                )
            if model.num_timesteps >= cfg.total_steps:
                break
            start = time.perf_counter()
            model.learn(cfg.eval_every, reset_num_timesteps=False)
            save_checkpoint(model, root / f"step-{model.num_timesteps:09d}")
            print(f"{cfg.eval_every / (time.perf_counter() - start):.1f} steps/s", flush=True)
        write_json(root / "complete.json", {"steps": model.num_timesteps, "run_id": root.name})
        write_metadata(
            root / "run.meta.json",
            stage="check-ppo-overfit",
            config=cfg,
            inputs=paths,
            outputs={"plan": root / "plan.json", "completion": root / "complete.json"},
        )
    finally:
        vector.close()
        logger.close()


if __name__ == "__main__":
    main()
