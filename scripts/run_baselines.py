"""Resumable M7 baseline selection on rl-val only; never reads test worlds."""

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Any

from altiseek.cache.build import header_for
from altiseek.cache.config import CacheConfig
from altiseek.cache.jobs import load_worlds, resume_metadata, write_json
from altiseek.cache.storage import CachePool
from altiseek.calibration.model import SensorModel, digest
from altiseek.config import load_config
from altiseek.env.config import SearchConfig
from altiseek.env.logging import EpisodeLog
from altiseek.env.navigation import Navigation
from altiseek.env.search import SearchEnv
from altiseek.policies.baselines import Baseline
from altiseek.policies.config import BaselineRun
from altiseek.registry.run import sha256_file, write_metadata


def candidates(cfg: BaselineRun) -> list[tuple[str, dict[str, Any]]]:
    return (
        [("random", {}), ("oracle", {})]
        + [
            (name, {"level": li, "top_share": cfg.top_share})
            for name in ("B1", "B2")
            for li in cfg.levels
        ]
        + [
            ("B3", {"coarse": c, "confirm": f, "candidate": t, "top_share": cfg.top_share})
            for c, f, t in itertools.product(
                cfg.coarse_levels, cfg.confirm_levels, cfg.candidate_thresholds
            )
        ]
        + [("B4", {}), ("B5", {})]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/policy/baselines.yaml"))
    parser.add_argument("--search-config", type=Path, default=Path("configs/env/search.yaml"))
    parser.add_argument(
        "--model", type=Path, default=Path("data/sensor_models/fcf8cdae58ca3947.json")
    )
    parser.add_argument("--out", type=Path, default=Path("data/baselines"))
    parser.add_argument(
        "--smoke", action="store_true", help="one world/variant and one setting per policy"
    )
    parser.add_argument("--methods", nargs="+", help="subset to process; same full-run identity")
    args = parser.parse_args()
    run = load_config(args.config, BaselineRun)
    search = load_config(args.search_config, SearchConfig)
    cache = load_config("configs/cache/default.yaml", CacheConfig)
    sensor = SensorModel.load(
        args.model,
        detector_id=cache.detector_id,
        fit_world_set_id=cache.world_set_ids["sensor-fit"],
    )
    worlds = load_worlds(cache, "rl-val")
    inputs: dict[str, Path] = {
        "model": args.model,
        "settings": args.config,
        "search": args.search_config,
    }
    for world in worlds:
        path = cache.out_dir / "rl-val" / f"{world.world_id}.npz"
        if resume_metadata(path, world, cache) is None:
            raise ValueError("unverified cache")
        inputs[f"cache_{world.world_id}"] = path
    identity = {
        "schema": 1,
        "implementation": "m7-v1",
        "config": run.model_dump(mode="json"),
        "search": search.model_dump(mode="json"),
        "model_sha256": sha256_file(args.model),
        "world_set_id": cache.world_set_ids["rl-val"],
        "smoke": args.smoke,
        "cache_ids": [header_for(w, cache).cache_id for w in worlds],
    }
    root = args.out / digest(identity)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "plan.json", identity)
    nav = Navigation(cache.render.levels, cache.render.camera.hfov_deg, search.energy)
    pool = CachePool(capacity=40)
    jobs = candidates(run)
    if args.smoke:
        worlds = worlds[:1]
        jobs = [
            next(j for j in jobs if j[0] == name)
            for name in ("random", "oracle", "B1", "B2", "B3", "B4", "B5")
        ]
    variants = ("V3",) if args.smoke else run.variants
    print(f"Run {root.name}: {len(worlds) * len(variants) * len(jobs)} episodes", flush=True)
    completed = []
    for world in worlds:
        for variant in variants:
            for name, params in jobs:
                if args.methods and name not in args.methods:
                    continue
                seed = run.seed + world.seed
                key = {
                    "world_id": world.world_id,
                    "variant": variant,
                    "method": name,
                    "parameters": params,
                    "seed": seed,
                }
                path = root / f"{digest(key)}.json"
                if path.exists():
                    saved = json.loads(path.read_text(encoding="utf-8"))
                    if saved["run_id"] != root.name or saved["key"] != key:
                        raise ValueError("episode resume mismatch")
                    completed.append(path)
                    continue
                env = SearchEnv(
                    [world],
                    cache,
                    sensor,
                    search.model_copy(update={"sensor_variant": variant}),
                    navigation=nav,
                    pool=pool,
                    oracle=name == "oracle",
                )
                logger = EpisodeLog(env)
                start = time.perf_counter()
                obs, info = logger.reset(seed=seed)
                policy = Baseline(
                    name,
                    nav,
                    sensor,
                    variant,
                    seed=seed + 1,
                    parameters=params,
                    oracle_cells=env._target_cells.copy() if name == "oracle" else None,
                )
                environment_s, policy_s = 0.0, 0.0
                while not info["ended"]:
                    tick = time.perf_counter()
                    action = policy.act(obs, info["action_mask"])
                    policy_s += time.perf_counter() - tick
                    if not info["action_mask"][action]:
                        raise ValueError("baseline selected a masked action")
                    tick = time.perf_counter()
                    obs, _, _, _, info = logger.step(action)
                    environment_s += time.perf_counter() - tick
                result = {
                    "run_id": root.name,
                    "key": key,
                    "family": world.family,
                    "metrics": logger.summary(),
                    "records": logger.records,
                    "timing": {
                        "wall_s": time.perf_counter() - start,
                        "environment_s": environment_s,
                        "policy_s": policy_s,
                    },
                }
                write_json(path, result)
                completed.append(path)
                m = result["metrics"]
                print(
                    f"{world.world_id} {variant} {name} {params}: "
                    f"found={m['found_fraction']:.2f} AUC={m['auc']:.3f} "
                    f"{result['timing']['wall_s']:.2f}s",
                    flush=True,
                )
    summary = root / ("partial.json" if args.methods else "complete.json")
    write_json(
        summary,
        {"run_id": root.name, "episodes": len(completed), "files": [p.name for p in completed]},
    )
    write_metadata(
        root / "run.meta.json",
        stage="run-baselines",
        config=run,
        inputs=inputs,
        outputs={"plan": root / "plan.json", "completion": summary},
        extra={
            "run_id": root.name,
            "world_set": "rl-val",
            "sensor_model_id": sensor.sensor_model_id,
        },
    )
    print(f"Completed: {root}", flush=True)


if __name__ == "__main__":
    main()
