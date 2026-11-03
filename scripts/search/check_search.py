"""M7 unlimited-coverage and V0 false-clear diagnostics, using rl-val only."""

import json
from pathlib import Path

import numpy as np

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
from altiseek.registry.run import write_metadata


def main() -> None:
    cache = load_config("configs/cache/default.yaml", CacheConfig)
    search = load_config("configs/env/search.yaml", SearchConfig)
    model_path = Path("data/sensor_models/fcf8cdae58ca3947.json")
    sensor = SensorModel.load(model_path, detector_id=cache.detector_id)
    worlds = load_worlds(cache, "rl-val")
    search = search.model_copy(
        update={"energy": search.energy.model_copy(update={"battery_s": 2000}), "max_steps": 600}
    )
    identity = {
        "implementation": "m7-v2-fixed-level-routing",
        "config": search.model_dump(mode="json"),
        "sensor_model_id": sensor.sensor_model_id,
        "world_set_id": cache.world_set_ids["rl-val"],
        "seed": 20261027,
        "variants": ["V0", "V1", "V2", "V3"],
        "levels": [0, 4],
    }
    root = Path("data/baselines/sanity") / digest(identity)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "plan.json", identity)
    nav = Navigation(cache.render.levels, cache.render.camera.hfov_deg, search.energy)
    pool = CachePool(40)
    outputs = {}
    for world in worlds:
        cache_path = cache.out_dir / "rl-val" / f"{world.world_id}.npz"
        if resume_metadata(cache_path, world, cache) is None:
            raise ValueError("unverified cache")
        for variant in identity["variants"]:
            for level in identity["levels"]:
                key = {"world_id": world.world_id, "variant": variant, "level": level}
                path = root / f"{digest(key)}.json"
                if path.exists():
                    saved = json.loads(path.read_text(encoding="utf-8"))
                    if saved["key"] != key or saved["run_id"] != root.name:
                        raise ValueError("sanity resume mismatch")
                else:
                    cfg = search.model_copy(
                        update={"start_level": level, "sensor_variant": variant}
                    )
                    env = SearchEnv([world], cache, sensor, cfg, navigation=nav, pool=pool)
                    logger = EpisodeLog(env)
                    obs, info = logger.reset(seed=identity["seed"] + world.seed)
                    policy = Baseline(
                        "B1", nav, sensor, variant, seed=0, parameters={"level": level}
                    )
                    while not info["ended"]:
                        action = policy.act(obs, info["action_mask"])
                        obs, _, _, _, info = logger.step(action)
                    count = cache.render.levels.views_per_world(cache.render.levels.levels[level])
                    covered = len(policy.visited) >= count or bool(env.found.all())
                    # Initial node can be absent from policy.visited if reset completes the task.
                    saved = {
                        "key": key,
                        "run_id": root.name,
                        "family": world.family,
                        "metrics": logger.summary(),
                        "complete_sweep_or_all_found": covered,
                    }
                    write_json(path, saved)
                outputs[path.stem] = path
        print(f"Checked {world.world_id}", flush=True)
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in outputs.values()]
    summary = {
        "run_id": root.name,
        "episodes": len(rows),
        "coverage_complete": all(r["complete_sweep_or_all_found"] for r in rows),
        "variants": {},
    }
    for variant in identity["variants"]:
        values = {
            level: float(
                np.mean(
                    [
                        r["metrics"]["found_fraction"]
                        for r in rows
                        if r["key"]["variant"] == variant and r["key"]["level"] == level
                    ]
                )
            )
            for level in identity["levels"]
        }
        summary["variants"][variant] = {
            "fine_found": values[0],
            "coarse_found": values[4],
            "fine_exceeds_coarse": values[0] > values[4],
        }
    v0 = [r for r in rows if r["key"]["variant"] == "V0" and r["key"]["level"] == 4]
    summary["v0_false_clear_crossings"] = sum(r["metrics"]["false_clear_crossings"] for r in v0)
    summary["v0_coarse_target_negative_updates"] = sum(
        r["metrics"]["coarse_target_negative_updates"] for r in v0
    )
    summary["passed"] = (
        summary["coverage_complete"]
        and summary["v0_false_clear_crossings"] > 0
        and all(v["fine_exceeds_coarse"] for v in summary["variants"].values())
    )
    write_json(root / "summary.json", summary)
    write_metadata(
        root / "check.meta.json",
        stage="check-search",
        config=search,
        inputs={"model": model_path},
        outputs={"summary": root / "summary.json", **outputs},
        extra={"run_id": root.name, "world_set_id": cache.world_set_ids["rl-val"]},
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
