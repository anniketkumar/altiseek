"""Replay saved validation actions exactly and measure core environment step time."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from altiseek.cache.config import CacheConfig
from altiseek.cache.jobs import load_worlds, write_json
from altiseek.cache.storage import CachePool
from altiseek.calibration.model import SensorModel, digest
from altiseek.config import load_config
from altiseek.env.config import SearchConfig
from altiseek.env.logging import EpisodeLog
from altiseek.env.navigation import Navigation
from altiseek.env.search import SearchEnv
from altiseek.registry.run import write_metadata


class TimedSearch(SearchEnv):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.step_seconds: list[float] = []

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        tick = time.perf_counter()
        result = super().step(action)
        self.step_seconds.append(time.perf_counter() - tick)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--limit", type=int, default=0, help="pilot only; zero replays all random/oracle episodes"
    )
    args = parser.parse_args()
    plan = json.loads((args.root / "plan.json").read_text(encoding="utf-8"))
    search = SearchConfig.model_validate(plan["search"])
    cache = load_config("configs/cache/default.yaml", CacheConfig)
    sensor_path = Path("data/sensor_models/fcf8cdae58ca3947.json")
    sensor = SensorModel.load(sensor_path, detector_id=cache.detector_id)
    worlds = load_worlds(cache, "rl-val")
    nav = Navigation(cache.render.levels, cache.render.camera.hfov_deg, search.energy)
    pool = CachePool(40)
    times, episodes, inputs = [], 0, {"model": sensor_path, "plan": args.root / "plan.json"}
    for world in worlds:
        for variant in plan["config"]["variants"]:
            for method in ("random", "oracle"):
                if args.limit and episodes >= args.limit:
                    break
                key = {
                    "world_id": world.world_id,
                    "variant": variant,
                    "method": method,
                    "parameters": {},
                    "seed": plan["config"]["seed"] + world.seed,
                }
                path = args.root / f"{digest(key)}.json"
                if not path.exists() and args.limit:
                    continue
                original = json.loads(path.read_text(encoding="utf-8"))
                env = TimedSearch(
                    [world],
                    cache,
                    sensor,
                    search.model_copy(update={"sensor_variant": variant}),
                    navigation=nav,
                    pool=pool,
                    oracle=method == "oracle",
                )
                logger = EpisodeLog(env)
                _, info = logger.reset(seed=key["seed"])
                for row in original["records"][1:]:
                    _, reward, _, _, info = logger.step(row["action"])
                    for field in ("found", "false_declarations", "energy_s", "time_s", "path_m"):
                        if not np.isclose(info[field], row[field], atol=1e-12, rtol=0):
                            raise ValueError(f"replay changed {field}")
                    if not np.isclose(reward, row["reward"], atol=1e-12, rtol=0):
                        raise ValueError("replay changed reward")
                if not info["ended"]:
                    raise ValueError("replay did not finish")
                for name, value in logger.summary().items():
                    if not np.allclose(value, original["metrics"][name], atol=1e-12, rtol=0):
                        raise ValueError(f"replay changed summary {name}")
                times.extend(env.step_seconds)
                episodes += 1
                inputs[path.stem] = path
    if not args.limit and episodes != len(worlds) * len(plan["config"]["variants"]) * 2:
        raise ValueError("incomplete replay")
    result = {
        "run_id": args.root.name,
        "episodes": episodes,
        "steps": len(times),
        "pilot": bool(args.limit),
        "outcomes_unchanged": True,
        "core_step_mean_ms": 1000 * float(np.mean(times)),
        "core_step_median_ms": 1000 * float(np.median(times)),
        "core_step_p95_ms": 1000 * float(np.quantile(times, 0.95)),
    }
    report = args.root / ("replay-pilot.json" if args.limit else "replay.json")
    write_json(report, result)
    write_metadata(
        report.with_suffix(".meta.json"),
        stage="benchmark-search",
        config=search,
        inputs=inputs,
        outputs={"report": report},
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
