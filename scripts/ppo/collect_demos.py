"""D66: resumable B5 demonstrations from rl-train B, with an inventory stop."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from altiseek.cache.build import header_for
from altiseek.cache.config import CacheConfig
from altiseek.cache.jobs import load_worlds, resume_metadata, write_json
from altiseek.cache.storage import CachePool
from altiseek.calibration.model import SensorModel, digest
from altiseek.config import load_config
from altiseek.env.config import SearchConfig
from altiseek.env.navigation import Navigation
from altiseek.env.search import SearchEnv
from altiseek.policies.baselines import Baseline
from altiseek.policies.config import SelectedBaselines
from altiseek.policies.imitation import ImitationConfig, collect_episode, load_shard, split_worlds
from altiseek.registry.run import sha256_file, write_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/policy/imitation_live.yaml"))
    parser.add_argument("--out", type=Path, default=Path("data/m12/demos"))
    parser.add_argument("--stop-after", type=int, help="pause after this many saved episodes")
    args = parser.parse_args()
    cfg = load_config(args.config, ImitationConfig)
    search_path = Path("configs/env/search_live.yaml")
    search = load_config(search_path, SearchConfig).model_copy(update={"discount": 1.0})
    cache = load_config("configs/cache/default.yaml", CacheConfig)
    sensor_path = Path("data/sensor_models/fcf8cdae58ca3947.json")
    sensor = SensorModel.load(
        sensor_path,
        detector_id=cache.detector_id,
        fit_world_set_id=cache.world_set_ids["sensor-fit"],
    )
    selected_path = Path("configs/policy/selected_live.yaml")
    selected = load_config(selected_path, SelectedBaselines)
    if (
        selected.source_run_id != "634beeb0cc8f4664"
        or selected.sensor_model_id != sensor.sensor_model_id
    ):
        raise ValueError("D65 expert selection mismatch")
    worlds = load_worlds(cache, "rl-train")
    partition = split_worlds(worlds, cfg.holdout_per_family)
    if len(worlds) != 160 or len(partition["holdout"]) != 16:
        raise ValueError("D66 requires the original 160 B worlds and 16-world holdout")
    inputs = {
        "config": args.config,
        "search": search_path,
        "sensor": sensor_path,
        "selected": selected_path,
        "energy": Path("configs/env/energy_live.yaml"),
    }
    for world in worlds:
        path = cache.out_dir / "rl-train" / f"{world.world_id}.npz"
        if resume_metadata(path, world, cache) is None:
            raise ValueError("Unverified B demonstration cache")
        inputs[world.world_id] = path
    sources = [
        Path("scripts/ppo/collect_demos.py"),
        Path("src/altiseek/policies/imitation.py"),
        Path("src/altiseek/policies/baselines.py"),
        Path("src/altiseek/policies/ppo.py"),
        *Path("src/altiseek/env").glob("*.py"),
        *Path("src/altiseek/belief").glob("*.py"),
    ]
    plan = {
        "implementation": "m12-b5-demos-v1",
        "config": cfg.model_dump(mode="json"),
        "search": search.model_dump(mode="json"),
        "partition": partition,
        "world_set_id": cache.world_set_ids["rl-train"],
        "world_ids": [w.world_id for w in worlds],
        "cache_ids": [header_for(w, cache).cache_id for w in worlds],
        "input_hashes": {name: sha256_file(path) for name, path in inputs.items()},
        "source_hashes": {p.as_posix(): sha256_file(p) for p in sorted(sources)},
    }
    root = args.out / digest(plan)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "plan.json", plan)
    if (root / "complete.json").exists():
        print(f"Already complete; preserving {root}", flush=True)
        return
    print(f"Demonstrations {root}; 160 worlds x {cfg.repeats} resets", flush=True)
    pool = CachePool(40)
    nav = Navigation(cache.render.levels, cache.render.camera.hfov_deg, search.energy)
    episodes = []
    started = time.perf_counter()
    for repeat in range(cfg.repeats):
        for world in worlds:
            seed = cfg.reset_seed_base + repeat * 1000000 + world.seed
            name = f"{world.world_id}-{repeat}"
            marker = root / f"{name}.json"
            path = root / f"{name}.npz"
            if marker.exists():
                saved = json.loads(marker.read_text(encoding="utf-8"))
                if (
                    saved["collection_id"] != root.name
                    or saved["world_id"] != world.world_id
                    or saved["repeat"] != repeat
                    or saved["seed"] != seed
                ):
                    raise ValueError("Demonstration resume identity mismatch")
                load_shard(path, saved["sha256"])
            else:
                env = SearchEnv([world], cache, sensor, search, navigation=nav, pool=pool)
                expert = Baseline(
                    "B5",
                    nav,
                    sensor,
                    "V3",
                    seed=seed + 1,
                    parameters=selected.settings[world.family]["V3"]["B5"],
                )
                arrays, summary = collect_episode(env, expert, seed)
                temporary = path.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, **arrays)
                temporary.replace(path)
                saved = {
                    "collection_id": root.name,
                    "world_id": world.world_id,
                    "world_set": "rl-train",
                    "split": "B",
                    "family": world.family,
                    "repeat": repeat,
                    "seed": seed,
                    "file": path.name,
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    **summary,
                }
                write_json(marker, saved)
            episodes.append(saved)
            print(
                f"{len(episodes)}/{len(worlds) * cfg.repeats}: {name} "
                f"transitions={saved['transitions']}",
                flush=True,
            )
            if args.stop_after is not None and len(episodes) >= args.stop_after:
                write_json(
                    root / "inventory.json",
                    {
                        "collection_id": root.name,
                        "episodes": len(episodes),
                        "transitions": sum(e["transitions"] for e in episodes),
                        "bytes": sum(e["bytes"] for e in episodes),
                        "elapsed_s_this_launch": time.perf_counter() - started,
                    },
                )
                print(f"Inventory stop; resume {root} without --stop-after", flush=True)
                return
    result = {
        "collection_id": root.name,
        "episodes": episodes,
        "transitions": sum(e["transitions"] for e in episodes),
        "bytes": sum(e["bytes"] for e in episodes),
        "partition": partition,
    }
    write_json(root / "complete.json", result)
    write_metadata(
        root / "run.meta.json",
        stage="collect-m12-demos",
        config=cfg,
        inputs=inputs,
        outputs={"plan": root / "plan.json", "completion": root / "complete.json"},
    )
    print(f"Completed {root}: {result['transitions']} transitions", flush=True)


if __name__ == "__main__":
    main()
