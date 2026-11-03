"""Complete M7's false-clear diagnostic with the initial observations omitted by step logs."""

import argparse
import json
from pathlib import Path

import numpy as np

from altiseek.cache.config import CacheConfig
from altiseek.cache.jobs import load_worlds, resume_metadata, write_json
from altiseek.cache.storage import CachePool
from altiseek.calibration.model import SensorModel, digest
from altiseek.config import load_config
from altiseek.env.config import SearchConfig
from altiseek.env.navigation import Navigation
from altiseek.env.search import SearchEnv
from altiseek.registry.run import sha256_file, write_metadata
from altiseek.world.sampler import prior_grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sanity_root", type=Path)
    args = parser.parse_args()
    plan = json.loads((args.sanity_root / "plan.json").read_text(encoding="utf-8"))
    cache = load_config("configs/cache/default.yaml", CacheConfig)
    cfg = SearchConfig.model_validate(plan["config"]).model_copy(
        update={"sensor_variant": "V0", "start_level": 4}
    )
    model_path = Path("data/sensor_models/fcf8cdae58ca3947.json")
    sensor = SensorModel.load(model_path, detector_id=cache.detector_id)
    if (
        plan["sensor_model_id"] != sensor.sensor_model_id
        or plan["world_set_id"] != cache.world_set_ids["rl-val"]
    ):
        raise ValueError("sanity input mismatch")
    nav = Navigation(cache.render.levels, cache.render.camera.hfov_deg, cfg.energy)
    pool = CachePool(40)
    rows, inputs = [], {"model": model_path, "source_summary": args.sanity_root / "summary.json"}
    for world in load_worlds(cache, "rl-val"):
        path = cache.out_dir / "rl-val" / f"{world.world_id}.npz"
        if resume_metadata(path, world, cache) is None:
            raise ValueError("unverified cache")
        inputs[world.world_id] = path
        env = SearchEnv([world], cache, sensor, cfg, navigation=nav, pool=pool)
        env.reset(seed=plan["seed"] + world.seed)
        q = prior_grid(world.prior_given, world.area_m, world.cell_m)
        p = -np.expm1(len(world.targets) * np.log1p(-q))
        cells = env._target_cells
        before = p[cells[:, 0], cells[:, 1]]
        after = env.belief.probabilities[cells[:, 0], cells[:, 1]]
        negative = (after < before - 1e-12) & ~env.found
        crossed = negative & (before >= 0.01) & (after < 0.01)
        rows.append(
            {
                "world_id": world.world_id,
                "targets": len(world.targets),
                "initially_below_threshold": int((before < 0.01).sum()),
                "negative_updates": int(negative.sum()),
                "crossings": int(crossed.sum()),
                "crossing_before": before[crossed].tolist(),
                "crossing_after": after[crossed].tolist(),
            }
        )
    result = {
        "diagnostic_version": "initial-observation-v1",
        "threshold": 0.01,
        "source_summary_sha256": sha256_file(args.sanity_root / "summary.json"),
        "source_run_id": args.sanity_root.name,
        "sensor_model_id": sensor.sensor_model_id,
        "world_set_id": cache.world_set_ids["rl-val"],
        "rows": rows,
        "initial_crossings": sum(r["crossings"] for r in rows),
        "initial_negative_updates": sum(r["negative_updates"] for r in rows),
        "targets": sum(r["targets"] for r in rows),
        "initially_below_threshold": sum(r["initially_below_threshold"] for r in rows),
    }
    report = args.sanity_root / f"initial-{digest(result)}.json"
    write_json(report, result)
    write_metadata(
        report.with_suffix(".meta.json"),
        stage="check-initial-observation",
        config=cfg,
        inputs=inputs,
        outputs={"report": report},
    )
    print(
        f"Initial crossings: {result['initial_crossings']}; "
        f"initially below threshold: {result['initially_below_threshold']}/{result['targets']}; "
        f"{report}"
    )


if __name__ == "__main__":
    main()
