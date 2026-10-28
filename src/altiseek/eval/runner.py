"""Paired, resumable evaluation of fixed policies; no fitting or selection."""

import json
import platform
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from filelock import FileLock
from sb3_contrib import MaskablePPO

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
from altiseek.eval.config import EvaluationProtocol, Scenario
from altiseek.policies.baselines import Baseline
from altiseek.policies.config import SelectedBaselines
from altiseek.registry.run import sha256_file, write_metadata
from altiseek.world.sampler import prior_grid
from altiseek.world.spec import WorldSpec


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verified_episode(path: Path, run_id: str, key: dict[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    saved = read(path)
    checksum = saved.pop("checksum")
    if checksum != digest(saved) or saved["run_id"] != run_id or saved["key"] != key:
        raise ValueError(f"episode checksum/identity mismatch: {path}")
    return saved


def save_episode(path: Path, result: dict[str, Any]) -> None:
    write_json(path, {**result, "checksum": digest(result)})


def policies(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Stable policy order: all deterministic references, random draws, PPO seeds."""
    result = [
        {"variant": variant, "method": method, "training_seed": None}
        for variant in ("V0", "V1", "V2", "V3")
        for method in ("random", "oracle", "B1", "B2", "B3", "B4", "B5")
    ]
    result.extend(
        {"variant": p["variant"], "method": "PPO", "training_seed": p["seed"]}
        for p in manifest["checkpoints"]
    )
    return result


def episode_keys(
    protocol: EvaluationProtocol, manifest: dict[str, Any], worlds: list[WorldSpec]
) -> list[dict[str, Any]]:
    return [
        {
            "scenario": scenario.name,
            "world_id": world.world_id,
            **policy,
            "repeat": repeat,
            "environment_seed": protocol.environment_seed + world.seed,
            "action_seed": protocol.action_seed + 1000000 * repeat + world.seed,
        }
        for scenario in protocol.scenarios
        for world in worlds
        for policy in policies(manifest)
        for repeat in range(protocol.action_repeats if policy["method"] in ("random", "PPO") else 1)
    ]


def validate_artifacts(manifest: dict[str, Any], selected_path: Path) -> None:
    if manifest["schema_version"] != 1:
        raise ValueError("unsupported frozen-artifact schema")
    selected = load_config(selected_path, SelectedBaselines)
    if digest(selected.model_dump(mode="json")) != manifest["baselines_id"]:
        raise ValueError("frozen baseline checksum mismatch")
    selection_report = read(Path(manifest["selection_report"]))
    if digest(selection_report) != manifest["selection_report_id"]:
        raise ValueError("B selection report checksum mismatch")
    expected = {(variant, seed) for variant in ("V0", "V3") for seed in (17, 29, 43)}
    checkpoints = manifest["checkpoints"]
    if len(checkpoints) != 6 or {(p["variant"], p["seed"]) for p in checkpoints} != expected:
        raise ValueError("must preserve all six selected policies")
    selection = selection_report["selected"]
    for checkpoint in checkpoints:
        source = next(
            s
            for s in selection
            if (s["variant"], s["seed"]) == (checkpoint["variant"], checkpoint["seed"])
        )
        if any(source[k] != v for k, v in checkpoint.items()):
            raise ValueError("checkpoint differs from frozen B selection")
        if sha256_file(checkpoint["checkpoint"]) != checkpoint["sha256"]:
            raise ValueError("selected checkpoint checksum mismatch")


def play(
    env: SearchEnv,
    key: dict[str, Any],
    parameters: dict[str, Any],
    model: MaskablePPO | None,
) -> dict[str, Any]:
    logger = EpisodeLog(env)
    obs, info = logger.reset(seed=key["environment_seed"])
    q = prior_grid(env.prior_override or env.world.prior_given, env.world.area_m, env.world.cell_m)
    cells = env._target_cells
    before = 1 - (1 - q[cells[:, 0], cells[:, 1]]) ** len(env.found)
    after = env.belief.probabilities[cells[:, 0], cells[:, 1]]
    initial_crossings = (
        int(((before >= 0.01) & (after < 0.01) & ~env.found).sum())
        if env.nav.nodes[env.node].level == env.level_count - 1
        else 0
    )
    rng = np.random.default_rng(key["action_seed"])
    policy = (
        None
        if model is not None
        else Baseline(
            key["method"],
            env.nav,
            env.sensor,
            key["variant"],
            seed=key["action_seed"],
            parameters=parameters,
            oracle_cells=env._target_cells.copy() if key["method"] == "oracle" else None,
        )
    )
    opportunities = descents = 0
    while not info["ended"]:
        mask = info["action_mask"]
        if model is None:
            assert policy is not None
            action = policy.act(obs, mask)
        else:
            tensors, _ = model.policy.obs_to_tensor(obs)
            with torch.no_grad():
                p = model.policy.get_distribution(tensors, action_masks=mask)
                probabilities = p.distribution.probs.cpu().numpy()[0].astype(float)
            action = int(rng.choice(len(probabilities), p=probabilities / probabilities.sum()))
        if not mask[action]:
            raise ValueError("policy selected masked action")
        # Behaviour measured against actual cached swimmer evidence, not hidden occupancy.
        node = env.nav.nodes[env.node]
        if node.level >= 3 and "variant" in info["events"]:
            detections, _ = env.cache.lookup(
                node.level, node.ix, node.iy, info["events"]["variant"]
            )
            detected = bool(
                np.any(
                    (detections["class_id"] == 0)
                    & (detections["score"] >= env.sensor.fit_config.tau)
                )
            )
            opportunities += int(detected)
            descents += int(detected and action == 5)
        obs, _, _, _, info = logger.step(action)
    metrics = logger.summary()
    metrics["initial_false_clear_crossings"] = initial_crossings
    metrics["total_false_clear_crossings"] = metrics["false_clear_crossings"] + initial_crossings
    return {
        "metrics": metrics,
        "records": logger.records,
        "coarse_detection_opportunities": opportunities,
        "immediate_descents": descents,
    }


def run(protocol_path: Path, manifest_path: Path, out: Path, *, smoke: bool, limit: int) -> Path:
    protocol = load_config(protocol_path, EvaluationProtocol)
    manifest = read(manifest_path)
    selected_path = Path("configs/policy/selected.yaml")
    validate_artifacts(manifest, selected_path)
    selected = load_config(selected_path, SelectedBaselines)
    cache = load_config("configs/cache/default.yaml", CacheConfig)
    search = load_config("configs/env/search.yaml", SearchConfig)
    sensor_path = Path(f"data/sensor_models/{protocol.sensor_model_id}.json")
    sensor = SensorModel.load(
        sensor_path,
        detector_id=cache.detector_id,
        fit_world_set_id=cache.world_set_ids["sensor-fit"],
    )
    if (
        selected.sensor_model_id != sensor.sensor_model_id
        or selected.world_set_id != cache.world_set_ids["rl-val"]
    ):
        raise ValueError("baseline/sensor/validation identity mismatch")
    if cache.world_set_ids["test"] != protocol.world_set_id:
        raise ValueError("test world-set mismatch")
    worlds = load_worlds(cache, "rl-val" if smoke else "test")
    if smoke:
        worlds = worlds[:1]
    elif [{"world_id": w.world_id, "seed": w.seed} for w in worlds] != manifest["worlds"]:
        raise ValueError("frozen test world IDs/seeds mismatch")
    inputs = {
        "protocol": protocol_path,
        "manifest": manifest_path,
        "baselines": selected_path,
        "sensor": sensor_path,
        "search": Path("configs/env/search.yaml"),
    }
    inputs.update({f"source:{p.as_posix()}": p for p in Path("src/altiseek").rglob("*.py")})
    inputs["runner"] = Path("scripts/evaluate.py")
    for world in worlds:
        path = cache.out_dir / world.world_set / f"{world.world_id}.npz"
        if resume_metadata(path, world, cache) is None:
            raise ValueError("unverified evaluation cache")
        inputs[f"cache:{world.world_id}"] = path
        inputs[f"world:{world.world_id}"] = (
            cache.worlds_dir / world.world_set / f"{world.world_id}.json"
        )
    for model in manifest["checkpoints"]:
        inputs[f"checkpoint:{model['variant']}:{model['seed']}"] = Path(model["checkpoint"])
    identity = {
        "runtime": {
            "python": platform.python_version(),
            "packages": {
                name: version(name)
                for name in ("numpy", "torch", "sb3-contrib", "stable-baselines3")
            },
            "torch_threads": 2,
        },
        "protocol": protocol.model_dump(mode="json"),
        "smoke": smoke,
        "inputs": {name: sha256_file(path) for name, path in sorted(inputs.items())},
        "search": search.model_dump(mode="json"),
        "cache": cache.model_dump(mode="json"),
    }
    root = out / digest(identity)
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(root / "runner.lock", timeout=0):
        plan_path = root / "plan.json"
        if plan_path.exists() and read(plan_path) != identity:
            raise ValueError("run plan mismatch")
        write_json(plan_path, identity)
        if not (root / "run.meta.json").exists():
            write_metadata(
                root / "run.meta.json",
                stage="evaluate",
                config=protocol,
                inputs=inputs,
                outputs={"plan": plan_path},
                extra={"run_id": root.name},
            )
        jobs = episode_keys(protocol, manifest, worlds)
        if smoke:
            # All policy/seed combinations nominal; one random draw for every perturbation.
            jobs = [
                k
                for k in jobs
                if k["scenario"] == "nominal"
                or (k["method"] == "random" and k["variant"] == "V3" and k["repeat"] == 0)
            ]
        print(f"Run {root}: {len(jobs)} episodes; smoke={smoke}", flush=True)
        torch.set_num_threads(2)
        pool = CachePool(capacity=80)
        loaded: dict[tuple[str, int], MaskablePPO] = {}
        worlds_by_id = {w.world_id: w for w in worlds}
        scenarios = {s.name: s for s in protocol.scenarios}
        navigation: dict[str, Navigation] = {}
        completed = newly_completed = 0
        started = time.perf_counter()
        for key in jobs:
            path = root / "episodes" / f"{digest(key)}.json"
            if verified_episode(path, root.name, key) is not None:
                completed += 1
                continue
            if limit and newly_completed >= limit:
                break
            scenario: Scenario = scenarios[key["scenario"]]
            world = worlds_by_id[key["world_id"]]
            config = search.model_copy(
                update={
                    "sensor_variant": key["variant"],
                    "energy": search.energy.model_copy(update={"battery_s": scenario.battery_s}),
                }
            )
            if scenario.name not in navigation:
                navigation[scenario.name] = Navigation(
                    cache.render.levels, cache.render.camera.hfov_deg, config.energy
                )
            env = SearchEnv(
                [world],
                cache,
                sensor,
                config,
                pool=pool,
                navigation=navigation[scenario.name],
                oracle=key["method"] == "oracle",
                prior_override=scenario.prior(world.prior_given),
            )
            model = None
            if key["method"] == "PPO":
                model_key = (key["variant"], key["training_seed"])
                if model_key not in loaded:
                    checkpoint = next(
                        p for p in manifest["checkpoints"] if (p["variant"], p["seed"]) == model_key
                    )
                    loaded[model_key] = MaskablePPO.load(checkpoint["checkpoint"], device="cpu")
                    loaded[model_key].policy.set_training_mode(False)
                model = loaded[model_key]
                if model.policy.features_extractor.zero_belief:
                    raise ValueError("main evaluation cannot remove belief")
                if (
                    model.observation_space != env.observation_space
                    or model.action_space != env.action_space
                ):
                    raise ValueError("checkpoint/environment spaces mismatch")
            params = (
                {}
                if model is not None
                else selected.settings[world.family][key["variant"]][key["method"]]
            )
            result = play(env, key, params, model)
            save_episode(
                path,
                {
                    "run_id": root.name,
                    "key": key,
                    "family": world.family,
                    "cache_id": header_for(world, cache).cache_id,
                    **result,
                },
            )
            completed += 1
            newly_completed += 1
            write_json(
                root / "status.json",
                {"completed": completed, "total": len(jobs), "complete": False},
            )
            if newly_completed % 10 == 0:
                print(
                    f"{completed}/{len(jobs)}; {time.perf_counter() - started:.1f} s this session",
                    flush=True,
                )
        write_json(
            root / "status.json",
            {"completed": completed, "total": len(jobs), "complete": completed == len(jobs)},
        )
        print(f"Stopped cleanly: {completed}/{len(jobs)}", flush=True)
    return root
