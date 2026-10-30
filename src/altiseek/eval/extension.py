"""D47 frozen M9 extension: artifact checks, paired keys and resumable episodes."""

import platform
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from filelock import FileLock
from sb3_contrib import MaskablePPO

from altiseek.cache.build import header_for
from altiseek.cache.config import CacheConfig
from altiseek.cache.jobs import load_worlds, write_json
from altiseek.cache.storage import CachePool
from altiseek.calibration.model import SensorModel, digest
from altiseek.config import Config, load_config
from altiseek.env.config import SearchConfig
from altiseek.env.navigation import Navigation
from altiseek.env.search import SearchEnv
from altiseek.eval.runner import play, policies, read, save_episode, verified_episode
from altiseek.policies.config import SelectedBaselines
from altiseek.registry.run import sha256_file, write_metadata
from altiseek.world.spec import WorldSpec

ABLATIONS = (
    "control",
    "relook",
    "declare",
    "unknown_k",
    "shaping",
    "egocentric",
    "appearance",
    "appearance_control",
)


class ExtensionProtocol(Config):
    schema_version: Literal[2] = 2
    environment_seed: Literal[20261027] = 20261027
    action_seed: Literal[20261028] = 20261028
    action_repeats: Literal[3] = 3
    bootstrap_seed: Literal[20261029] = 20261029
    bootstrap_samples: Literal[2000] = 2000
    battery_s: Literal[500] = 500
    ablations: tuple[str, ...] = ABLATIONS
    robustness: tuple[str, ...] = ("detector", "camera", "stress-low", "stress-high")


def selected_run(root: Path) -> dict[str, Any]:
    """Recompute B-only raw-AUC selection and verify every training checkpoint."""
    plan, done = read(root / "plan.json"), read(root / "complete.json")
    if digest(plan) != root.name or done["run_id"] != root.name or plan["smoke"]:
        raise ValueError("invalid production training identity")
    if plan["config"]["selection_metric"] != "mean_auc" or done["steps"] != 131072:
        raise ValueError("wrong training budget/selection rule")
    expected_keys = {(wid, r) for wid in plan["world_ids"]["rl-val"] for r in range(3)}
    reports = []
    for step in range(16384, 131073, 16384):
        path = root / f"eval-{step:09d}.json"
        report = read(path)
        episodes = report["episodes"]
        if (
            report["steps"] != step
            or report["zero_belief"]
            or report["action_rule"] != "sampled"
            or report["repeats"] != 3
            or len(episodes) != len(expected_keys)
            or {(e["world_id"], e["repeat"]) for e in episodes} != expected_keys
            or not np.isclose(
                np.mean([e["metrics"]["auc"] for e in episodes]),
                report["mean_auc"],
                rtol=0,
                atol=1e-12,
            )
        ):
            raise ValueError("invalid B validation report")
        checkpoint = root / f"step-{step:09d}"
        complete = read(checkpoint / "complete.json")
        if complete["steps"] != step or any(
            sha256_file(checkpoint / name) != value for name, value in complete["hashes"].items()
        ):
            raise ValueError("checkpoint checksum mismatch")
        reports.append(report)
    best = max(reports, key=lambda r: (r["mean_auc"], -r["steps"]))
    path = root / f"step-{best['steps']:09d}" / "model.zip"
    if done["selected_step"] != best["steps"] or done["checkpoint_sha256"] != sha256_file(path):
        raise ValueError("completion differs from B checkpoint selection")
    return {
        "variant": "V3",
        "seed": plan["seed"],
        "run_id": root.name,
        "checkpoint": path.as_posix(),
        "sha256": sha256_file(path),
        "selected_step": best["steps"],
        "validation_auc": best["mean_auc"],
        "search": plan["search"],
        "plan_sha256": sha256_file(root / "plan.json"),
        "validation_sha256": sha256_file(root / done["validation"]),
        "training_worlds": len(plan["world_ids"]["rl-train"]),
        "validation_worlds": len(plan["world_ids"]["rl-val"]),
    }


def keys(protocol: ExtensionProtocol, manifest: dict, *, smoke: bool = False) -> list[dict]:
    jobs = []
    for case, dataset in manifest["datasets"].items():
        worlds = dataset["smoke_worlds"] if smoke else dataset["worlds"]
        selected = (
            [
                {"method": "PPO", "variant": "V3", "training_seed": p["seed"]}
                for p in manifest["ablations"][case]
            ]
            if case in ABLATIONS
            else policies(manifest["historical"])
        )
        for world in worlds:
            for policy in selected:
                for repeat in range(3 if policy["method"] in ("PPO", "random") else 1):
                    jobs.append(
                        {
                            "scenario": case,
                            "world_id": world["world_id"],
                            **policy,
                            "repeat": repeat,
                            "environment_seed": protocol.environment_seed + world["seed"],
                            "action_seed": protocol.action_seed + 1000000 * repeat + world["seed"],
                        }
                    )
    return jobs


def validate_manifest(manifest: dict) -> None:
    if manifest["schema_version"] != 2 or set(manifest["ablations"]) != set(ABLATIONS):
        raise ValueError("invalid extension manifest")
    if set(manifest["datasets"]) != set(ABLATIONS) | {
        "detector",
        "camera",
        "stress-low",
        "stress-high",
    }:
        raise ValueError("missing extension datasets")
    for case, checkpoints in manifest["ablations"].items():
        if len(checkpoints) != 3 or {p["seed"] for p in checkpoints} != {17, 29, 43}:
            raise ValueError(f"missing training seeds: {case}")
        for checkpoint in checkpoints:
            root = Path(checkpoint["checkpoint"]).parent.parent
            if selected_run(root) != checkpoint:
                raise ValueError("frozen selection changed")
    for path, checksum in manifest["files"].items():
        if sha256_file(path) != checksum:
            raise ValueError(f"frozen input checksum mismatch: {path}")


def play_extension(env: SearchEnv, key: dict, parameters: dict, model: Any) -> dict:
    result = play(env, key, parameters, model)
    # v1's diagnostic initializer assumes known K. Recompute its initial crossing
    # from the actual unknown-K reset rather than mislabelling a prior difference.
    if not env.config.known_count:
        initial = SearchEnv(env._worlds, env.cache_config, env.sensor, env.config, pool=env.pool)
        initial.reset(seed=key["environment_seed"])
        from altiseek.world.sampler import prior_grid

        q = prior_grid(initial.world.prior_given, initial.world.area_m, initial.world.cell_m)
        cells = initial._target_cells
        before = 1 - (1 - q[cells[:, 0], cells[:, 1]]) ** initial.config.assumed_count
        after = initial.belief.probabilities[cells[:, 0], cells[:, 1]]
        crossing = int(((before >= 0.01) & (after < 0.01) & ~initial.found).sum())
        result["metrics"]["initial_false_clear_crossings"] = crossing
        result["metrics"]["total_false_clear_crossings"] = (
            result["metrics"]["false_clear_crossings"] + crossing
        )
    return result


def run_extension(
    protocol_path: Path, manifest_path: Path, out: Path, *, smoke: bool, limit: int
) -> Path:
    protocol = load_config(protocol_path, ExtensionProtocol)
    manifest = read(manifest_path)
    validate_manifest(manifest)
    if tuple(manifest["ablations"]) != protocol.ablations:
        raise ValueError("protocol cases mismatch")
    selected = load_config("configs/policy/selected.yaml", SelectedBaselines)
    sources = sorted(Path("src/altiseek").rglob("*.py"))
    identity = {
        "protocol": protocol.model_dump(mode="json"),
        "manifest": manifest,
        "manifest_sha256": sha256_file(manifest_path),
        "smoke": smoke,
        "sources": {p.as_posix(): sha256_file(p) for p in sources},
        "runtime": {
            "python": platform.python_version(),
            "torch_threads": 2,
            "packages": {n: version(n) for n in ("numpy", "torch", "sb3-contrib")},
        },
    }
    root = out / digest(identity)
    root.mkdir(parents=True, exist_ok=True)
    jobs = keys(protocol, manifest, smoke=smoke)
    with FileLock(root / "runner.lock", timeout=0):
        write_json(root / "plan.json", identity)
        write_metadata(
            root / "run.meta.json",
            stage="evaluate-extension",
            config=protocol,
            inputs={"protocol": protocol_path, "manifest": manifest_path},
            outputs={"plan": root / "plan.json"},
        )
        torch.set_num_threads(2)
        contexts = {}
        for case, dataset in manifest["datasets"].items():
            cache = CacheConfig.model_validate(
                dataset["smoke_cache"] if smoke else dataset["cache"]
            )
            name = dataset["smoke_set"] if smoke else dataset["world_set"]
            worlds = {w.world_id: w for w in load_worlds(cache, name)}
            listed = dataset["smoke_worlds"] if smoke else dataset["worlds"]
            if any(worlds[w["world_id"]].seed != w["seed"] for w in listed):
                raise ValueError("world seed mismatch")
            sensor = SensorModel.load(
                Path(dataset["sensor"]),
                detector_id=cache.detector_id,
                fit_world_set_id="5dc45d0e1ea5afbd",
            )
            contexts[case] = (cache, worlds, sensor, CachePool(80))
        loaded = {}
        navigation = {}
        completed = added = 0
        started = time.perf_counter()
        print(f"Run {root}: {len(jobs)} episodes; smoke={smoke}", flush=True)
        for key in jobs:
            path = root / "episodes" / f"{digest(key)}.json"
            if verified_episode(path, root.name, key) is not None:
                completed += 1
                continue
            if limit and added >= limit:
                break
            case = key["scenario"]
            cache, worlds, sensor, pool = contexts[case]
            world: WorldSpec = worlds[key["world_id"]]
            model = None
            config = load_config("configs/env/search.yaml", SearchConfig)
            if key["method"] == "PPO":
                choices = manifest["ablations"].get(case, manifest["historical"]["checkpoints"])
                checkpoint = next(
                    p
                    for p in choices
                    if p["seed"] == key["training_seed"] and p["variant"] == key["variant"]
                )
                if "search" in checkpoint:
                    config = SearchConfig.model_validate(checkpoint["search"])
                name = checkpoint["checkpoint"]
                if name not in loaded:
                    loaded[name] = MaskablePPO.load(name, device="cpu")
                    loaded[name].policy.set_training_mode(False)
                model = loaded[name]
                if model.policy.features_extractor.zero_belief:
                    raise ValueError("unexpected belief removal")
            config = config.model_copy(update={"sensor_variant": key["variant"]})
            nav_key = digest(config.energy.model_dump(mode="json"))
            if nav_key not in navigation:
                navigation[nav_key] = Navigation(
                    cache.render.levels, cache.render.camera.hfov_deg, config.energy
                )
            env = SearchEnv(
                [world],
                cache,
                sensor,
                config,
                pool=pool,
                navigation=navigation[nav_key],
                oracle=key["method"] == "oracle",
            )
            if model is not None and (
                model.observation_space != env.observation_space
                or model.action_space != env.action_space
            ):
                raise ValueError("checkpoint/environment spaces mismatch")
            params = {} if model else selected.settings[world.family][key["variant"]][key["method"]]
            result = play_extension(env, key, params, model)
            save_episode(
                path,
                {
                    "run_id": root.name,
                    "key": key,
                    "family": world.family,
                    "condition": world.condition,
                    "target_count": len(world.targets),
                    "cache_id": header_for(world, cache).cache_id,
                    **result,
                },
            )
            completed += 1
            added += 1
            write_json(
                root / "status.json",
                {"completed": completed, "total": len(jobs), "complete": False},
            )
            if added % 50 == 0:
                print(f"{completed}/{len(jobs)}; {time.perf_counter() - started:.1f} s", flush=True)
        write_json(
            root / "status.json",
            {"completed": completed, "total": len(jobs), "complete": completed == len(jobs)},
        )
        print(f"Stopped cleanly: {completed}/{len(jobs)}", flush=True)
    return root
