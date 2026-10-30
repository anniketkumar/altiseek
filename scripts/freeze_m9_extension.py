"""Verify completed B selections and cache contracts, then freeze M9 v2 artifacts."""

from pathlib import Path

from run_m9_ablations import CASES

from altiseek.cache.build import header_for
from altiseek.cache.config import CacheConfig
from altiseek.cache.jobs import load_worlds, resume_metadata, write_json
from altiseek.calibration.model import SensorModel
from altiseek.config import load_config
from altiseek.env.config import SearchConfig
from altiseek.eval.extension import selected_run
from altiseek.eval.runner import read, validate_artifacts
from altiseek.policies.ppo import PPOConfig
from altiseek.policies.training import training_worlds
from altiseek.registry.run import sha256_file


def main() -> None:
    original = load_config("configs/cache/default.yaml", CacheConfig)
    b_worlds = {s: load_worlds(original, s) for s in ("rl-train", "rl-val")}
    historical = read(Path("configs/eval/artifacts_v1.json"))
    validate_artifacts(historical, Path("configs/policy/selected.yaml"))
    manifest = {
        "schema_version": 2,
        "historical": historical,
        "ablations": {},
        "datasets": {},
        "files": {},
        "v1_run": "runs/eval/m9-v1/341aadf1366d8f05",
    }
    files = manifest["files"]
    for path in (
        "configs/policy/selected.yaml",
        "configs/eval/artifacts_v1.json",
        "configs/env/search.yaml",
        "docs/m9-results/results.json",
        "data/m9-stress/worlds/pairs.json",
    ):
        files[path] = sha256_file(path)
    for case, (policy_name, search_name) in CASES.items():
        policy = load_config(f"configs/policy/{policy_name}.yaml", PPOConfig)
        search = load_config(f"configs/env/{search_name}.yaml", SearchConfig).model_copy(
            update={"sensor_variant": "V3", "discount": policy.gamma}
        )
        worlds = training_worlds(b_worlds, policy)
        expected_worlds = {s: [w.world_id for w in ws] for s, ws in worlds.items()}
        checkpoints = []
        for seed in (17, 29, 43):
            roots = list(Path(f"runs/ppo/m9-ablations/{case}-seed{seed}").glob("*/complete.json"))
            if len(roots) != 1:
                raise ValueError(f"expected one completed production run: {case}/{seed}")
            root = roots[0].parent
            plan = read(root / "plan.json")
            if (
                plan["config"] != policy.model_dump(mode="json")
                or plan["search"] != search.model_dump(mode="json")
                or plan["world_ids"] != expected_worlds
                or plan["seed"] != seed
            ):
                raise ValueError("training config or B world selection differs")
            if any(sha256_file(p) != h for p, h in plan["source_hashes"].items()):
                raise ValueError("training source differs from the frozen implementation")
            checkpoint = selected_run(root)
            checkpoints.append(checkpoint)
            files[checkpoint["checkpoint"]] = checkpoint["sha256"]
            print(
                f"{case} seed {seed}: step {checkpoint['selected_step']}, "
                f"B AUC {checkpoint['validation_auc']:.4f}",
                flush=True,
            )
        manifest["ablations"][case] = checkpoints
    for p in historical["checkpoints"]:
        files[p["checkpoint"]] = p["sha256"]
    for case in (*CASES, "detector", "camera", "stress-low", "stress-high"):
        artifact_case = "stress" if case.startswith("stress-") else case
        cache = (
            original
            if case in CASES
            else load_config(f"data/m9-robustness/{artifact_case}/config.json", CacheConfig)
        )
        name = case if case.startswith("stress-") else "test"
        if case in ("detector", "camera"):
            registered = read(Path(f"data/m9-robustness/{case}/sensor-artifacts.json"))
            if len(registered) != 1:
                raise ValueError("ambiguous replacement sensor")
            sensor_path = registered[0]["location"]
            smoke_cache, smoke_set = cache, "sensor-fit"
        else:
            sensor_path = "data/sensor_models/fcf8cdae58ca3947.json"
            smoke_cache, smoke_set = original, "rl-val"
        sensor = SensorModel.load(
            Path(sensor_path), detector_id=cache.detector_id, fit_world_set_id="5dc45d0e1ea5afbd"
        )
        files[sensor_path] = sha256_file(sensor_path)
        dataset = {
            "cache": cache.model_dump(mode="json"),
            "world_set": name,
            "sensor": sensor_path,
            "sensor_model_id": sensor.sensor_model_id,
            "smoke_cache": smoke_cache.model_dump(mode="json"),
            "smoke_set": smoke_set,
        }
        for field, source, world_set in (
            ("worlds", cache, name),
            ("smoke_worlds", smoke_cache, smoke_set),
        ):
            worlds = load_worlds(source, world_set)
            if field == "smoke_worlds":
                worlds = worlds[:1]
            elif len(worlds) != (40 if case.startswith("stress-") else 80):
                raise ValueError("wrong evaluation world count")
            expected_split = "B" if field == "smoke_worlds" or case.startswith("stress-") else "T"
            if any(w.split != expected_split for w in worlds):
                raise ValueError("evaluation split mismatch")
            dataset[field] = []
            for w in worlds:
                path = source.out_dir / world_set / f"{w.world_id}.npz"
                if resume_metadata(path, w, source) is None:
                    raise ValueError("missing verified cache")
                contract = header_for(w, source).model_dump(
                    mode="json", exclude={"world_id", "cache_id"}
                )
                if contract != sensor.cache_contract:
                    raise ValueError("sensor/cache contract mismatch")
                for p in (
                    path,
                    path.with_suffix(".meta.json"),
                    path.with_suffix(".validation.json"),
                    source.worlds_dir / world_set / f"{w.world_id}.json",
                ):
                    files[p.as_posix()] = sha256_file(p)
                dataset[field].append(
                    {
                        "world_id": w.world_id,
                        "seed": w.seed,
                        "family": w.family,
                        "condition": w.condition,
                        "targets": len(w.targets),
                    }
                )
        manifest["datasets"][case] = dataset
    output = Path("configs/eval/artifacts_v2.json")
    if output.exists() and read(output) != manifest:
        raise ValueError("refusing to change an existing frozen manifest")
    write_json(output, manifest)
    print(f"Frozen {len(files)} input checksums in {output}", flush=True)


if __name__ == "__main__":
    main()
