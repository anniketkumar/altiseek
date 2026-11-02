"""Audit complete B calibration and freeze M10 artifacts before any new T run."""

import argparse
import json
from pathlib import Path

from altiseek.cache.config import CacheConfig
from altiseek.cache.jobs import load_libraries, load_worlds, resume_metadata
from altiseek.calibration.model import SensorModel, digest
from altiseek.config import load_config
from altiseek.deploy.calibration import load_view
from altiseek.registry.run import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/m10/transfer"))
    args = parser.parse_args()
    root = args.calibration
    status = json.loads((root / "status.json").read_text())
    if (
        status["state"] != "awaiting_calibration_review"
        or status["completed_worlds"] != 40
        or status["completed_views"] != 8800
    ):
        raise ValueError("Complete D52 B calibration/refit is required")
    cache = load_config("configs/cache/default.yaml", CacheConfig)
    load_libraries(cache)  # Verify original imagery hashes once before frozen export.
    b_worlds = load_worlds(cache, "sensor-fit")
    definition = json.loads((root / "definition.json").read_text())
    if definition["world_ids"] != [w.world_id for w in b_worlds]:
        raise ValueError("B fit-world identity/order mismatch")
    contract = json.loads((root / "contract.json").read_text())
    keys = [
        (li, ix, iy)
        for li, level in enumerate(cache.render.levels.levels)
        for ix, iy in cache.render.levels.lattice(level)
    ]
    ids, collection = [], {}
    for world in b_worlds:
        hashes = []
        for key in keys:
            item = load_view(root / "views" / world.world_id, key, digest(contract))
            if item is None or item[0]["world_id"] != world.world_id:
                raise ValueError("Incomplete/mixed B calibration collection")
            hashes.append(item[0]["sha256"])
        ids.append(digest(hashes))
        collection[world.world_id] = hashes
    model_path = Path(status["model_path"])
    refit = SensorModel.load(
        model_path,
        detector_id=cache.detector_id,
        fit_world_set_id=cache.world_set_ids["sensor-fit"],
    )
    if (
        sha256_file(model_path) != status["model_sha256"]
        or refit.fit_cache_ids != tuple(sorted(ids))
        or refit.cache_contract.get("deployment") != contract
    ):
        raise ValueError("Refit model/collection provenance mismatch")
    stats = json.loads((root / "fit.json").read_text())
    if stats["worlds"] != 40 or len(stats["levels"]) != 5:
        raise ValueError("Incomplete B fit report")
    for level in stats["levels"]:
        if (
            level["binary"]["recall"]["denominator"] <= 0
            or level["binary"]["false_alarm"]["denominator"] <= 0
        ):
            raise ValueError("B calibration lacks occupied or empty trials")
    args.out.mkdir(parents=True, exist_ok=True)
    collection_file = args.out / f"calibration-index-{status['run_id']}.json"
    collection_file.write_text(json.dumps(collection, indent=2) + "\n")
    original_path = Path("data/sensor_models/fcf8cdae58ca3947.json")
    original = SensorModel.load(
        original_path,
        detector_id=cache.detector_id,
        fit_world_set_id=cache.world_set_ids["sensor-fit"],
    )
    worlds = load_worlds(cache, "test")
    families = sorted({w.family for w in worlds})
    if len(families) != 8:
        raise ValueError("Expected eight fixed T scenario families")
    selected = []
    for family in families:
        group = sorted(
            (w for w in worlds if w.family == family),
            key=lambda w: digest(["m10-transfer-selection-v1", w.world_id]),
        )
        selected.extend(group[:3])
    checkpoints = [
        p
        for p in json.loads(Path("configs/eval/artifacts_v1.json").read_text())["checkpoints"]
        if p["variant"] == "V3"
    ]
    if sorted(p["seed"] for p in checkpoints) != [17, 29, 43]:
        raise ValueError("All three frozen V3 checkpoints are required")
    inputs = {str(p): sha256_file(p) for p in Path("src/altiseek").rglob("*.py")}
    for pattern in ("*flight*.py", "freeze_m10_transfer.py", "run_m10_transfer.py", "export_deploy_world.py"):
        inputs.update({str(p): sha256_file(p) for p in Path("scripts").glob(pattern)})
    inputs.update({str(p): sha256_file(p) for p in Path("configs").rglob("*.yaml")})
    for p in [
        Path("docs/m10-transfer-design.md"),
        Path("configs/cache/default.yaml"),
        Path("configs/env/search.yaml"),
        Path("configs/env/energy.yaml"),
        Path("configs/policy/selected.yaml"),
        Path("configs/eval/artifacts_v1.json"),
        root / "definition.json",
        root / "contract.json",
        root / "fit.json",
        collection_file,
        original_path,
        model_path,
        cache.checkpoint,
    ]:
        inputs[str(p)] = sha256_file(p)
    for item in checkpoints:
        if sha256_file(item["checkpoint"]) != item["sha256"]:
            raise ValueError("Frozen PPO checkpoint changed")
        inputs[item["checkpoint"]] = item["sha256"]
    for world in selected:
        path = cache.out_dir / "test" / f"{world.world_id}.npz"
        if resume_metadata(path, world, cache) is None:
            raise ValueError("Unverified cached reference input")
        inputs[str(path)] = sha256_file(path)
        spec_path = cache.worlds_dir / "test" / f"{world.world_id}.json"
        inputs[str(spec_path)] = sha256_file(spec_path)
    sensors = {
        name: {
            "path": str(path),
            "sha256": sha256_file(path),
            "sensor_model_id": model.sensor_model_id,
            "cache_contract": model.cache_contract,
        }
        for name, path, model in (
            ("original", original_path, original),
            ("refit", model_path, refit),
        )
    }
    protocol = {
        "status": "frozen",
        "schema_version": 1,
        "design": "m10-transfer-v1",
        "world_ids": [w.world_id for w in selected],
        "worlds": [{"world_id": w.world_id, "seed": w.seed, "family": w.family} for w in selected],
        "checkpoints": checkpoints,
        "sensors": sensors,
        "inputs": inputs,
        "b_preflight_world_id": "003b0c54f581c65b",
        "live_episodes": 192,
        "cached_episodes": 96,
        "calibration_run": str(root),
        "calibration_quality_note": (
            "Integrity verified; quality limitations remain in the full B fit report. "
            "No T outcomes used."
        ),
    }
    protocol["protocol_id"] = digest(protocol)
    destination = args.out / f"{protocol['protocol_id']}.json"
    encoded = json.dumps(protocol, indent=2) + "\n"
    if destination.exists() and destination.read_text() != encoded:
        raise ValueError("Refuse to replace a different frozen transfer manifest")
    destination.write_text(encoded)
    (args.out / "protocol-path.txt").write_text(str(destination) + "\n")
    print(destination, flush=True)


if __name__ == "__main__":
    main()
