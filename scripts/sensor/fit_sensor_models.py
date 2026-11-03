"""Fit V0-V3 on verified sensor-fit B caches; optionally evaluate a frozen model on T."""

import argparse
import json
from pathlib import Path

from altiseek.cache.build import header_for
from altiseek.cache.config import CacheConfig
from altiseek.cache.jobs import load_worlds, resume_metadata, write_json
from altiseek.cache.storage import WorldCache
from altiseek.calibration.config import SensorConfig
from altiseek.calibration.fit import calibration_report, fit, measurements
from altiseek.calibration.model import SensorModel
from altiseek.calibration.outcomes import WorldOutcomes, collect_world
from altiseek.config import load_config
from altiseek.registry.manifest import Artifact, find, register
from altiseek.registry.run import sha256_file, write_metadata


def collect(
    cache: CacheConfig, cfg: SensorConfig, name: str
) -> tuple[list[WorldOutcomes], list[str], dict[str, Path], dict]:
    worlds = load_worlds(cache, name)
    if any(w.split != ("B" if name == "sensor-fit" else "T") for w in worlds):
        raise ValueError("split mismatch")
    results, ids, inputs = [], [], {}
    contract = None
    for i, spec in enumerate(worlds):
        path = cache.out_dir / name / f"{spec.world_id}.npz"
        if resume_metadata(path, spec, cache) is None:
            raise ValueError(f"unverified cache: {path}")
        header = header_for(spec, cache)
        current = header.model_dump(mode="json", exclude={"cache_id", "world_id"})
        if contract is not None and contract != current:
            raise ValueError("mixed cache contracts")
        contract = current
        results.append(collect_world(spec, WorldCache.load(path, expected=header), cfg))
        ids.append(header.cache_id)
        inputs[f"cache_{spec.world_id}"] = path
        inputs[f"world_{spec.world_id}"] = cache.worlds_dir / name / f"{spec.world_id}.json"
        print(f"{name}: {i + 1}/{len(worlds)}", flush=True)
    return results, ids, inputs, contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sensor/default.yaml"))
    parser.add_argument("--cache-config", type=Path, default=Path("configs/cache/default.yaml"))
    parser.add_argument("--out", type=Path, default=Path("data/sensor_models"))
    parser.add_argument("--manifest", type=Path, default=Path("registry/manifest.json"))
    parser.add_argument("--evaluate", type=Path, help="frozen model to evaluate, never re-fit")
    args = parser.parse_args()
    cfg, cache = load_config(args.config, SensorConfig), load_config(args.cache_config, CacheConfig)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.evaluate:
        model = SensorModel.load(
            args.evaluate,
            detector_id=cache.detector_id,
            fit_world_set_id=cache.world_set_ids["sensor-fit"],
        )
        if model.fit_config != cfg:
            raise ValueError("evaluation config differs from frozen fit")
        worlds, _, inputs, contract = collect(cache, cfg, "test")
        if contract != model.cache_contract:
            raise ValueError("test cache contract mismatch")
        stats = measurements(worlds, cfg)
        stats.update(
            sensor_model_id=model.sensor_model_id,
            test_world_set_id=cache.world_set_ids["test"],
            calibration=calibration_report(worlds, model),
        )
        report = args.out / f"{model.sensor_model_id}.test.json"
        write_json(report, stats)
        inputs["model"] = args.evaluate
        write_metadata(
            report.with_suffix(".meta.json"),
            stage="evaluate-sensor-models",
            config=cfg,
            inputs=inputs,
            outputs={"report": report},
        )
        return
    worlds, ids, inputs, contract = collect(cache, cfg, "sensor-fit")
    model, stats = fit(worlds, cfg, cache, ids, contract)
    path = args.out / f"{model.sensor_model_id}.json"
    report = path.with_suffix(".fit.json")
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != model.model_dump(
        mode="json"
    ):
        raise ValueError("refuse different existing sensor model")
    model.save(path)
    write_json(report, stats)
    meta = write_metadata(
        path.with_suffix(".meta.json"),
        stage="fit-sensor-models",
        config=cfg,
        inputs=inputs,
        outputs={"model": path, "report": report},
        extra={"sensor_model_id": model.sensor_model_id},
    )
    existing = find(args.manifest, "sensor_model", model.sensor_model_id)
    if existing is not None:
        if existing.sha256 != sha256_file(path):
            raise ValueError("registered model checksum mismatch")
    else:
        register(
            args.manifest,
            Artifact(
                kind="sensor_model",
                artifact_id=model.sensor_model_id,
                sha256=sha256_file(path),
                location=path.as_posix(),
                parents={"detector_id": model.detector_id, "world_set_id": model.fit_world_set_id},
                created_utc=meta["created_utc"],
                notes={"variants": ["V0", "V1", "V2", "V3"], "boundary": cfg.boundary_rule},
            ),
        )
    print(f"Sensor model: {model.sensor_model_id}; {path}", flush=True)


if __name__ == "__main__":
    main()
