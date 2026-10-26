"""Build all four cache sets from frozen inputs, resume attached outputs, then validate.

The generation limit is eight hours, with headroom for validation and output saving.
An orderly partial run exits successfully with status partial so its files can be saved.
Identity/checksum failures raise errors; they are never treated as missing work.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from altiseek.cache.config import CacheConfig
from altiseek.cache.jobs import (
    attached_config,
    load_libraries,
    load_worlds,
    resume_metadata,
    write_json,
)
from altiseek.config import load_config
from altiseek.detector.infer import detector_id_of
from altiseek.registry.manifest import Artifact, find, load_manifest, register
from altiseek.world.spec import WorldSpec


def resume_inputs(cfg: CacheConfig, input_root: Path, worlds: dict[str, list[WorldSpec]]) -> int:
    expected = {(name, w.world_id): w for name, specs in worlds.items() for w in specs}
    restored = 0
    for source in sorted(input_root.rglob("*.npz")):
        key = (source.parent.name, source.stem)
        if source.parent.parent.name != "caches" or key not in expected:
            continue
        spec = expected[key]
        meta = resume_metadata(source, spec, cfg)
        if meta is None:
            continue
        target = cfg.out_dir / spec.world_set / source.name
        if target.exists():
            existing = resume_metadata(target, spec, cfg)
            if (
                existing is None
                or existing["outputs"]["cache"]["sha256"] != meta["outputs"]["cache"]["sha256"]
            ):
                raise ValueError(f"conflicting resumed cache: {spec.world_id}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_suffix(".npz.tmp")
            shutil.copyfile(source, temp)
            temp.replace(target)
            shutil.copyfile(source.with_suffix(".meta.json"), target.with_suffix(".meta.json"))
            resume_metadata(target, spec, cfg)
            restored += 1
        registry = cfg.out_dir / "artifacts.json"
        known = find(registry, "cache", meta["cache_id"])
        if known is not None:
            if known.sha256 != meta["outputs"]["cache"]["sha256"]:
                raise ValueError("resumed registry checksum mismatch")
        else:
            register(
                registry,
                Artifact(
                    kind="cache",
                    artifact_id=meta["cache_id"],
                    sha256=meta["outputs"]["cache"]["sha256"],
                    location=f"local:{target.as_posix()}",
                    parents={
                        "world_id": spec.world_id,
                        "detector_id": cfg.detector_id,
                        "world_set_id": cfg.world_set_ids[spec.world_set],
                    },
                    created_utc=meta["created_utc"],
                    notes={"views": meta["views"]},
                ),
            )
    return restored


def completion(cfg: CacheConfig, worlds: dict[str, list[WorldSpec]]) -> dict:
    registry = {a.artifact_id: a for a in load_manifest(cfg.out_dir / "artifacts.json")}
    counts = {}
    for name, specs in worlds.items():
        built = validated = 0
        for spec in specs:
            path = cfg.out_dir / name / f"{spec.world_id}.npz"
            meta = resume_metadata(path, spec, cfg)
            if meta is None:
                continue
            artifact = registry.get(meta["cache_id"])
            if (
                artifact is None
                or artifact.sha256 != meta["outputs"]["cache"]["sha256"]
                or artifact.parents
                != {
                    "world_id": spec.world_id,
                    "detector_id": cfg.detector_id,
                    "world_set_id": cfg.world_set_ids[name],
                }
            ):
                raise ValueError("cache registry missing or mismatched")
            built += 1
            validation = path.with_suffix(".validation.json")
            if not validation.exists():
                continue
            record = json.loads(validation.read_text(encoding="utf-8"))
            if (
                record["passed"] is not True
                or record["cache_id"] != meta["cache_id"]
                or record["inputs"]["cache"]["sha256"] != artifact.sha256
                or record["score_atol"] != 1e-4
                or record["box_atol_px"] != 0.05
                or len(record["view_indices"]) != 20
            ):
                raise ValueError("cache validation missing required checks or mismatched")
            validated += 1
        counts[name] = {"expected": len(specs), "built": built, "validated": validated}
    return counts


def main(argv: Sequence[str] | None = None) -> None:
    import numpy as np
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-seconds", type=float, default=28800)
    args = parser.parse_args(argv)
    if not 0 < args.max_seconds <= 28800:
        parser.error("generation budget must be positive and at most eight hours")
    out = Path("/kaggle/working")
    if not out.is_dir() or not torch.cuda.is_available():
        raise SystemExit("Run on Kaggle with GPU T4 x2 and Internet enabled")
    started = time.perf_counter()
    runtime = {"numpy": np.__version__, "torch": torch.__version__}
    write_json(out / "runtime.json", runtime)
    if np.__version__ != "2.1.3":
        raise ValueError("imported NumPy must be 2.1.3")
    original = load_config("configs/cache/default.yaml", CacheConfig)
    cfg = attached_config(original, Path("/kaggle/input"))
    if cfg is None:
        raise SystemExit("Attach the private altiseek-m5-inputs dataset")
    checkpoints = sorted(
        p for p in Path("/kaggle/input").rglob("best.pt") if detector_id_of(p) == cfg.detector_id
    )
    if not checkpoints:
        raise SystemExit("Attach the altiseek-detector-v3 notebook output")
    cfg = cfg.model_copy(update={"checkpoint": checkpoints[0], "out_dir": out / "caches"})
    load_libraries(cfg)
    worlds = {name: load_worlds(cfg, name) for name in cfg.world_set_ids}
    restored = resume_inputs(cfg, Path("/kaggle/input"), worlds)
    report = {
        "status": "building",
        "runtime": runtime,
        "restored_worlds": restored,
        "generation_budget_s": args.max_seconds,
        "world_set_ids": cfg.world_set_ids,
    }
    write_json(out / "production-summary.json", report)
    overrides = [
        f"checkpoint={cfg.checkpoint}",
        f"libraries.B={cfg.libraries['B']}",
        f"libraries.T={cfg.libraries['T']}",
        f"worlds_dir={cfg.worlds_dir}",
        f"out_dir={cfg.out_dir}",
    ]
    subprocess.run(
        [
            sys.executable,
            "-u",
            "scripts/build_cache.py",
            "--max-seconds",
            str(args.max_seconds),
            *overrides,
        ],
        check=True,
    )
    report["status"] = "validating"
    write_json(out / "production-summary.json", report)
    subprocess.run(
        [sys.executable, "-u", "scripts/validate_cache.py", "--existing-only", *overrides],
        check=True,
    )
    counts = completion(cfg, worlds)
    finished = all(v["validated"] == v["expected"] for v in counts.values())
    report.update(
        status="complete" if finished else "partial",
        sets=counts,
        elapsed_s=time.perf_counter() - started,
    )
    write_json(out / "production-summary.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
