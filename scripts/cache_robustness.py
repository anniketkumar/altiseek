"""Bounded, resumable D46 GPU cache pilot and production on Kaggle."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from cache_production import completion, resume_inputs

from altiseek.cache.config import CacheConfig
from altiseek.cache.jobs import attached_config, load_libraries, load_worlds, write_json
from altiseek.cache.robustness import CASES, case_config
from altiseek.config import load_config
from altiseek.detector.infer import detector_id_of


def stage(script: str, config: Path, sets: list[str], extra: list[str]) -> None:
    subprocess.run(
        [
            sys.executable,
            "-u",
            f"scripts/{script}.py",
            "--config",
            str(config),
            *extra,
            "--sets",
            *sets,
        ],
        check=True,
    )


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-seconds", type=float, default=19000)
    parser.add_argument("--pilot-only", action="store_true")
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 19000:
        parser.error("budget must be positive and at most 19000 seconds")
    if not torch.cuda.is_available() or not Path("/kaggle/working").is_dir():
        raise SystemExit("Requires a Kaggle GPU session")
    started = time.perf_counter()
    inputs = Path("/kaggle/input")
    out = Path("/kaggle/working/m9-cache")
    original = load_config("configs/cache/default.yaml", CacheConfig)
    attached = attached_config(original, inputs)
    if attached is None:
        raise ValueError("attach the original private M5 dataset")
    load_libraries(attached)
    subprocess.run(
        [
            sys.executable,
            "scripts/generate_stress_worlds.py",
            "--library",
            str(attached.libraries["B"]),
        ],
        check=True,
    )
    checkpoints = {detector_id_of(p): p for p in inputs.rglob("best.pt")}
    configs = {}
    for case in CASES:
        cfg = case_config(attached, case, out)
        if cfg.detector_id not in checkpoints:
            raise ValueError(f"missing checkpoint {cfg.detector_id}")
        cfg = cfg.model_copy(update={"checkpoint": checkpoints[cfg.detector_id]})
        worlds = {name: load_worlds(cfg, name) for name in cfg.world_set_ids}
        # Restore only this case; detector/camera cases share world IDs.
        for marker in inputs.rglob("robustness-summary.json"):
            source = marker.parent / case
            if source.is_dir():
                resume_inputs(cfg, source, worlds)
        config_path = out / case / "config.json"
        write_json(config_path, cfg.model_dump(mode="json"))
        configs[case] = (cfg, config_path, worlds)
    report = {"state": "pilot", "cases": {}, "budget_s": args.max_seconds}
    report_path = out / "robustness-summary.json"
    write_json(report_path, report)
    # One B and one T world per swap, one low/high stress pair; same cache artifacts
    # continue into production only after all six live validation checks pass.
    for case, (_cfg, path, worlds) in configs.items():
        stage("build_cache", path, list(worlds), ["--limit", "1"])
        stage("validate_cache", path, list(worlds), ["--limit", "1"])
        report["cases"][case] = {"pilot_passed": True}
        write_json(report_path, report)
    if not args.pilot_only:
        for case, (_cfg, path, worlds) in configs.items():
            remaining = args.max_seconds - (time.perf_counter() - started)
            if remaining <= 0:
                break
            report["state"] = f"building-{case}"
            write_json(report_path, report)
            stage("build_cache", path, list(worlds), ["--max-seconds", str(remaining)])
            report["state"] = f"validating-{case}"
            write_json(report_path, report)
            stage("validate_cache", path, list(worlds), ["--existing-only"])
    for case, (cfg, _, worlds) in configs.items():
        report["cases"][case]["sets"] = completion(cfg, worlds)
    done = all(
        s["validated"] == s["expected"]
        for c in report["cases"].values()
        for s in c["sets"].values()
    )
    report.update(state="complete" if done else "partial", elapsed_s=time.perf_counter() - started)
    write_json(report_path, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
