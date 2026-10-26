"""Kaggle-only M5 pilot: rebuild libraries/worlds, verify IDs, time two B worlds and validate.

Attach the saved output of aniketkumarpradhan/altiseek-detector-v3 as notebook input.
No API credential is needed. This script does not launch the full 320-world build.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

from altiseek.cache.config import CacheConfig
from altiseek.cache.jobs import load_libraries, load_worlds, write_json
from altiseek.config import load_config
from altiseek.detector.infer import detector_id_of


def stage(script: str, *args: str) -> None:
    command = [sys.executable, f"scripts/{script}.py", *args]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    import torch

    out = Path("/kaggle/working")
    if not out.is_dir() or not torch.cuda.is_available():
        raise SystemExit("Run in Kaggle with Internet on and a GPU accelerator enabled.")
    start = time.perf_counter()
    cfg = load_config("configs/cache/default.yaml", CacheConfig)
    checkpoints = [
        p for p in Path("/kaggle/input").rglob("best.pt") if detector_id_of(p) == cfg.detector_id
    ]
    if not checkpoints:
        raise SystemExit(
            "Add Input: your altiseek-detector-v3 notebook output; best.pt is missing."
        )
    checkpoint = sorted(checkpoints)[0]
    cfg = cfg.model_copy(
        update={
            "checkpoint": checkpoint,
            "libraries": {s: out / "libraries" / s for s in cfg.libraries},
            "worlds_dir": out / "worlds",
            "out_dir": out / "caches",
        }
    )
    # The bootstrap notebook clones a pinned commit. Rebuild the M1 tables there.
    for name in ("prepare_data", "fit_camera", "make_splits"):
        stage(name)
    split = json.loads(Path("data/processed/make_splits.meta.json").read_text())["split_id"]
    if split != "783b4403804291a2":
        raise ValueError(f"split_id mismatch: {split}")
    report = {"split_id": split, "detector_id": cfg.detector_id, "status": "preparing"}
    write_json(out / "pilot-summary.json", report)
    prep_manifest = str(out / "preparation-artifacts.json")
    if not all((p / "library.meta.json").exists() for p in cfg.libraries.values()):
        # Existing partial directories are retained for diagnosis, never silently replaced.
        stage(
            "build_libraries",
            "--workers",
            "2",
            "--manifest",
            prep_manifest,
            f"out_dir={out / 'libraries'}",
        )
    observed = {
        s: json.loads((p / "library.meta.json").read_text())["library_id"]
        for s, p in cfg.libraries.items()
    }
    report["library_ids"] = observed
    report["expected_library_ids"] = cfg.library_ids
    if observed != cfg.library_ids:
        report["status"] = "library_id_mismatch"
        write_json(out / "pilot-summary.json", report)
        raise ValueError(
            "Library IDs differ. Stop here; do not build worlds/caches with these IDs."
        )
    load_libraries(cfg)  # validates index and every image hash
    if not all((cfg.worlds_dir / n / "world_set.meta.json").exists() for n in cfg.world_set_ids):
        stage(
            "generate_worlds",
            "--manifest",
            prep_manifest,
            f"libraries.B={cfg.libraries['B']}",
            f"libraries.T={cfg.libraries['T']}",
            f"out_dir={cfg.worlds_dir}",
        )
    sets = {name: load_worlds(cfg, name) for name in cfg.world_set_ids}
    report["world_set_ids"] = cfg.world_set_ids
    report["preparation_s"] = time.perf_counter() - start
    write_json(out / "pilot-summary.json", report)
    overrides = [
        f"checkpoint={checkpoint}",
        f"libraries.B={cfg.libraries['B']}",
        f"libraries.T={cfg.libraries['T']}",
        f"worlds_dir={cfg.worlds_dir}",
        f"out_dir={cfg.out_dir}",
    ]
    stage("build_cache", "--sets", "sensor-fit", "--limit", "2", *overrides)
    stage("validate_cache", "--sets", "sensor-fit", "--limit", "2", *overrides)
    metrics = [
        json.loads((cfg.out_dir / "sensor-fit" / f"{w.world_id}.meta.json").read_text())
        for w in sets["sensor-fit"][:2]
    ]
    mean_s = sum(m["timing"]["total_s"] for m in metrics) / len(metrics)
    report.update(
        status="pilot_passed",
        worlds=[
            {"world_id": m["world_id"], "timing": m["timing"], "bytes": m["bytes"]} for m in metrics
        ],
        mean_world_s=mean_s,
        projected_cache_hours={n: len(ws) * mean_s / 3600 for n, ws in sets.items()},
        elapsed_s=time.perf_counter() - start,
    )
    write_json(out / "pilot-summary.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
