"""Kaggle-only M5 pilot: rebuild libraries/worlds, verify IDs, time two B worlds and validate.

Attach the saved output of aniketkumarpradhan/altiseek-detector-v3 as notebook input.
No API credential is needed. This script does not launch the full 320-world build.
"""

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from altiseek.cache.config import CacheConfig
from altiseek.cache.jobs import load_libraries, load_worlds, write_json
from altiseek.config import load_config
from altiseek.detector.infer import detector_id_of


def stage(script: str, *args: str) -> None:
    command = [sys.executable, f"scripts/{script}.py", *args]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def attached_config(cfg: CacheConfig, input_root: Path) -> CacheConfig | None:
    """Use byte-preserved inputs only when their manifest matches the frozen IDs."""
    markers = sorted(input_root.rglob("m5-inputs.json"))
    if not markers:
        return None
    if len(markers) != 1:
        raise ValueError("Attach exactly one M5 input bundle")
    marker = markers[0]
    manifest = json.loads(marker.read_text(encoding="utf-8"))
    expected = {
        "split_id": "783b4403804291a2",
        "library_ids": cfg.library_ids,
        "world_set_ids": cfg.world_set_ids,
    }
    if manifest != expected:
        raise ValueError("M5 input manifest does not match the frozen IDs")
    root = marker.parent
    candidate = cfg.model_copy(
        update={
            "libraries": {s: root / "libraries" / s for s in cfg.libraries},
            "worlds_dir": root / "worlds",
        }
    )
    for split, path in candidate.libraries.items():
        meta = json.loads((path / "library.meta.json").read_text(encoding="utf-8"))
        if meta["split_id"] != expected["split_id"] or meta["split"] != split:
            raise ValueError("attached library split mismatch")
    return candidate


def main(argv: Sequence[str] | None = None) -> None:
    import numpy as np
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-input-bundle", action="store_true")
    args = parser.parse_args(argv)
    out = Path("/kaggle/working")
    if not out.is_dir() or not torch.cuda.is_available():
        raise SystemExit("Run in Kaggle with Internet on and a GPU accelerator enabled.")
    start = time.perf_counter()
    runtime = {"numpy": np.__version__, "numpy_file": np.__file__, "torch": torch.__version__}
    write_json(out / "runtime.json", runtime)
    if np.__version__ != "2.1.3":
        raise ValueError(f"Imported NumPy {np.__version__} != pinned 2.1.3; inspect runtime.json")
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
    attached = attached_config(cfg, Path("/kaggle/input"))
    if args.require_input_bundle and attached is None:
        raise SystemExit("Add the private M5 input dataset containing m5-inputs.json first.")
    if attached is not None:
        cfg = attached
    # The bootstrap notebook clones a pinned commit. Rebuild the M1 tables there.
    split = "783b4403804291a2"
    if attached is None:
        for name in ("prepare_data", "fit_camera", "make_splits"):
            stage(name)
        split = json.loads(Path("data/processed/make_splits.meta.json").read_text())["split_id"]
    if split != "783b4403804291a2":
        raise ValueError(f"split_id mismatch: {split}")
    report = {
        "split_id": split,
        "detector_id": cfg.detector_id,
        "status": "preparing",
        "runtime": runtime,
        "input_mode": "attached" if attached else "rebuilt",
    }
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
