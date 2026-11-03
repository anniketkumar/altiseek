"""build-cache: GPU batches and atomic per-world checkpoints; --limit 2 for a cost pilot.

For another cloud session, copy completed cache files and their metadata into out_dir
from the previous kernel output. Resume verifies both identity and file SHA-256.
--max-seconds stops between worlds, leaving time for Kaggle to save its output.
"""

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

import cv2

from altiseek.cache.build import build_world, header_for
from altiseek.cache.config import CacheConfig
from altiseek.cache.jobs import (
    gpu_detector,
    load_libraries,
    load_worlds,
    resume_metadata,
    write_json,
)
from altiseek.config import load_config
from altiseek.registry.manifest import Artifact, find, register
from altiseek.registry.run import write_metadata


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/cache/default.yaml"))
    parser.add_argument("--sets", nargs="+", default=["sensor-fit", "rl-train", "rl-val", "test"])
    parser.add_argument("--limit", type=int, default=None, help="max worlds per set (pilot only)")
    parser.add_argument("--max-seconds", type=float, default=36000)
    parser.add_argument("--device", default="0")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.max_seconds <= 0:
        parser.error("--max-seconds must be positive")
    cfg = load_config(args.config, CacheConfig, args.overrides)
    cv2.setNumThreads(1)
    libraries = load_libraries(cfg)
    sets = {name: load_worlds(cfg, name) for name in args.sets}
    detector = gpu_detector(cfg, args.device)
    started = time.perf_counter()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    registry = cfg.out_dir / "artifacts.json"
    for name, worlds in sets.items():
        root = cfg.out_dir / name
        root.mkdir(parents=True, exist_ok=True)
        # Reconstruct from verified files, never trust a stale completion manifest.
        # Keep completed worlds outside a new --limit, too.
        prior = {w.world_id: resume_metadata(root / f"{w.world_id}.npz", w, cfg) for w in worlds}
        completed = {
            wid: {"cache_id": m["cache_id"], "sha256": m["outputs"]["cache"]["sha256"]}
            for wid, m in prior.items()
            if m is not None
        }
        for spec in worlds[: args.limit]:
            path = root / f"{spec.world_id}.npz"
            meta_path = path.with_suffix(".meta.json")
            header = header_for(spec, cfg)
            meta = prior[spec.world_id]
            if meta is not None:
                print(f"resume {name}/{spec.world_id}", flush=True)
            else:
                if time.perf_counter() - started >= args.max_seconds:
                    print(
                        "time budget reached between worlds; resume from saved output", flush=True
                    )
                    return
                cache, timing = build_world(spec, libraries[spec.split], cfg, detector)
                cache.save(path)
                meta = write_metadata(
                    meta_path,
                    stage="build-cache",
                    config=cfg,
                    inputs={
                        "world": cfg.worlds_dir / name / f"{spec.world_id}.json",
                        "checkpoint": cfg.checkpoint,
                    },
                    outputs={"cache": path},
                    extra={
                        "world_id": spec.world_id,
                        "world_set_id": cfg.world_set_ids[name],
                        "detector_id": cfg.detector_id,
                        "cache_id": header.cache_id,
                        "views": len(cache.views),
                        "detections": len(cache.detections),
                        "timing": timing,
                        "bytes": path.stat().st_size,
                    },
                )
                print(
                    json.dumps(
                        {
                            "set": name,
                            "world": spec.world_id,
                            **timing,
                            "bytes": path.stat().st_size,
                        }
                    ),
                    flush=True,
                )
            digest = meta["outputs"]["cache"]["sha256"]
            existing = find(registry, "cache", header.cache_id)
            if existing is not None:
                if existing.sha256 != digest:
                    raise ValueError("registered cache checksum mismatch")
            else:
                register(
                    registry,
                    Artifact(
                        kind="cache",
                        artifact_id=header.cache_id,
                        sha256=digest,
                        location=f"local:{path.as_posix()}",
                        parents={
                            "world_id": spec.world_id,
                            "detector_id": cfg.detector_id,
                            "world_set_id": cfg.world_set_ids[name],
                        },
                        created_utc=meta["created_utc"],
                        notes={"views": meta["views"]},
                    ),
                )
            completed[spec.world_id] = {"cache_id": header.cache_id, "sha256": digest}
            write_json(
                root / "completed.json",
                {
                    "world_set_id": cfg.world_set_ids[name],
                    "expected_worlds": len(worlds),
                    "complete": len(completed) == len(worlds),
                    "worlds": completed,
                },
            )


if __name__ == "__main__":
    main()
