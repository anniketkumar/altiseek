"""validate-cache: seeded live GPU checks, with tolerances fixed before seeing test output."""

import argparse
from collections.abc import Sequence
from pathlib import Path

import cv2

from altiseek.cache.build import header_for
from altiseek.cache.config import CacheConfig
from altiseek.cache.jobs import gpu_detector, load_libraries, load_worlds, resume_metadata
from altiseek.cache.storage import WorldCache
from altiseek.cache.validate import validate_world
from altiseek.config import load_config
from altiseek.registry.run import write_metadata


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/cache/default.yaml"))
    parser.add_argument("--sets", nargs="+", default=["sensor-fit", "rl-train", "rl-val", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--samples", type=int, default=20, help="random views per world")
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--score-atol", type=float, default=1e-4)
    parser.add_argument("--box-atol-px", type=float, default=0.05)
    parser.add_argument("--device", default="0")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.samples < 1 or args.seed < 0 or args.score_atol < 0 or args.box_atol_px < 0:
        parser.error("samples must be positive; seed and tolerances must be nonnegative")
    cfg = load_config(args.config, CacheConfig, args.overrides)
    cv2.setNumThreads(1)
    libraries = load_libraries(cfg)
    detector = gpu_detector(cfg, args.device)
    for name in args.sets:
        for spec in load_worlds(cfg, name)[: args.limit]:
            path = cfg.out_dir / name / f"{spec.world_id}.npz"
            if resume_metadata(path, spec, cfg) is None:
                raise ValueError(f"cache or metadata is missing: {path}")
            cache = WorldCache.load(path, expected=header_for(spec, cfg))
            seed = args.seed + spec.seed
            indices = validate_world(
                cache,
                spec,
                libraries[spec.split],
                cfg,
                detector,
                seed=seed,
                samples=args.samples,
                score_atol=args.score_atol,
                box_atol_px=args.box_atol_px,
            )
            write_metadata(
                path.with_suffix(".validation.json"),
                stage="validate-cache",
                config=cfg,
                inputs={"cache": path, "checkpoint": cfg.checkpoint},
                outputs={},
                extra={
                    "cache_id": cache.header.cache_id,
                    "seed": seed,
                    "view_indices": indices,
                    "score_atol": args.score_atol,
                    "box_atol_px": args.box_atol_px,
                    "passed": True,
                },
            )
            print(f"validated {name}/{spec.world_id}: {len(indices)} views", flush=True)


if __name__ == "__main__":
    main()
