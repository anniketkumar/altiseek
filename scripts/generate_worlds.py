"""generate-worlds: world specs for every world set and scenario family (CLAUDE.md §11, §16).

Each world set draws on one split's library (``scripts/build_libraries.py``). Every family of the
main grid gets ``per_family`` worlds, with seeds ``seed_base + 10_000 x family index + i``.
Writes, per world set,

    <out_dir>/<set>/<world_id>.json, worlds.csv, world_set.meta.json

and registers the set in the artifact manifest. Its ID is the first 16 hex digits of the sha256
of ``worlds.csv``. Run from the repository root:

    uv run python scripts/generate_worlds.py [--config configs/world/default.yaml] [--overwrite]
"""

import argparse
import hashlib
import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from altiseek.config import load_config
from altiseek.data.tables import write_csv
from altiseek.registry.manifest import Artifact, register
from altiseek.registry.run import git_state, write_metadata
from altiseek.world.config import WorldConfig
from altiseek.world.library import Library
from altiseek.world.sampler import sample_world


@dataclass(frozen=True)
class WorldRow:
    """One world, a row of ``worlds.csv``."""

    world_id: str
    seed: int
    family: str
    condition: int
    targets: int
    distractors: int


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate world specs.")
    parser.add_argument("--config", type=Path, default=Path("configs/world/default.yaml"))
    parser.add_argument("--manifest", type=Path, default=Path("registry/manifest.json"))
    parser.add_argument("--overwrite", action="store_true", help="replace existing world sets")
    parser.add_argument("overrides", nargs="*", help="config overrides, dotted.key=value")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, WorldConfig, args.overrides)
    libraries = {split: Library.load(path) for split, path in cfg.libraries.items()}
    families = cfg.grid.families()
    commit = (git_state() or {}).get("commit", "")
    for world_set in cfg.world_sets:
        out = cfg.out_dir / world_set.name
        if out.exists():
            if not args.overwrite:
                raise SystemExit(f"{out} exists; pass --overwrite to regenerate it")
            shutil.rmtree(out)
        out.mkdir(parents=True)
        library = libraries[world_set.split]
        rows = []
        for f, family in enumerate(families):
            for i in range(world_set.per_family):
                seed = world_set.seed_base + 10_000 * f + i
                spec = sample_world(seed, family, world_set, cfg, library)
                spec.save(out / f"{spec.world_id}.json")
                rows.append(
                    WorldRow(
                        spec.world_id,
                        seed,
                        spec.family,
                        spec.condition,
                        len(spec.targets),
                        len(spec.objects) - len(spec.targets),
                    )
                )
        write_csv(out / "worlds.csv", WorldRow, rows)
        digest = hashlib.sha256((out / "worlds.csv").read_bytes()).hexdigest()
        summary = {
            "world_set": world_set.name,
            "world_set_id": digest[:16],
            "split": world_set.split,
            "library_id": library.library_id,
            "worlds": len(rows),
            "families": len(families),
            "targets": sum(r.targets for r in rows),
            "distractors": sum(r.distractors for r in rows),
        }
        write_metadata(
            out / "world_set.meta.json",
            stage="generate-worlds",
            config=cfg,
            inputs={"library": library.path / "library.meta.json"},
            outputs={"worlds": out / "worlds.csv"},
            extra=summary,
        )
        register(
            args.manifest,
            Artifact(
                kind="world_set",
                artifact_id=digest[:16],
                sha256=digest,
                location=f"local:{out.as_posix()}",
                parents={"library_id": library.library_id, "code_commit": commit},
                created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
                notes=summary,
            ),
        )
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
