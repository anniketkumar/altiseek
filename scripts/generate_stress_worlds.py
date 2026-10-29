"""Generate the frozen D46 B stress pairs; resume only byte-identical specs."""

import argparse
import hashlib
import json
from pathlib import Path

from generate_worlds import WorldRow

from altiseek.cache.jobs import write_json
from altiseek.config import load_config
from altiseek.data.tables import write_csv
from altiseek.registry.run import write_metadata
from altiseek.world.config import WorldConfig
from altiseek.world.library import Library
from altiseek.world.spec import WorldSpec
from altiseek.world.stress import stress_pair


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=Path("data/libraries/B"))
    parser.add_argument("--out", type=Path, default=Path("data/m9-stress/worlds"))
    args = parser.parse_args()
    cfg = load_config("configs/world/default.yaml", WorldConfig)
    library = Library.load(args.library)
    root = args.out
    rows = {name: [] for name in ("stress-low", "stress-high")}
    pairs = []
    families = [(p, t) for p in cfg.grid.prior_sigma_m for t in cfg.grid.targets]
    for f, family in enumerate(families):
        for i in range(10):
            seed = 5000000 + 10000 * f + i
            low, high = stress_pair(seed, family, cfg, library)
            pairs.append(
                {
                    "seed": seed,
                    "family": "-".join(family),
                    "low": low.world_id,
                    "high": high.world_id,
                }
            )
            for spec in (low, high):
                out = root / spec.world_set
                out.mkdir(parents=True, exist_ok=True)
                path = out / f"{spec.world_id}.json"
                if path.exists():
                    if WorldSpec.load(path) != spec:
                        raise ValueError("existing stress world differs")
                else:
                    spec.save(path)
                rows[spec.world_set].append(
                    WorldRow(
                        spec.world_id,
                        seed,
                        spec.family,
                        spec.condition,
                        len(spec.targets),
                        len(spec.objects) - len(spec.targets),
                    )
                )
    ids = {}
    for name, records in rows.items():
        out = root / name
        write_csv(out / "worlds.csv", WorldRow, records)
        ids[name] = hashlib.sha256((out / "worlds.csv").read_bytes()).hexdigest()[:16]
        write_metadata(
            out / "world_set.meta.json",
            stage="generate-stress-worlds",
            config=cfg,
            inputs={
                "library": library.path / "library.meta.json",
                "sampler": Path("src/altiseek/world/stress.py"),
            },
            outputs={"worlds": out / "worlds.csv"},
            extra={
                "world_set": name,
                "world_set_id": ids[name],
                "split": "B",
                "library_id": library.library_id,
                "worlds": len(records),
                "distractor_ranges": {"low": [0, 3], "high": [30, 40]},
            },
        )
    write_json(root / "pairs.json", {"world_set_ids": ids, "pairs": pairs})
    print(json.dumps(ids), flush=True)


if __name__ == "__main__":
    main()
