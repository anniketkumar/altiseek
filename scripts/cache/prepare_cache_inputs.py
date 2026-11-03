"""Package verified original libraries and worlds for a private Kaggle input (D32).

Run locally. The ZIP preserves every artifact byte and contains no checkpoint, raw
source frame, credential or git metadata. It is written under gitignored data/.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from altiseek.cache.config import CacheConfig
from altiseek.cache.jobs import load_libraries, load_worlds
from altiseek.config import load_config
from altiseek.registry.run import sha256_file, write_metadata


def package(cfg: CacheConfig, out: Path) -> None:
    libraries = load_libraries(cfg)
    worlds = {n: load_worlds(cfg, n) for n in cfg.world_set_ids}
    files: dict[str, Path] = {}
    for split, lib in libraries.items():
        meta = json.loads((lib.path / "library.meta.json").read_text(encoding="utf-8"))
        if meta["split_id"] != "783b4403804291a2":
            raise ValueError("library split_id mismatch")
        names = ["backgrounds.csv", "objects.csv", "conditions.json", "library.meta.json"]
        names += [f"backgrounds/{r.patch_id}.png" for r in lib.backgrounds]
        names += [f"objects/{r.crop_id}.png" for r in lib.objects]
        files.update({f"libraries/{split}/{n}": lib.path / n for n in names})
    for name, specs in worlds.items():
        names = ["worlds.csv", "world_set.meta.json", *(f"{w.world_id}.json" for w in specs)]
        files.update({f"worlds/{name}/{n}": cfg.worlds_dir / name / n for n in names})
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_suffix(".zip.tmp")
    with ZipFile(temp, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "m5-inputs.json",
            json.dumps(
                {
                    "split_id": "783b4403804291a2",
                    "library_ids": cfg.library_ids,
                    "world_set_ids": cfg.world_set_ids,
                }
            ),
        )
        for name, source in sorted(files.items()):
            archive.write(source, name)
    # Verify actual packaged bytes before declaring the bundle ready.
    with ZipFile(temp) as archive:
        for name, source in files.items():
            if archive.read(name) != source.read_bytes():
                raise ValueError(f"packaged bytes mismatch: {name}")
    temp.replace(out)
    write_metadata(
        out.with_suffix(".meta.json"),
        stage="prepare-cache-inputs",
        config=cfg,
        inputs={s: lib.path / "library.meta.json" for s, lib in libraries.items()},
        outputs={"bundle": out},
        extra={
            "files": len(files) + 1,
            "library_ids": cfg.library_ids,
            "world_set_ids": cfg.world_set_ids,
        },
    )
    print(
        json.dumps(
            {
                "path": str(out),
                "bytes": out.stat().st_size,
                "sha256": sha256_file(out),
                "files": len(files) + 1,
            }
        ),
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/kaggle/altiseek-m5-inputs.zip"))
    args = parser.parse_args(argv)
    package(load_config("configs/cache/default.yaml", CacheConfig), args.out)


if __name__ == "__main__":
    main()
