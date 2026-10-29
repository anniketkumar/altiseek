"""Import completed D46 cases through the original strict M5 archive verifier.

The downloaded cloud summary is preserved. Each complete case is repackaged in
the original archive layout; no cache, metadata or validation bytes are changed.
Partial cases stay available for cloud resumption and are not imported.
"""

import argparse
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from altiseek.cache.collect import collect_archive
from altiseek.cache.config import CacheConfig
from altiseek.cache.jobs import write_json
from altiseek.cache.robustness import CASES, case_config
from altiseek.config import load_config
from altiseek.registry.run import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="downloaded m9-cache directory")
    parser.add_argument("--out", type=Path, default=Path("data/m9-robustness"))
    args = parser.parse_args()
    summary_path = args.source / "robustness-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    original = load_config("configs/cache/default.yaml", CacheConfig)
    results = {}
    for case in CASES:
        record = summary["cases"][case]
        if not record["pilot_passed"] or any(
            s["expected"] != s["built"] or s["expected"] != s["validated"]
            for s in record["sets"].values()
        ):
            results[case] = {"state": "partial_not_imported"}
            continue
        cfg = case_config(original, case, args.out)
        root = args.out / case
        root.mkdir(parents=True, exist_ok=True)
        archive_path = root / "cloud-case.zip"
        with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
            archive.writestr(
                "production-summary.json",
                json.dumps(
                    {
                        "status": "complete",
                        "world_set_ids": cfg.world_set_ids,
                        "sets": record["sets"],
                        "elapsed_s": summary["elapsed_s"],
                        "source_summary_sha256": sha256_file(summary_path),
                    }
                ),
            )
            for path in sorted((args.source / case / "caches").rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(args.source / case).as_posix())
        results[case] = collect_archive(archive_path, cfg, root / "artifacts.json")
        write_json(root / "config.json", cfg.model_dump(mode="json"))
    write_json(
        args.out / "collection.json",
        {
            "source_summary_sha256": sha256_file(summary_path),
            "cases": results,
        },
    )
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
