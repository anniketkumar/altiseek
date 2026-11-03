"""Import a complete saved Kaggle cache output ZIP after strict verification."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from altiseek.cache.collect import collect_archive
from altiseek.cache.config import CacheConfig
from altiseek.config import load_config


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/cache/default.yaml"))
    parser.add_argument("--manifest", type=Path, default=Path("registry/manifest.json"))
    args = parser.parse_args(argv)
    report = collect_archive(args.archive, load_config(args.config, CacheConfig), args.manifest)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
