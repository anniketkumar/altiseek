"""Run the frozen M9 extension, with B-only smoke and episode-boundary resume."""

import argparse
from pathlib import Path

from altiseek.eval.extension import run_extension


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("configs/eval/protocol_v2.yaml"))
    parser.add_argument("--manifest", type=Path, default=Path("configs/eval/artifacts_v2.json"))
    parser.add_argument("--out", type=Path, default=Path("runs/eval/m9-v2"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("limit must be nonnegative")
    run_extension(args.protocol, args.manifest, args.out, smoke=args.smoke, limit=args.limit)


if __name__ == "__main__":
    main()
