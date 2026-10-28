"""Evaluate frozen M9 policies; --smoke uses B only, --limit pauses resumably."""

import argparse
from pathlib import Path

from altiseek.eval.runner import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("configs/eval/protocol_v1.yaml"))
    parser.add_argument("--artifacts", type=Path, default=Path("configs/eval/artifacts_v1.json"))
    parser.add_argument("--out", type=Path, default=Path("runs/eval/m9-v1"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="new episodes before a clean pause")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    run(args.protocol, args.artifacts, args.out, smoke=args.smoke, limit=args.limit)


if __name__ == "__main__":
    main()
