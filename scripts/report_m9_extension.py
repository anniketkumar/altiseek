"""Rebuild M9 extension tables and figures from verified saved episodes."""

import argparse
from pathlib import Path

from altiseek.eval.extension_report import report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, default=Path("docs/m9-extension-results"))
    args = parser.parse_args()
    report(args.run, args.out)


if __name__ == "__main__":
    main()
