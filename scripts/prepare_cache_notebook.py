"""Write a credential-free Kaggle pilot notebook pinned to a pushed commit."""

import argparse
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path


def notebook(commit: str) -> dict:
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("expected a full commit SHA")
    code = f"""import subprocess
import sys
from pathlib import Path

commit = {commit!r}
repo = Path("/tmp/altiseek-cache-pilot")
if not repo.exists():
    subprocess.run(["git", "clone", "--quiet", "https://github.com/anniketkumar/altiseek.git",
                    str(repo)], check=True)
subprocess.run(["git", "checkout", "--quiet", commit], cwd=repo, check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-input", "-e", "."],
               cwd=repo, check=True)
log_path = Path("/kaggle/working/cache-pilot.log")
with log_path.open("w") as log:
    process = subprocess.Popen([sys.executable, "-u", "scripts/cache_pilot.py",
                                "--require-input-bundle"], cwd=repo,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    status = process.wait()
if status:
    raise RuntimeError("Pilot stopped. Download pilot-summary.json and cache-pilot.log; "
                       "do not change IDs.")
"""
    intro = """# AltiSeek M5 cache pilot

Keep this notebook **private**. Enable **Internet** and **GPU T4 x2**.
Use **Add Input** to attach the saved output of your `altiseek-detector-v3` notebook.
Also attach the **private** dataset created from `altiseek-m5-inputs.zip`.
Its input tree must contain `m5-inputs.json`, `libraries/`, and `worlds/`.
Then use **Save Version → Save & Run All** for a background run that saves outputs.

This runs a two-world sensor-fit pilot, not the full cache build. It verifies the original
library/world bytes and IDs, and skips rebuilding them. No API token is needed.
Download `pilot-summary.json` and `cache-pilot.log` from the saved output afterward.
"""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": [
            {
                "cell_type": "markdown",
                "id": "instructions",
                "metadata": {},
                "source": intro.splitlines(keepends=True),
            },
            {
                "cell_type": "code",
                "id": "pilot",
                "metadata": {},
                "source": code.splitlines(keepends=True),
                "execution_count": None,
                "outputs": [],
            },
        ],
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/kaggle/altiseek-cache-pilot.ipynb"))
    args = parser.parse_args(argv)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    remote = subprocess.check_output(["git", "branch", "-r", "--contains", commit], text=True)
    if not remote.strip():
        raise SystemExit("Push the tested commit before preparing the pilot notebook.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(notebook(commit), indent=2) + "\n", encoding="utf-8")
    print(f"{args.out}: pinned to {commit}")


if __name__ == "__main__":
    main()
