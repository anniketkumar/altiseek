"""Monitor the existing cloud job, download output, verify caches and fit B sensors.

Never submits a kernel or starts held-out evaluation. Partial jobs stop for review.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from filelock import FileLock

from altiseek.cache.jobs import write_json


def network_command(command: list[str], status_path: Path) -> str:
    """Wait through network/API failures without interrupting the remote job."""
    failures = 0
    while True:
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                return result.stdout.strip()
            code = result.returncode
        except OSError:
            code = None
        failures += 1
        write_json(
            status_path,
            {
                "state": "waiting_for_kaggle",
                "failed_attempts": failures,
                "returncode": code,
            },
        )
        print(f"Kaggle request failed (attempt {failures}); retrying in 60 s", flush=True)
        time.sleep(60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", default="altiseek-m9-cache")
    args = parser.parse_args()
    root = Path("data/kaggle") / args.slug
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "watch-status.json"
    cli = shutil.which("kaggle") or str(Path.home() / ".local/bin/kaggle")
    kernel = f"aniketkumarpradhan/{args.slug}"
    with FileLock(root / "watch.lock", timeout=0):
        while True:
            status = network_command([cli, "kernels", "status", kernel], status_path)
            write_json(status_path, {"state": "monitoring", "cloud": status})
            print(status, flush=True)
            if any(word in status.upper() for word in ("COMPLETE", "ERROR", "CANCEL")):
                break
            time.sleep(60)
        network_command([cli, "kernels", "output", kernel, "-p", str(root / "output")], status_path)
        if "COMPLETE" not in status.upper():
            write_json(status_path, {"state": "cloud_failed", "cloud": status})
            return
        markers = list((root / "output").rglob("robustness-summary.json"))
        if len(markers) != 1:
            raise ValueError("expected one cloud summary")
        subprocess.run(
            [sys.executable, "scripts/collect_robustness.py", str(markers[0].parent)], check=True
        )
        collection = json.loads(Path("data/m9-robustness/collection.json").read_text())
        imported = json.loads(markers[0].read_text())["cases"]
        for case in ("detector", "camera"):
            if case not in imported or collection["cases"][case].get("complete") is not True:
                continue
            target = Path("data/m9-robustness") / case
            write_json(status_path, {"state": "fitting_B_sensor", "case": case})
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "scripts/fit_sensor_models.py",
                    "--cache-config",
                    str(target / "config.json"),
                    "--out",
                    str(target / "sensor_models"),
                    "--manifest",
                    str(target / "sensor-artifacts.json"),
                ],
                check=True,
            )
        complete = all(c.get("complete") is True for c in collection["cases"].values())
        write_json(
            status_path,
            {
                "state": "awaiting_review" if complete else "partial",
                "held_out_evaluation_started": False,
            },
        )


if __name__ == "__main__":
    main()
