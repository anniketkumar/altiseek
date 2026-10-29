"""Monitor the existing cloud job, download output, verify caches and fit B sensors.

Never submits a kernel or starts held-out evaluation. Partial jobs stop for review.
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from filelock import FileLock

from altiseek.cache.jobs import write_json


def main() -> None:
    root = Path("data/kaggle/altiseek-m9-cache")
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "watch-status.json"
    cli = shutil.which("kaggle") or str(Path.home() / ".local/bin/kaggle")
    kernel = "aniketkumarpradhan/altiseek-m9-cache"
    with FileLock(root / "watch.lock", timeout=0):
        while True:
            result = subprocess.run(
                [cli, "kernels", "status", kernel], capture_output=True, text=True, check=True
            )
            status = result.stdout.strip()
            write_json(status_path, {"state": "monitoring", "cloud": status})
            print(status, flush=True)
            if any(word in status.upper() for word in ("COMPLETE", "ERROR", "CANCEL")):
                break
            time.sleep(60)
        subprocess.run([cli, "kernels", "output", kernel, "-p", str(root / "output")], check=True)
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
        for case in ("detector", "camera"):
            if collection["cases"][case].get("complete") is not True:
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
