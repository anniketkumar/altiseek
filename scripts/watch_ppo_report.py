"""Wait for the existing PPO suite, then report it; never starts or restarts training."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from altiseek.cache.jobs import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/ppo/m8-initial"))
    parser.add_argument("--hours", type=float, default=12)
    args = parser.parse_args()
    deadline = time.monotonic() + args.hours * 3600
    while time.monotonic() < deadline:
        status = json.loads((args.root / "status.json").read_text(encoding="utf-8"))
        if any(row["status"] == "failed" for row in status["runs"]):
            write_json(args.root / "report-status.json", {"status": "training-failed"})
            raise RuntimeError("suite failed; report withheld, inspect seed logs")
        if status["complete"]:
            result = subprocess.run(
                [sys.executable, "scripts/report_ppo.py", str(args.root)], check=False
            )
            write_json(
                args.root / "report-status.json",
                {
                    "status": "complete" if result.returncode == 0 else "report-failed",
                    "returncode": result.returncode,
                },
            )
            if result.returncode:
                raise RuntimeError("report failed; inspect report.log")
            return
        time.sleep(30)
    write_json(args.root / "report-status.json", {"status": "timed-out"})
    raise TimeoutError("suite still incomplete; training was not modified")


if __name__ == "__main__":
    main()
