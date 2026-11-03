"""Start the frozen M10 transfer runner detached in WSL, with the ROS environment.

Refuses to start a second runner (launch lock, plus the runner's own lock).
Log: data/m10/transfer/run.log; PID: data/m10/transfer/run.pid. A detached job
is not a notification service: check status.json and the log.
"""

import argparse
import shlex
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--stop-after-preflight", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    folder = repo / "data/m10/transfer"
    folder.mkdir(parents=True, exist_ok=True)
    runner = [
        str(Path.home() / ".venvs/altiseek/bin/python"),
        "-u",
        "scripts/deploy/run_m10_transfer.py",
        str(args.protocol),
        *(["--stop-after-preflight"] if args.stop_after_preflight else []),
    ]
    script = (
        "set -eo pipefail; source /opt/ros/jazzy/setup.bash; "
        f"source {Path.home()}/altiseek-m10/ros2_ws/install/setup.bash; "
        f"exec {shlex.join(runner)}"
    )
    lock = Path.home() / "altiseek-m10/transfer-launch.lock"
    with (folder / "run.log").open("a") as log:
        log.write(f"\n=== launch {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===\n")
        log.flush()
        process = subprocess.Popen(
            ["flock", "-n", str(lock), "bash", "-c", script],
            cwd=repo,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    (folder / "run.pid").write_text(f"{process.pid}\n")
    time.sleep(3)
    print(process.pid, "running" if process.poll() is None else f"exited {process.returncode}")


if __name__ == "__main__":
    main()
