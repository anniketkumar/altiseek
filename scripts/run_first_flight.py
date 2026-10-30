"""Own and clean up one local, headless PX4/Gazebo first-flight check."""

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    stack = Path.home() / "altiseek-m10"
    with (stack / "first-flight.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = stack / "flights" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        run.mkdir(parents=True, exist_ok=False)
        pointer = repo / "data/m10/first-flight-path.txt"
        pointer.write_text(str(run) + "\n", encoding="utf-8")
        env = os.environ.copy()
        env.update(
            ROS_DOMAIN_ID="49",
            ROS_LOCALHOST_ONLY="1",
            GZ_PARTITION="altiseek-first-flight",
            GZ_IP="127.0.0.1",
            HEADLESS="1",
            PX4_SIM_MODEL="gz_x500",
            PX4_UXRCE_DDS_PORT="8899",
            PX4_PARAM_COM_RC_IN_MODE="4",
            PX4_PARAM_COM_RCL_EXCEPT="4",
            PYTHONPATH=str(repo / "src") + os.pathsep + env.get("PYTHONPATH", ""),
            LD_LIBRARY_PATH=str(stack / "install/lib")
            + os.pathsep
            + env.get("LD_LIBRARY_PATH", ""),
        )
        processes = []
        logs = []
        result = {"state": "starting", "run": str(run)}
        status = run / "status.json"

        def save() -> None:
            status.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

        def start(name: str, command: list[str], cwd: Path) -> subprocess.Popen:
            log = (run / f"{name}.log").open("w")
            logs.append(log)
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
            processes.append(process)
            return process

        save()
        try:
            start("agent", [str(stack / "install/bin/MicroXRCEAgent"), "udp4", "-p", "8899"], run)
            build = stack / "PX4-Autopilot/build/px4_sitl_default"
            # PX4 startup uses this local environment file to find bundled assets.
            (run / "gz_env.sh").write_bytes((build / "rootfs/gz_env.sh").read_bytes())
            px4 = start("px4", [str(build / "bin/px4"), "-d", str(build / "etc")], run)
            start(
                "gcs",
                [
                    str(Path.home() / ".venvs/px4-build/bin/python"),
                    str(repo / "scripts/first_flight_gcs.py"),
                ],
                repo,
            )
            # PX4 links its etc/ to the provided data directory; working data
            # and logs stay in this new run directory.
            result["state"] = "flight"
            save()
            controller = start(
                "controller",
                [sys.executable, str(repo / "scripts/first_flight.py"), str(run)],
                repo,
            )
            deadline = time.monotonic() + 205
            while controller.poll() is None:
                if px4.poll() is not None:
                    raise RuntimeError(f"PX4 exited during flight: {px4.returncode}")
                if time.monotonic() > deadline:
                    raise TimeoutError("First-flight controller exceeded its deadline")
                time.sleep(0.5)
            code = controller.returncode
            result.update(state="passed" if code == 0 else "failed", controller_exit=code)
            result["px4_exit_before_cleanup"] = px4.poll()
        except Exception as error:
            result.update(state="failed", error=repr(error))
            raise
        finally:
            for process in reversed(processes):
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
            for process in processes:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
            for log in logs:
                log.close()
            result["owned_processes_stopped"] = True
            save()
        print(json.dumps(result), flush=True)
        if result["state"] != "passed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
