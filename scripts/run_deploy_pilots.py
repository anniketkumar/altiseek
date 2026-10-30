"""Detached/resumable B integration of all three frozen PPO seeds; stop on failure."""

import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from altiseek.registry.run import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--world-export", type=Path, default=Path.home() / "altiseek-m10/worlds/003b0c54f581c65b"
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    paths = [
        *sorted((repo / "src/altiseek").rglob("*.py")),
        *sorted((repo / "scripts").glob("*flight*.py")),
        Path(__file__).resolve(),
    ]
    definition = {
        "scope": "B three-view PPO integration; not transfer evaluation",
        "seeds": [17, 29, 43],
        "world_export_sha256": sha256_file(args.world_export / "export.json"),
        "policy_manifest_sha256": sha256_file(repo / "configs/eval/artifacts_v1.json"),
        "sources": {str(p.relative_to(repo)): sha256_file(p) for p in paths},
    }
    identity = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()[:16]
    root = Path.home() / "altiseek-m10/ppo-pilots" / identity
    root.mkdir(parents=True, exist_ok=True)
    with (root / "suite.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (root / "definition.json").write_text(json.dumps(definition, indent=2) + "\n")
        (repo / "data/m10/ppo-pilots-path.txt").write_text(str(root) + "\n")
        status = {"state": "running", "run_id": identity, "completed": [], "active_seed": None}

        def save() -> None:
            temporary = root / "status.tmp"
            temporary.write_text(json.dumps(status, indent=2) + "\n")
            temporary.replace(root / "status.json")

        try:
            for seed in definition["seeds"]:
                complete = root / f"seed-{seed}.json"
                if complete.exists():
                    saved = json.loads(complete.read_text())
                    report = Path(saved["flight_run"]) / "search-summary.json"
                    if (
                        saved["run_id"] != identity
                        or sha256_file(report) != saved["report_sha256"]
                        or not json.loads(report.read_text())["passed"]
                    ):
                        raise ValueError("Existing pilot result failed verification")
                    status["completed"].append(seed)
                    continue
                status.update(active_seed=seed)
                save()
                print(f"PPO seed {seed}: starting B integration", flush=True)
                with (root / f"seed-{seed}.log").open("a") as log:
                    process = subprocess.run(
                        [
                            sys.executable,
                            str(repo / "scripts/run_camera_flight.py"),
                            "--world-export",
                            str(args.world_export),
                            "--search-pilot",
                            "--method",
                            "PPO",
                            "--training-seed",
                            str(seed),
                        ],
                        cwd=repo,
                        stdout=log,
                        stderr=log,
                        check=False,
                    )
                flight = Path((repo / "data/m10/camera-flight-path.txt").read_text().strip())
                if process.returncode:
                    raise RuntimeError(f"Seed {seed} failed; inspect {flight}")
                report = flight / "search-summary.json"
                result = json.loads(report.read_text())
                if (
                    not result["passed"]
                    or result["training_seed"] != seed
                    or result["method"] != "PPO"
                ):
                    raise ValueError("Pilot result identity/completion mismatch")
                complete.write_text(
                    json.dumps(
                        {
                            "run_id": identity,
                            "flight_run": str(flight),
                            "report_sha256": sha256_file(report),
                            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        },
                        indent=2,
                    )
                    + "\n"
                )
                status["completed"].append(seed)
                save()
            status.update(state="completed", active_seed=None)
        except Exception as error:
            status.update(state="failed", error=repr(error))
            raise
        finally:
            save()
        print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
