"""Frozen M10 transfer (D56): B refit preflight, cached references, live T cases.

Resumable: verified completions are reused. Isolated operational flight failures
are logged and deferred (three attempts per case, three consecutive failures
stop); integrity or unknown failures stop for review. Nothing is dropped.
"""

import argparse
import fcntl
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO

from altiseek.cache.config import CacheConfig
from altiseek.calibration.model import SensorModel, digest
from altiseek.config import load_config
from altiseek.deploy.transfer import (
    METHODS,
    OperationalFailure,
    live_cases,
    load_protocol,
    operational_failure,
    paired_seed,
    run_deferred,
    sensor_path,
)
from altiseek.env.config import SearchConfig
from altiseek.env.search import SearchEnv
from altiseek.eval.runner import play
from altiseek.policies.config import SelectedBaselines
from altiseek.registry.run import sha256_file
from altiseek.world.spec import WorldSpec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--stop-after-preflight", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    protocol = load_protocol(args.protocol)
    pid = protocol["protocol_id"]
    root = Path.home() / "altiseek-m10/transfers" / pid
    root.mkdir(parents=True, exist_ok=True)
    (repo / "data/m10/transfer-run-path.txt").write_text(str(root) + "\n")
    flight_pointer = repo / "data/m10/camera-flight-path.txt"
    with (root / "transfer.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {
            "state": "running",
            "stage": "B-refit-preflight",
            "protocol_id": pid,
            "live_complete": 0,
            "live_total": protocol["live_episodes"],
            "cached_complete": 0,
            "cached_total": protocol["cached_episodes"],
        }
        failures_log = root / "failures.jsonl"
        failed_attempts: Counter = Counter()
        if failures_log.exists():
            for line in failures_log.read_text().splitlines():
                failed_attempts[digest(json.loads(line)["case"])] += 1

        def save() -> None:
            state["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            temporary = root / "status.tmp"
            temporary.write_text(json.dumps(state, indent=2) + "\n")
            temporary.replace(root / "status.json")

        def command(name: str, argv: list[str]) -> int:
            load_protocol(args.protocol)  # Every input, source and artifact still frozen.
            with (root / f"{name}.log").open("a") as log:
                return subprocess.run(
                    [sys.executable, *argv], cwd=repo, stdout=log, stderr=log, check=False
                ).returncode

        def required(name: str, argv: list[str]) -> None:
            if command(name, argv):
                raise RuntimeError(f"{name} failed; inspect {root / (name + '.log')}")

        def completed(path: Path, key: dict) -> dict | None:
            if not path.exists():
                return None
            saved = json.loads(path.read_text())
            if saved["key"] != key or saved["protocol_id"] != pid:
                raise ValueError("Transfer completion identity mismatch")
            if sha256_file(Path(saved["report_path"])) != saved["sha256"]:
                raise ValueError("Transfer completion checksum mismatch")
            return saved

        def record_failure(case: dict, error: OperationalFailure) -> None:
            with failures_log.open("a") as log:
                entry = {
                    "case": case,
                    "error": str(error),
                    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                log.write(json.dumps(entry) + "\n")

        def live_case(case: dict) -> bool:
            smoke = case["smoke"]
            destination = root / ("preflight" if smoke else "live") / f"{digest(case)}.json"
            destination.parent.mkdir(exist_ok=True)
            if completed(destination, case) is not None:
                return False
            state["active_case"] = case
            save()
            world_id, method, seed, sensor = (
                case["world_id"],
                case["method"],
                case["training_seed"],
                case["sensor"],
            )
            folder = "rl-val" if smoke else "test"
            required(
                f"export-{world_id}",
                [
                    "scripts/deploy/export_deploy_world.py",
                    f"data/worlds/{folder}/{world_id}.json",
                    "--protocol",
                    str(args.protocol),
                ],
            )
            before = flight_pointer.read_text() if flight_pointer.exists() else ""
            started = time.monotonic()
            command(
                f"flight-{digest(case)}",
                [
                    "scripts/deploy/run_m10_transfer_flight.py",
                    "--world-export",
                    str(Path.home() / "altiseek-m10/worlds" / world_id),
                    "--search-pilot",
                    "--normal-launch",
                    "--measured",
                    "--max-views",
                    "0",
                    "--method",
                    method,
                    "--training-seed",
                    str(seed or 17),
                    "--protocol",
                    str(args.protocol),
                    "--sensor-case",
                    sensor,
                ],
            )
            wall_s = time.monotonic() - started
            pointer = flight_pointer.read_text()
            if pointer == before:
                raise RuntimeError("Flight launcher failed before creating a run; review")
            flight = Path(pointer.strip())
            report_path = flight / "search-summary.json"
            if not report_path.exists():
                raise RuntimeError(f"Flight {flight} wrote no search summary; review")
            result = json.loads(report_path.read_text())
            cleanup = json.loads((flight / "status.json").read_text())
            flight_report = flight / "flight-summary.json"
            flight_summary = json.loads(flight_report.read_text()) if flight_report.exists() else {}
            if not result["passed"]:
                if operational_failure(result, cleanup, flight_summary):
                    raise OperationalFailure(f"{result['error']} in {flight}")
                raise ValueError(f"Live case failed without a recoverable cause: {flight}")
            if (
                not cleanup["owned_processes_stopped"]
                or result["protocol_id"] != pid
                or result["world_id"] != world_id
                or result["sensor_case"] != sensor
                or result["method"] != method
                or (result["training_seed"] or 0) != seed
                or not result["measured_energy"]
                or not result["normal_edge_launch"]
                or result["battery_s"] != 500
                or result["view_cap"] != 0
                or not result["search"][-1]["ended"]
                or result["environment_seed"] != paired_seed(world_id, "launch")
                or result["action_seed"] != paired_seed(world_id, "actions", seed)
            ):
                raise ValueError(f"Live result failed identity/completion checks: {flight}")
            destination.write_text(
                json.dumps(
                    {
                        "protocol_id": pid,
                        "key": case,
                        "report_path": str(report_path),
                        "sha256": sha256_file(report_path),
                        "wall_s": wall_s,
                    },
                    indent=2,
                )
                + "\n"
            )
            if not smoke:
                state["live_complete"] = len(list(destination.parent.glob("*.json")))
            return True

        state["live_complete"] = len(list((root / "live").glob("*.json")))
        try:
            save()
            preflight = [
                {
                    "world_id": protocol["b_preflight_world_id"],
                    "method": method,
                    "training_seed": seed,
                    "sensor": "refit",
                    "smoke": True,
                }
                for method, seed in METHODS
            ]
            run_deferred(preflight, live_case, record_failure, failed_attempts)
            if args.stop_after_preflight:
                state.update(state="awaiting_preflight_review", active_case=None)
                return
            state.update(stage="matched-cached-references")
            save()
            cache = load_config("configs/cache/default.yaml", CacheConfig)
            search = load_config("configs/env/search.yaml", SearchConfig)
            selected = load_config("configs/policy/selected.yaml", SelectedBaselines)
            sensor = SensorModel.load(
                sensor_path(protocol, "original"),
                detector_id=cache.detector_id,
                fit_world_set_id=cache.world_set_ids["sensor-fit"],
            )
            torch.set_num_threads(2)
            models = {}
            for world_id in protocol["world_ids"]:
                world = WorldSpec.load(cache.worlds_dir / "test" / f"{world_id}.json")
                for method, seed in METHODS:
                    key = {
                        "world_id": world_id,
                        "method": method,
                        "training_seed": seed,
                        "variant": "V3",
                        "environment_seed": paired_seed(world_id, "launch"),
                        "action_seed": paired_seed(world_id, "actions", seed),
                    }
                    destination = root / "cached" / f"{digest(key)}.json"
                    destination.parent.mkdir(exist_ok=True)
                    if completed(destination, key) is None:
                        model = None
                        env = SearchEnv([world], cache, sensor, search)
                        if method == "PPO":
                            if seed not in models:
                                item = next(c for c in protocol["checkpoints"] if c["seed"] == seed)
                                models[seed] = MaskablePPO.load(item["checkpoint"], device="cpu")
                                models[seed].policy.set_training_mode(False)
                            model = models[seed]
                            if (
                                model.policy.features_extractor.zero_belief
                                or model.observation_space != env.observation_space
                            ):
                                raise ValueError("Cached reference policy contract mismatch")
                        parameters = {} if model else selected.settings[world.family]["V3"][method]
                        result = play(env, key, parameters, model)
                        report_path = destination.with_suffix(".result.json")
                        report_path.write_text(json.dumps(result, indent=2) + "\n")
                        destination.write_text(
                            json.dumps(
                                {
                                    "protocol_id": pid,
                                    "key": key,
                                    "report_path": str(report_path),
                                    "sha256": sha256_file(report_path),
                                },
                                indent=2,
                            )
                            + "\n"
                        )
                    state["cached_complete"] += 1
                    save()
            state.update(stage="live-transfer")
            cases = [{**case, "smoke": False} for case in live_cases(protocol["world_ids"])]
            run_deferred(cases, live_case, record_failure, failed_attempts)
            state.update(stage="reporting", active_case=None)
            save()
            required(
                "report", ["scripts/deploy/report_m10_transfer.py", str(args.protocol), str(root)]
            )
            required("demo", ["scripts/deploy/demo_m10_transfer.py", str(args.protocol), str(root)])
            state.update(state="awaiting_final_review", stage="reports-and-demo-complete")
        except Exception as error:
            state.update(state="failed", error=repr(error))
            raise
        finally:
            save()
        print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
