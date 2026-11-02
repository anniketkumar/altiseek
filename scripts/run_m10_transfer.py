"""Frozen B refit preflight, matched cached references and resumable live T cases."""

import argparse
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO

from altiseek.cache.config import CacheConfig
from altiseek.calibration.model import SensorModel, digest
from altiseek.config import load_config
from altiseek.deploy.transfer import load_protocol, paired_seed, sensor_path
from altiseek.env.config import SearchConfig
from altiseek.env.search import SearchEnv
from altiseek.eval.runner import play
from altiseek.policies.config import SelectedBaselines
from altiseek.registry.run import sha256_file
from altiseek.world.spec import WorldSpec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    protocol = load_protocol(args.protocol)
    root = Path.home() / "altiseek-m10/transfers" / protocol["protocol_id"]
    root.mkdir(parents=True, exist_ok=True)
    (repo / "data/m10/transfer-run-path.txt").write_text(str(root) + "\n")
    methods = [("B5", 0), ("PPO", 17), ("PPO", 29), ("PPO", 43)]
    with (root / "transfer.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {"state": "running", "stage": "B-refit-preflight", "protocol_id": protocol["protocol_id"],
                 "live_complete": 0, "live_total": 192, "cached_complete": 0}

        def save() -> None:
            state["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            temporary = root / "status.tmp"
            temporary.write_text(json.dumps(state, indent=2) + "\n")
            temporary.replace(root / "status.json")

        def command(name: str, argv: list[str]) -> None:
            load_protocol(args.protocol)
            with (root / f"{name}.log").open("a") as log:
                result = subprocess.run([sys.executable, *argv], cwd=repo, stdout=log, stderr=log)
            if result.returncode:
                raise RuntimeError(f"{name} failed; inspect {root / (name + '.log')}")

        def completed(path: Path, key: dict) -> dict | None:
            if not path.exists():
                return None
            saved = json.loads(path.read_text())
            if saved["key"] != key or saved["protocol_id"] != protocol["protocol_id"]:
                raise ValueError("Transfer completion identity mismatch")
            if sha256_file(saved["report_path"]) != saved["sha256"]:
                raise ValueError("Transfer completion checksum mismatch")
            return json.loads(Path(saved["report_path"]).read_text())

        def live_case(world_id: str, method: str, seed: int, sensor: str, smoke: bool) -> None:
            key = {"world_id": world_id, "method": method, "training_seed": seed,
                   "sensor": sensor, "smoke": smoke}
            identity = digest(key)
            destination = root / ("smoke" if smoke else "live") / f"{identity}.json"
            destination.parent.mkdir(exist_ok=True)
            if completed(destination, key) is not None:
                return
            state["active_case"] = key
            save()
            folder = "rl-val" if smoke else "test"
            command(f"export-{world_id}", ["scripts/export_deploy_world.py",
                    f"data/worlds/{folder}/{world_id}.json", "--protocol", str(args.protocol)])
            command(f"flight-{identity}", ["scripts/run_camera_flight.py", "--world-export",
                    str(Path.home() / "altiseek-m10/worlds" / world_id), "--search-pilot",
                    "--normal-launch", "--measured", "--max-views", "3" if smoke else "0",
                    "--method", method, "--training-seed", str(seed or 17),
                    "--protocol", str(args.protocol), "--sensor-case", sensor])
            flight = Path((repo / "data/m10/camera-flight-path.txt").read_text().strip())
            report_path = flight / "search-summary.json"
            result = json.loads(report_path.read_text())
            cleanup = json.loads((flight / "status.json").read_text())
            if (not result["passed"] or not cleanup["owned_processes_stopped"]
                    or result["protocol_id"] != protocol["protocol_id"]
                    or result["world_id"] != world_id or result["sensor_case"] != sensor
                    or result["method"] != method or (result["training_seed"] or 0) != seed
                    or not result["measured_energy"] or not result["normal_edge_launch"]):
                raise ValueError("Live transfer result failed identity/completion checks")
            if not smoke and not result["search"][-1]["ended"]:
                raise ValueError("Held-out episode ended at a pilot cap")
            destination.write_text(json.dumps({"protocol_id": protocol["protocol_id"], "key": key,
                "report_path": str(report_path), "sha256": sha256_file(report_path)}, indent=2) + "\n")

        try:
            save()
            for method, seed in methods:
                live_case(protocol["b_preflight_world_id"], method, seed, "refit", True)
            state.update(stage="matched-cached-references")
            save()
            cache = load_config("configs/cache/default.yaml", CacheConfig)
            search = load_config("configs/env/search.yaml", SearchConfig)
            selected = load_config("configs/policy/selected.yaml", SelectedBaselines)
            sensor = SensorModel.load(sensor_path(protocol, "original"), detector_id=cache.detector_id,
                                      fit_world_set_id=cache.world_set_ids["sensor-fit"])
            torch.set_num_threads(2)
            models = {}
            for world_id in protocol["world_ids"]:
                world = WorldSpec.load(cache.worlds_dir / "test" / f"{world_id}.json")
                for method, seed in methods:
                    key = {"world_id": world_id, "method": method, "training_seed": seed,
                           "variant": "V3", "environment_seed": paired_seed(world_id, "launch"),
                           "action_seed": paired_seed(world_id, "actions", seed)}
                    destination = root / "cached" / f"{digest(key)}.json"
                    destination.parent.mkdir(exist_ok=True)
                    if completed(destination, key) is None:
                        model = None
                        env = SearchEnv([world], cache, sensor, search)
                        if method == "PPO":
                            if seed not in models:
                                checkpoint = next(p for p in protocol["checkpoints"] if p["seed"] == seed)
                                models[seed] = MaskablePPO.load(checkpoint["checkpoint"], device="cpu")
                                models[seed].policy.set_training_mode(False)
                            model = models[seed]
                            if model.policy.features_extractor.zero_belief or model.observation_space != env.observation_space:
                                raise ValueError("Cached reference policy contract mismatch")
                        result = play(env, key, selected.settings[world.family]["V3"][method] if method != "PPO" else {}, model)
                        report_path = destination.with_suffix(".result.json")
                        report_path.write_text(json.dumps(result, indent=2) + "\n")
                        destination.write_text(json.dumps({"protocol_id": protocol["protocol_id"],
                            "key": key, "report_path": str(report_path), "sha256": sha256_file(report_path)}, indent=2) + "\n")
                    state["cached_complete"] += 1
                    save()
            cases = [(w, method, seed, sensor) for w in protocol["world_ids"]
                     for method, seed in methods for sensor in ("original", "refit")]
            cases.sort(key=lambda case: digest(["m10-order-v1", *case]))
            state.update(stage="live-transfer")
            for case in cases:
                live_case(*case, False)
                state["live_complete"] += 1
                save()
            state.update(stage="reporting")
            save()
            command("report", ["scripts/report_m10_transfer.py", str(args.protocol), str(root)])
            command("demo", ["scripts/demo_m10_transfer.py", str(args.protocol), str(root)])
            state.update(state="awaiting_final_review", stage="reports-and-demo-complete", active_case=None)
        except Exception as error:
            state.update(state="failed", error=repr(error))
            raise
        finally:
            save()
        print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
