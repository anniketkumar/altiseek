"""Continue full-budget B validation, then gated resumable B calibration and refit.

Does not launch T evaluation: that requires the frozen transfer protocol after
B acquisition/fit review. Defers isolated flight failures; integrity failures stop immediately.
"""

import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

from altiseek.cache.build import header_for
from altiseek.cache.config import CacheConfig
from altiseek.cache.jobs import load_worlds
from altiseek.calibration.config import SensorConfig
from altiseek.calibration.fit import fit
from altiseek.calibration.model import digest
from altiseek.config import load_config
from altiseek.deploy.calibration import load_view, outcomes
from altiseek.registry.run import sha256_file


class FlightFailure(RuntimeError):
    """A failed flight with verified cleanup and a recoverable controller error."""


def recoverable_failure(stage: str, cleanup: dict, summary: dict) -> bool:
    """Only isolated flight/capture failures with confirmed cleanup may be deferred."""
    reason = summary.get("error") or ""
    return bool(
        stage.startswith("calibration-")
        and cleanup.get("owned_processes_stopped") is True
        and summary.get("passed") is False
        and reason.startswith(
            (
                "ValueError('Actual camera missed policy goal by ",
                "ValueError('Image/pose mismatch ",
                "TimeoutError(",
            )
        )
    )


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    cache = load_config(repo / "configs/cache/default.yaml", CacheConfig)
    worlds = load_worlds(cache, "sensor-fit")
    if len(worlds) != 40 or any(w.split != "B" for w in worlds):
        raise ValueError("Expected the complete frozen 40-world B sensor-fit set")
    sources = [
        *sorted((repo / "src/altiseek").rglob("*.py")),
        *sorted((repo / "scripts").glob("*flight*.py")),
        repo / "scripts/export_deploy_world.py",
        repo / "scripts/continue_m10.py",
    ]
    definition = {
        "protocol": "D52",
        "protocol_sha256": sha256_file(repo / "docs/m10-live-protocol.md"),
        "cache_config": cache.model_dump(mode="json"),
        "world_ids": [w.world_id for w in worlds],
        "sources": {str(p.relative_to(repo)): sha256_file(p) for p in sources},
        "detector_sha256": sha256_file(cache.checkpoint),
        "calibration_views_per_world": 220,
    }
    identity = digest(definition)
    native = Path.home() / "altiseek-m10"
    root = native / "continuations" / identity
    root.mkdir(parents=True, exist_ok=True)
    with (native / "continuation.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = root / "status.json"
        prior = json.loads(previous.read_text()) if previous.exists() else {}
        if prior.get("state") == "awaiting_calibration_review":
            return
        deferred_first = None
        if prior.get("state") == "failed":
            flight = Path((repo / "data/m10/camera-flight-path.txt").read_text().strip())
            cleanup = json.loads((flight / "status.json").read_text())
            summary = json.loads((flight / "calibration-summary.json").read_text())
            if not recoverable_failure(prior.get("stage", ""), cleanup, summary) or summary.get(
                "world_id"
            ) != prior.get("active_world"):
                raise ValueError("Previous failure requires review; refusing automatic recovery")
            deferred_first = prior["active_world"]
            with (root / "failures.jsonl").open("a") as log:
                log.write(
                    json.dumps(
                        {
                            "stage": prior["stage"],
                            "flight": str(flight),
                            "error": summary["error"],
                            "action": "deferred on scheduler takeover",
                        }
                    )
                    + "\n"
                )
        (root / "scheduler.json").write_text(
            json.dumps(
                {
                    "version": "D53",
                    "source_sha256": sha256_file(Path(__file__)),
                    "max_passes": 3,
                    "consecutive_failure_limit": 3,
                },
                indent=2,
            )
            + "\n"
        )
        (root / "definition.json").write_text(json.dumps(definition, indent=2) + "\n")
        (repo / "data/m10/continuation-path.txt").write_text(str(root) + "\n")
        status = {
            "run_id": identity,
            "state": "running",
            "stage": "preflight",
            "completed_worlds": 0,
            "completed_views": 0,
        }

        def save() -> None:
            status["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            temporary = root / "status.tmp"
            temporary.write_text(json.dumps(status, indent=2) + "\n")
            temporary.replace(root / "status.json")

        def verify_sources() -> None:
            if any(
                sha256_file(repo / name) != value for name, value in definition["sources"].items()
            ):
                raise ValueError("Acquisition sources changed during the frozen continuation")

        def command(name: str, arguments: list[str]) -> None:
            verify_sources()
            with (root / f"{name}.log").open("a") as log:
                process = subprocess.run(
                    [sys.executable, *arguments], cwd=repo, stdout=log, stderr=log, check=False
                )
            if process.returncode:
                raise RuntimeError(
                    f"{name} exited {process.returncode}; inspect {root / (name + '.log')}"
                )

        def stage(name: str, arguments: list[str], report_name: str) -> dict:
            completed = root / f"{name}.json"
            if completed.exists():
                saved = json.loads(completed.read_text())
                report = Path(saved["report"])
                if saved["run_id"] != identity or sha256_file(report) != saved["sha256"]:
                    raise ValueError("Completed continuation stage checksum mismatch")
                result = json.loads(report.read_text())
                if not result["passed"]:
                    raise ValueError("Completed continuation stage is not passing")
                return result
            status.update(stage=name)
            save()
            before = (repo / "data/m10/camera-flight-path.txt").read_text().strip()
            try:
                command(name, ["scripts/run_camera_flight.py", *arguments])
            except RuntimeError as error:
                after = (repo / "data/m10/camera-flight-path.txt").read_text().strip()
                flight = Path(after)
                cleanup = json.loads((flight / "status.json").read_text())
                summary = json.loads((flight / report_name).read_text())
                reason = summary.get("error") or ""
                # Only known capture/flight errors are recoverable. Unknown errors,
                # checksum failures, export failures and incomplete cleanup stop.
                if after == before or not recoverable_failure(name, cleanup, summary):
                    raise
                with (root / "failures.jsonl").open("a") as log:
                    log.write(
                        json.dumps(
                            {
                                "stage": name,
                                "flight": after,
                                "error": reason,
                                "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            }
                        )
                        + "\n"
                    )
                raise FlightFailure(str(error)) from error
            flight = Path((repo / "data/m10/camera-flight-path.txt").read_text().strip())
            report = flight / report_name
            result = json.loads(report.read_text())
            cleanup = json.loads((flight / "status.json").read_text())
            if not result["passed"] or not cleanup["owned_processes_stopped"]:
                raise ValueError("Flight stage failed completion/cleanup checks")
            completed.write_text(
                json.dumps(
                    {"run_id": identity, "report": str(report), "sha256": sha256_file(report)},
                    indent=2,
                )
                + "\n"
            )
            return result

        try:
            # Review/checkpoint verification completed in the previous all-seed pilots.
            pilot_root = Path((repo / "data/m10/ppo-pilots-path.txt").read_text().strip())
            pilots = json.loads((pilot_root / "status.json").read_text())
            if pilots["state"] != "completed" or pilots["completed"] != [17, 29, 43]:
                raise ValueError("All frozen PPO integration pilots must be completed")
            for seed in (17, 29, 43):
                item = json.loads((pilot_root / f"seed-{seed}.json").read_text())
                if (
                    sha256_file(Path(item["flight_run"]) / "search-summary.json")
                    != item["report_sha256"]
                ):
                    raise ValueError("PPO integration report changed")
            integration_id = "003b0c54f581c65b"
            command(
                "export-integration",
                ["scripts/export_deploy_world.py", f"data/worlds/rl-val/{integration_id}.json"],
            )
            result = stage(
                "full-budget-B5",
                [
                    "--world-export",
                    str(native / "worlds" / integration_id),
                    "--search-pilot",
                    "--normal-launch",
                    "--measured",
                    "--max-views",
                    "0",
                ],
                "search-summary.json",
            )
            if (
                not result["measured_energy"]
                or not result["normal_edge_launch"]
                or result["battery_s"] != 500
            ):
                raise ValueError("Full-budget B check used incorrect settings")

            keys = [
                (li, ix, iy)
                for li, level in enumerate(cache.render.levels.levels)
                for ix, iy in cache.render.levels.lattice(level)
            ]
            contract = {
                "renderer": "Gazebo-Harmonic-ogre-D52-v1",
                "acquisition_id": identity,
                "whole_cell_visibility": True,
                "pose_source": "Gazebo ground truth",
                "camera": cache.render.camera.model_dump(mode="json"),
                "detector_id": cache.detector_id,
                "variants_per_lattice_view": 1,
            }
            contract_id = digest(contract)
            (root / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
            collected = []
            collection_ids = []
            pending = list(enumerate(worlds))
            if deferred_first:
                pending.sort(key=lambda item: item[1].world_id == deferred_first)
            consecutive_failures = 0
            for attempt in range(1, 4):
                status.update(attempt_pass=attempt)
                deferred = []
                for wi, world in pending:
                    storage = root / "views" / world.world_id
                    status.update(stage="calibration", active_world=world.world_id)
                    save()
                    command(
                        f"export-{world.world_id}",
                        [
                            "scripts/export_deploy_world.py",
                            str(cache.worlds_dir / "sensor-fit" / f"{world.world_id}.json"),
                        ],
                    )
                    arguments = [
                        "--world-export",
                        str(native / "worlds" / world.world_id),
                        "--calibration-dir",
                        str(storage),
                        "--calibration-contract",
                        contract_id,
                        "--normal-launch",
                    ]
                    if wi == 0:
                        stage(
                            "calibration-pilot",
                            [*arguments, "--calibration-pilot"],
                            "calibration-summary.json",
                        )
                    try:
                        stage(
                            f"calibration-{world.world_id}", arguments, "calibration-summary.json"
                        )
                    except FlightFailure:
                        deferred.append((wi, world))
                        consecutive_failures += 1
                        status.update(deferred_worlds=[w.world_id for _, w in deferred])
                        save()
                        if consecutive_failures >= 3:
                            raise RuntimeError(
                                "Three consecutive flight failures; review required"
                            ) from None
                        continue
                    consecutive_failures = 0
                    views = [load_view(storage, key, contract_id) for key in keys]
                    if any(view is None or view[0]["world_id"] != world.world_id for view in views):
                        raise ValueError("Calibration world is incomplete or has mixed identity")
                    collected.append((world, views))
                    collection_ids.append(digest([view[0]["sha256"] for view in views]))
                    status.update(
                        completed_worlds=len(collected), completed_views=len(collected) * len(keys)
                    )
                    save()
                pending = deferred
                status.update(deferred_worlds=[w.world_id for _, w in pending])
                save()
                if not pending:
                    break
            if pending:
                raise RuntimeError("Retry passes exhausted; incomplete worlds retained")
            # Preserve the frozen input order even when worlds completed out of order.
            ordered = {
                w.world_id: (w, views, cid)
                for (w, views), cid in zip(collected, collection_ids, strict=True)
            }
            collected = [(ordered[w.world_id][0], ordered[w.world_id][1]) for w in worlds]
            collection_ids = [ordered[w.world_id][2] for w in worlds]
            status.update(stage="B-sensor-refit")
            save()
            sensor = load_config(repo / "configs/sensor/default.yaml", SensorConfig)
            observations = [
                outcomes(world, views, sensor, len(cache.render.levels.levels))
                for world, views in collected
            ]
            shared_contract = header_for(worlds[0], cache).model_dump(
                mode="json", exclude={"world_id", "cache_id"}
            )
            model, stats = fit(
                observations,
                sensor,
                cache,
                collection_ids,
                {**shared_contract, "deployment": contract},
            )
            model_path = repo / "data/sensor_models" / f"{model.sensor_model_id}.json"
            model.save(model_path)
            (root / "fit.json").write_text(json.dumps(stats, indent=2) + "\n")
            status.update(
                state="awaiting_calibration_review",
                stage="B-calibration-complete",
                sensor_model_id=model.sensor_model_id,
                model_sha256=sha256_file(model_path),
                model_path=str(model_path),
            )
        except Exception as error:
            status.update(state="failed", error=repr(error))
            raise
        finally:
            save()
        print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
