"""Independently audit the completed D52 B calibration acquisition and its refit.

Read-only with respect to the acquisition: re-verifies the run identity, every
stage report, all 8,800 per-view checksums/contracts and capture limits, then
refits from the saved views and requires the identical sensor model. Records
controller revision (D52 original or D54 adapter) per flight and all failures.
Writes docs/m10-deployment/m10-calibration-review.json.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from altiseek.cache.build import header_for
from altiseek.cache.config import CacheConfig
from altiseek.cache.jobs import load_worlds
from altiseek.calibration.config import SensorConfig
from altiseek.calibration.fit import fit
from altiseek.calibration.model import SensorModel, digest
from altiseek.config import load_config
from altiseek.deploy.calibration import load_view, outcomes
from altiseek.registry.run import sha256_file

# D52 capture acceptance limits, as enforced in the flight controllers.
POSE_DELTA_LIMIT_S = 0.04
GOAL_ERROR_LIMIT_M = 1.0
ORIGINAL_SENSOR = "fcf8cdae58ca3947"


def summary(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    return {
        "count": int(a.size),
        "mean": float(a.mean()),
        "p50": float(np.quantile(a, 0.5)),
        "p95": float(np.quantile(a, 0.95)),
        "max": float(a.max()),
    }


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    root = Path((repo / "data/m10/continuation-path.txt").read_text().strip())
    status = json.loads((root / "status.json").read_text())
    definition = json.loads((root / "definition.json").read_text())
    contract = json.loads((root / "contract.json").read_text())
    findings: list[str] = []

    identity = digest(definition)
    if identity != status["run_id"] or root.name != identity:
        raise ValueError("Continuation definition does not reproduce its run identity")
    changed = sorted(
        name
        for name, value in definition["sources"].items()
        if not (repo / name).exists() or sha256_file(repo / name) != value
    )
    if status["state"] != "awaiting_calibration_review" or status["deferred_worlds"]:
        raise ValueError("Calibration is not complete without deferred worlds")
    contract_id = digest(contract)
    if contract["acquisition_id"] != identity:
        raise ValueError("Contract belongs to another acquisition")

    cache = load_config(repo / "configs/cache/default.yaml", CacheConfig)
    worlds = load_worlds(cache, "sensor-fit")
    if [w.world_id for w in worlds] != definition["world_ids"]:
        raise ValueError("Frozen sensor-fit world order changed")
    if sha256_file(cache.checkpoint) != definition["detector_sha256"]:
        raise ValueError("Detector checkpoint changed")
    keys = [
        (li, ix, iy)
        for li, level in enumerate(cache.render.levels.levels)
        for ix, iy in cache.render.levels.lattice(level)
    ]

    def stage(name: str, report_name: str) -> dict:
        saved = json.loads((root / f"{name}.json").read_text())
        report = Path(saved["report"])
        if saved["run_id"] != identity or sha256_file(report) != saved["sha256"]:
            raise ValueError(f"{name}: stage report checksum mismatch")
        if report.name != report_name or not json.loads(report.read_text())["passed"]:
            raise ValueError(f"{name}: stage report is not passing")
        cleanup = json.loads((report.parent / "status.json").read_text())
        if cleanup["state"] != "passed" or not cleanup["owned_processes_stopped"]:
            raise ValueError(f"{name}: flight cleanup not confirmed")
        return {"flight": str(report.parent), "report_sha256": saved["sha256"]}

    stages = {"full-budget-B5": stage("full-budget-B5", "search-summary.json")}
    stages["calibration-pilot"] = stage("calibration-pilot", "calibration-summary.json")

    flights: dict[str, dict] = {}
    collected, collection_ids = [], []
    per_level = [Counter() for _ in cache.render.levels.levels]
    deltas, errors = [], []
    for world in worlds:
        stages[f"calibration-{world.world_id}"] = stage(
            f"calibration-{world.world_id}", "calibration-summary.json"
        )
        summary_path = Path(stages[f"calibration-{world.world_id}"]["flight"])
        report = json.loads((summary_path / "calibration-summary.json").read_text())
        if (
            report["world_id"] != world.world_id
            or report["contract_id"] != contract_id
            or report["completed"] != len(keys)
        ):
            raise ValueError(f"{world.world_id}: calibration summary disagrees")
        views = [load_view(root / "views" / world.world_id, key, contract_id) for key in keys]
        for key, view in zip(keys, views, strict=True):
            if view is None:
                raise ValueError(f"{world.world_id}: missing view {key}")
            meta = view[0]
            if meta["world_id"] != world.world_id or meta["level"] != key[0]:
                raise ValueError(f"{world.world_id}: view {key} has mixed identity")
            if (
                meta["pose_delta_s"] > POSE_DELTA_LIMIT_S
                or meta["goal_error_m"] > GOAL_ERROR_LIMIT_M
            ):
                raise ValueError(f"{world.world_id}: view {key} violates capture limits")
            deltas.append(meta["pose_delta_s"])
            errors.append(meta["goal_error_m"])
            per_level[key[0]]["views"] += 1
            per_level[key[0]]["visible_cells"] += int(view[2].sum())
            flight = flights.setdefault(meta["flight_run"], {"views": 0, "worlds": set()})
            flight["views"] += 1
            flight["worlds"].add(world.world_id)
        collected.append((world, views))
        collection_ids.append(digest([view[0]["sha256"] for view in views]))

    controllers = Counter()
    retry_records = 0
    for path, flight in sorted(flights.items()):
        run = Path(path)
        recovery = run / "capture-recovery.json"
        retries = run / "capture-retries.jsonl"
        flight["controller"] = "D54 adapter" if recovery.exists() else "D52 original"
        if recovery.exists():
            flight["adapter_sources_sha256"] = json.loads(recovery.read_text())["sources_sha256"]
        flight["retry_records"] = len(retries.read_text().splitlines()) if retries.exists() else 0
        retry_records += flight["retry_records"]
        flight["worlds"] = sorted(flight["worlds"])
        controllers[flight["controller"]] += flight["views"]

    failures = [json.loads(line) for line in (root / "failures.jsonl").read_text().splitlines()]
    failed_flights = {}
    for failure in failures:
        run = Path(failure["flight"])
        cleanup = json.loads((run / "status.json").read_text())
        if not cleanup["owned_processes_stopped"]:
            raise ValueError(f"Failed flight {run} lacks confirmed cleanup")
        failed_flights[str(run)] = {
            "stage": failure["stage"],
            "error": failure["error"],
            "views_contributed": flights.get(str(run), {}).get("views", 0),
        }

    # Refit from verified views exactly as the scheduler did; identity must reproduce.
    sensor = load_config(repo / "configs/sensor/default.yaml", SensorConfig)
    observations = [
        outcomes(world, views, sensor, len(cache.render.levels.levels))
        for world, views in collected
    ]
    shared = header_for(worlds[0], cache).model_dump(mode="json", exclude={"world_id", "cache_id"})
    model, stats = fit(
        observations, sensor, cache, collection_ids, {**shared, "deployment": contract}
    )
    saved_path = Path(status["model_path"])
    saved = SensorModel.load(
        saved_path,
        detector_id=cache.detector_id,
        fit_world_set_id=cache.world_set_ids["sensor-fit"],
        cache_contract={**shared, "deployment": contract},
    )
    if not model.sensor_model_id == saved.sensor_model_id == status["sensor_model_id"]:
        raise ValueError("Refit does not reproduce the saved sensor model")
    if sha256_file(saved_path) != status["model_sha256"]:
        raise ValueError("Saved sensor model checksum changed")
    if json.loads(json.dumps(stats)) != json.loads((root / "fit.json").read_text()):
        raise ValueError("Refit statistics differ from the saved fit report")

    original = SensorModel.load(
        repo / f"data/sensor_models/{ORIGINAL_SENSOR}.json", detector_id=cache.detector_id
    )
    original_fit = json.loads((repo / f"data/sensor_models/{ORIGINAL_SENSOR}.fit.json").read_text())
    comparison = []
    for li, (new, old) in enumerate(zip(saved.levels, original.levels, strict=True)):
        row = stats["levels"][li]
        old_row = original_fit["levels"][li]
        calib = stats["calibration"][li]
        old_calib = original_fit["calibration"][li]
        comparison.append(
            {
                "level": new.name,
                "views": per_level[li]["views"],
                "visible_cells": per_level[li]["visible_cells"],
                "gazebo": {
                    "binary_recall": row["binary"]["recall"],
                    "binary_false_alarm": row["binary"]["false_alarm"],
                    "soft_recall": row["soft"]["recall"],
                    "soft_false_alarm": row["soft"]["false_alarm"],
                    "temperature": new.temperature,
                    "ece_before": calib["before"]["ece"],
                    "ece_after": calib["after"]["ece"],
                },
                "cached_original": {
                    "binary_recall_raw": old_row["binary"]["recall"]["raw"],
                    "binary_recall_ci95": old_row["binary"]["recall"]["ci95"],
                    "binary_false_alarm_raw": old_row["binary"]["false_alarm"]["raw"],
                    "soft_recall_raw": old_row["soft"]["recall"]["raw"],
                    "soft_false_alarm_raw": old_row["soft"]["false_alarm"]["raw"],
                    "temperature": old.temperature,
                    "ece_before": old_calib["before"]["ece"],
                    "ece_after": old_calib["after"]["ece"],
                },
            }
        )
        if row["binary"]["conditional_recall"]["denominator"] != 0:
            findings.append(f"{new.name}: unexpected conditional-recall trials")
        recall_ci = row["binary"]["recall"]["ci95"]
        old_recall = old_row["binary"]["recall"]["raw"]
        if not recall_ci[0] <= old_recall <= recall_ci[1]:
            findings.append(
                f"{new.name}: cached binary recall {old_recall:.3f} lies outside the Gazebo "
                f"95% CI [{recall_ci[0]:.3f}, {recall_ci[1]:.3f}]"
            )
        if calib["after"]["ece"] > calib["before"]["ece"]:
            findings.append(f"{new.name}: temperature scaling increased fit-set ECE")
    if changed:
        findings.append(
            "Current tree differs from frozen acquisition sources (re-verification of "
            f"identity needs the acquisition commit): {changed}"
        )

    review = {
        "decision": "D55",
        "run_id": identity,
        "contract_id": contract_id,
        "sensor_model_id": saved.sensor_model_id,
        "sensor_model_sha256": status["model_sha256"],
        "fit_world_set_id": saved.fit_world_set_id,
        "detector_id": saved.detector_id,
        "worlds": len(collected),
        "views": sum(len(v) for _, v in collected),
        "refit_reproduced": True,
        "stages": stages,
        "pose_delta_s": summary(deltas),
        "goal_error_m": summary(errors),
        "views_by_controller": dict(controllers),
        "capture_retry_records": retry_records,
        "flights": flights,
        "failures": failures,
        "failed_flights": failed_flights,
        "global_binary": {
            "gazebo": stats["global_binary"],
            "cached_original": original_fit["global_binary"],
        },
        "levels": comparison,
        "conditional_recall": "not measured: one view per lattice location (D52)",
        "frozen_sources_changed": changed,
        "findings": findings,
    }
    out = repo / "docs/m10-deployment/m10-calibration-review.json"
    out.write_text(json.dumps(review, indent=2) + "\n")
    print(json.dumps({k: review[k] for k in ("run_id", "sensor_model_id", "views", "findings")}))


if __name__ == "__main__":
    main()
