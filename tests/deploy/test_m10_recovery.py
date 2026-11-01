"""Recovery must never treat integrity failures as transient flight problems."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fcntl")
SPEC = importlib.util.spec_from_file_location(
    "m10_recovery", Path(__file__).resolve().parents[2] / "scripts/continue_m10_resilient.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_telemetry_recovery_requires_specific_flight_reason():
    summary = {"passed": False, "error": "RuntimeError('Flight ended before policy observation')"}
    cleanup = {"owned_processes_stopped": True}
    assert not MODULE.recoverable_failure("calibration-world", cleanup, summary)
    assert MODULE.recoverable_failure(
        "calibration-world", cleanup, summary, {"reason": "Telemetry became stale or invalid"}
    )
    assert not MODULE.recoverable_failure(
        "calibration-world", {}, summary, {"reason": "Telemetry became stale or invalid"}
    )


@pytest.mark.parametrize(
    "reason",
    [
        "ValueError('Actual camera missed policy goal by 2.011 m')",
        "ValueError('Image/pose mismatch 0.060 s')",
        "TimeoutError('No settled camera frame')",
    ],
)
def test_recoverable_flight_requires_cleanup(reason):
    summary = {"passed": False, "error": reason}
    assert MODULE.recoverable_failure(
        "calibration-world", {"owned_processes_stopped": True}, summary
    )
    assert not MODULE.recoverable_failure("calibration-world", {}, summary)
    assert not MODULE.recoverable_failure(
        "export-world", {"owned_processes_stopped": True}, summary
    )


@pytest.mark.parametrize(
    "reason",
    [
        "ValueError('checksum mismatch')",
        "ValueError('wrong world identity')",
        "RuntimeError('unknown controller failure')",
        "",
        None,
    ],
)
def test_integrity_and_unknown_errors_stop(reason):
    assert not MODULE.recoverable_failure(
        "calibration-world",
        {"owned_processes_stopped": True},
        {"passed": False, "error": reason},
    )


@pytest.mark.parametrize("failure_kind", ["transient", "consecutive", "integrity"])
def test_failed_world_is_deferred_and_fit_order_preserved(tmp_path, monkeypatch, failure_kind):
    repo = tmp_path / "repo"
    native = tmp_path / "altiseek-m10"
    (repo / "data/m10").mkdir(parents=True)
    native.mkdir()
    monkeypatch.setattr(MODULE, "__file__", str(repo / "scripts/continue_m10_resilient.py"))
    monkeypatch.setattr(MODULE.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(MODULE, "sha256_file", lambda path: "verified-test-hash")
    worlds = [SimpleNamespace(world_id=f"world-{i}", split="B") for i in range(40)]
    levels = SimpleNamespace(levels=[0], lattice=lambda _: [(0, 0)])
    cache = SimpleNamespace(
        model_dump=lambda **_: {},
        checkpoint=Path("detector"),
        detector_id="detector",
        worlds_dir=Path("worlds"),
        render=SimpleNamespace(levels=levels, camera=SimpleNamespace(model_dump=lambda **_: {})),
    )
    monkeypatch.setattr(MODULE, "load_config", lambda *args: cache)
    monkeypatch.setattr(MODULE, "load_worlds", lambda *args: worlds)
    monkeypatch.setattr(
        MODULE, "header_for", lambda *args: SimpleNamespace(model_dump=lambda **_: {})
    )
    monkeypatch.setattr(
        MODULE,
        "load_view",
        lambda storage, *args: ({"world_id": storage.name, "sha256": storage.name}, None, None),
    )
    monkeypatch.setattr(MODULE, "outcomes", lambda world, *args: world.world_id)
    fitted = []

    def fit(observations, *args):
        fitted.extend(observations)
        return SimpleNamespace(sensor_model_id="test", save=lambda _: None), {}

    monkeypatch.setattr(MODULE, "fit", fit)
    pilots = tmp_path / "pilots"
    pilots.mkdir()
    (pilots / "status.json").write_text(
        json.dumps({"state": "completed", "completed": [17, 29, 43]})
    )
    for seed in (17, 29, 43):
        (pilots / f"seed-{seed}.json").write_text(
            json.dumps(
                {
                    "flight_run": str(pilots),
                    "report_sha256": "verified-test-hash",
                }
            )
        )
    (repo / "data/m10/ppo-pilots-path.txt").write_text(str(pilots))
    pointer = repo / "data/m10/camera-flight-path.txt"
    pointer.write_text("previous-flight")
    attempts = []

    def run(command, **kwargs):
        if command[1] != "scripts/run_camera_flight.py":
            return SimpleNamespace(returncode=0)
        flight = tmp_path / f"flight-{len(list(tmp_path.glob('flight-*')))}"
        flight.mkdir()
        pointer.write_text(str(flight))
        summary = {
            "passed": True,
            "measured_energy": True,
            "normal_edge_launch": True,
            "battery_s": 500,
        }
        report = "search-summary.json"
        failed = False
        if "--calibration-dir" in command:
            world_id = Path(command[command.index("--calibration-dir") + 1]).name
            report = "calibration-summary.json"
            if "--calibration-pilot" not in command:
                failures = (
                    {"world-1", "world-2", "world-3"}
                    if failure_kind == "consecutive"
                    else {"world-1"}
                )
                failed = world_id in failures and world_id not in attempts
                attempts.append(world_id)
            summary.update(world_id=world_id, passed=not failed)
            if failed:
                summary["error"] = "ValueError('Actual camera missed policy goal by 2.011 m')"
                if failure_kind == "integrity":
                    summary["error"] = "ValueError('checksum mismatch')"
        (flight / report).write_text(json.dumps(summary))
        (flight / "status.json").write_text(json.dumps({"owned_processes_stopped": True}))
        return SimpleNamespace(returncode=int(failed))

    monkeypatch.setattr(MODULE.subprocess, "run", run)
    if failure_kind != "transient":
        with pytest.raises(RuntimeError):
            MODULE.main()
        assert not fitted
        assert attempts == (
            ["world-0", "world-1", "world-2", "world-3"]
            if failure_kind == "consecutive"
            else ["world-0", "world-1"]
        )
        return
    MODULE.main()
    assert attempts == [w.world_id for w in worlds] + ["world-1"]
    assert fitted == [w.world_id for w in worlds]
    root = Path((repo / "data/m10/continuation-path.txt").read_text().strip())
    assert json.loads((root / "status.json").read_text())["state"] == "awaiting_calibration_review"
    assert len((root / "failures.jsonl").read_text().splitlines()) == 1
    # Completed runs never acquire again.
    MODULE.main()
    assert len(attempts) == 41
