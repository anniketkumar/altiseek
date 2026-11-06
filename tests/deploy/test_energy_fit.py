"""Scientific guards for the D63 B-only estimator, using synthetic records only."""

import copy
import json
from dataclasses import replace

import numpy as np
import pytest

from altiseek.deploy.energy_fit import (
    SOURCES,
    b_worlds,
    calibration_intervals,
    median_fit,
    native_path,
    residual,
    verified_search,
)
from altiseek.env.energy import EnergyModel
from altiseek.registry.run import sha256_file

ENERGY = EnergyModel(
    horizontal_speed_mps=10,
    vertical_speed_mps=3,
    dwell_s=2,
    climb_multiplier=1.5,
    battery_s=500,
)


def view(stamp, elapsed, goal, *, flight="flight-a", world="world-a", level=0):
    return {
        "stamp_s": stamp,
        "elapsed_s": 9999,
        "flight_and_inference_wall_s": 8888,
        "goal_enu_m": goal,
        "level": level,
        "world_id": world,
        "flight_run": flight,
        "key": [level, int(goal[0]), int(goal[1])],
        "measured": {
            "elapsed_s": elapsed,
            "energy_s": elapsed,
            "started": True,
            "exhausted": False,
        },
    }


def test_primary_uses_meter_and_commanded_3d_move_not_capture_or_wall_time():
    a = view(100, 6, [0, 0, 30])
    b = view(500, 20, [30, 40, 36], level=1)
    row = residual(a, b, ENERGY, measured=True)
    assert row["elapsed_s"] == 14
    assert row["nominal_s"] == 9  # 5 horizontal + 2 vertical + 2 dwell
    assert row["overhead_s"] == 5
    assert row["energy_residual_s"] == 4  # climb power retained separately
    assert row["motion"] == "climb"


def test_negative_residual_is_retained():
    row = residual(view(0, 3, [0, 0, 30]), view(1, 4, [0, 0, 30]), ENERGY, measured=True)
    assert row["overhead_s"] == -1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("elapsed_s", 6),
        ("elapsed_s", float("nan")),
        ("energy_s", -1),
        ("exhausted", True),
        ("started", False),
    ],
)
def test_invalid_meter_is_rejected(field, value):
    a, b = view(1, 6, [0, 0, 30]), view(2, 20, [10, 0, 30])
    b["measured"][field] = value
    with pytest.raises(ValueError, match=r"Non-monotone|Non-finite|invalid meter"):
        residual(a, b, ENERGY, measured=True)


def test_equal_flight_weight_not_pooled_step_median_and_bootstrap_units():
    flights = [[1], [2, 2], [3], [4], [100] * 100]
    result = median_fit(flights)
    assert result["overhead_s"] == 3
    assert np.median([v for flight in flights for v in flight]) == 100
    # The bootstrap's support is flight medians, not individual transitions.
    assert result["ci95_s"] == [1, 100]
    assert result == median_fit(flights)
    assert median_fit([[0, 2], [1], [1], [1], [1]])["flight_medians_s"][0] == 1


def test_fit_requires_all_five_flights():
    with pytest.raises(ValueError, match="five nonempty"):
        median_fit([[1], [2], [3], [4], []])


def test_calibration_uses_capture_clock_and_never_bridges_restarts():
    records = [
        view(10, 999, [0, 0, 30]),
        view(20, 999, [25, 0, 30]),
        view(1, 999, [100, 100, 30], flight="flight-b"),
        view(13, 999, [100, 150, 30], flight="flight-b"),
    ]
    rows = calibration_intervals(list(reversed(records)), ENERGY)
    assert len(rows) == 2
    assert [r["overhead_s"] for r in rows] == [5.5, 5]
    assert all("energy_residual_s" not in r for r in rows)
    with pytest.raises(ValueError, match="bridge"):
        residual(records[0], records[2], ENERGY, measured=False)
    duplicate = copy.deepcopy(records[0])
    with pytest.raises(ValueError, match="Non-monotone"):
        calibration_intervals([records[0], duplicate], ENERGY)


def test_test_set_is_rejected_before_any_file_is_opened(tmp_path):
    with pytest.raises(ValueError, match="Only declared B"):
        b_worlds(tmp_path, "test")


@pytest.mark.parametrize("recorded", ["/tmp/data.json", "/home/aniket/altiseek-m10/../test.json"])
def test_native_path_cannot_escape_source_root(tmp_path, recorded):
    with pytest.raises(ValueError, match=r"traversal|not in the subpath"):
        native_path(tmp_path, recorded)


def test_pinned_search_rejects_checksum_path_and_test_world(tmp_path):
    source = SOURCES[1]
    relative = f"camera-flights/{source.flight}/search-summary.json"
    report_path = tmp_path / relative
    report_path.parent.mkdir(parents=True)
    report = {
        "passed": True,
        "world_id": source.world,
        "method": "B5",
        "training_seed": None,
        "measured_energy": True,
        "normal_edge_launch": True,
        "battery_s": 500,
        "view_cap": 0,
        "error": "",
        "search": [{"ended": True}],
        "protocol_id": "5560516d08907341",
        "sensor_case": "refit",
    }
    report_path.write_text(json.dumps(report))
    report_path.with_name("status.json").write_text(
        json.dumps({"state": "passed", "owned_processes_stopped": True})
    )
    source = replace(source, sha256=sha256_file(report_path))
    manifest = {
        "protocol_id": "5560516d08907341",
        "key": {
            "world_id": source.world,
            "method": "B5",
            "training_seed": 0,
            "sensor": "refit",
            "smoke": True,
        },
        "report_path": f"/home/aniket/altiseek-m10/{relative}",
        "sha256": source.sha256,
    }
    manifest_path = tmp_path / source.manifest
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest))
    assert verified_search(tmp_path, source)[0] == report
    report_path.write_text(json.dumps({**report, "world_id": "test-world"}))
    with pytest.raises(ValueError, match="checksum/path"):
        verified_search(tmp_path, source)
    source = replace(source, sha256=sha256_file(report_path))
    manifest["sha256"] = source.sha256
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="B measured-search"):
        verified_search(tmp_path, source)
    manifest["report_path"] = "/home/aniket/altiseek-m10/live/other.json"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum/path"):
        verified_search(tmp_path, source)
