import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from altiseek.calibration.model import digest
from altiseek.deploy.transfer import (
    OperationalFailure,
    authorize_world,
    live_cases,
    load_protocol,
    operational_failure,
    paired_seed,
    retry_record,
    run_deferred,
)
from altiseek.deploy.transfer_report import (
    paired,
    rate_difference,
    recovered_fraction,
    world_values,
)
from altiseek.registry.run import sha256_file
from tests.cache.test_cache import built, cfg, library

__all__ = ["built", "cfg", "library"]
SCRIPTS = Path(__file__).resolve().parents[2] / "scripts/deploy"


def test_protocol_rejects_mutation_and_changed_inputs(tmp_path):
    source = tmp_path / "model"
    source.write_text("frozen")
    body = {"status": "frozen", "inputs": {str(source): sha256_file(source)}}
    body["protocol_id"] = digest(body)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(body))
    assert load_protocol(path)["protocol_id"] == body["protocol_id"]
    source.write_text("changed")
    with pytest.raises(ValueError, match="input changed"):
        load_protocol(path)
    body["status"] = "draft"
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="not frozen"):
        load_protocol(path)


def test_scope_and_seeds_are_paired_without_outcomes(built):
    world, _ = built
    protocol = {"b_preflight_world_id": world.world_id, "world_ids": []}
    with pytest.raises(ValueError, match="preflight"):
        authorize_world(protocol, world)  # Fixture is not the frozen rl-val set.
    target = world.model_copy(update={"split": "T", "world_set": "test"}).identified()
    with pytest.raises(ValueError, match="held-out"):
        authorize_world(protocol, target)
    protocol["world_ids"] = [target.world_id]
    authorize_world(protocol, target)
    assert paired_seed(target.world_id, "launch") == paired_seed(target.world_id, "launch")
    assert paired_seed(target.world_id, "actions", 17) != paired_seed(
        target.world_id, "actions", 29
    )


def test_live_cases_cover_every_pairing_once_in_fixed_order():
    worlds = [f"w{i:02d}" for i in range(24)]
    cases = live_cases(worlds)
    assert len(cases) == 192
    assert len({digest(c) for c in cases}) == 192
    assert cases == live_cases(reversed(worlds))
    assert Counter(c["sensor"] for c in cases) == {"original": 96, "refit": 96}
    assert Counter((c["method"], c["training_seed"]) for c in cases)[("PPO", 29)] == 48


@pytest.mark.parametrize(
    ("error", "flight", "cleanup", "expected"),
    [
        ("ValueError('Actual camera missed policy goal by 1.2 m')", {}, True, True),
        ("ValueError('Image/pose mismatch 0.060 s')", {}, True, True),
        ("TimeoutError('Policy waypoint capture timed out')", {}, True, True),
        ("ValueError('Actual camera missed policy goal by 1.2 m')", {}, False, False),
        ("RuntimeError('Flight ended before policy observation')", {}, True, False),
        (
            "RuntimeError('Flight ended before policy observation')",
            {"reason": "Telemetry became stale or invalid"},
            True,
            True,
        ),
        (
            "RuntimeError('Flight is already landing')",
            {"reason": "Telemetry became stale or invalid"},
            True,
            True,
        ),
        ("RuntimeError('Flight is already landing')", {"reason": "Hold complete"}, True, False),
        ("RuntimeError('Flight is already landing')", {}, True, False),
        ("ValueError('Frozen live sensor acquisition contract mismatch')", {}, True, False),
        ("", {}, True, False),
    ],
)
def test_only_recognised_failures_with_cleanup_are_operational(error, flight, cleanup, expected):
    summary = {"passed": False, "error": error}
    assert operational_failure(summary, {"owned_processes_stopped": cleanup}, flight) is expected
    assert not operational_failure({**summary, "passed": True}, {"owned_processes_stopped": 1}, {})


def test_retry_record_serialises_live_meter_snapshots():
    meter = {"elapsed_s": np.float64(32.8), "time_per_level": np.zeros(5), "exhausted": False}
    line = retry_record(1, "Image/pose mismatch 0.060 s", np.array([1.0, 2.0, 3.0]), 2, meter)
    record = json.loads(line)
    assert record["meter"]["time_per_level"] == [0.0] * 5
    assert record["goal"] == [1.0, 2.0, 3.0]
    assert json.loads(retry_record(2, "x", (0, 0, 0), np.int64(4), None))["level"] == 4


def test_recovery_adapter_logs_through_tested_helper():
    source = (SCRIPTS / "search_recovery.py").read_text()
    assert "retry_record(attempt, reason, goal, level, meter)" in source
    assert "json.dumps(record" not in source


def test_deferred_queue_retries_failures_last_and_persists_attempts():
    cases = [{"id": i} for i in range(4)]
    flown, logged = [], []
    failures = {1: 1}

    def attempt(case):
        if failures.get(case["id"], 0):
            failures[case["id"]] -= 1
            raise OperationalFailure("Actual camera missed policy goal")
        flown.append(case["id"])
        return True

    counts = Counter()
    run_deferred(cases, attempt, lambda c, e: logged.append(c["id"]), counts)
    assert flown == [0, 2, 3, 1]
    assert logged == [1]
    assert counts[digest({"id": 1})] == 1


def test_deferred_queue_stops_on_repeats_exhaustion_and_unknown_errors():
    def failing(case):
        raise OperationalFailure("timeout")

    with pytest.raises(RuntimeError, match="consecutive"):
        run_deferred([{"id": i} for i in range(5)], failing, lambda c, e: None, Counter())
    # A relaunch cannot reset an exhausted case's persisted attempt count.
    with pytest.raises(RuntimeError, match="exhausted"):
        run_deferred([{"id": 0}], failing, lambda c, e: None, Counter({digest({"id": 0}): 3}))
    attempts = Counter()
    one = [{"id": 9}]
    with pytest.raises(RuntimeError, match="passes exhausted"):
        run_deferred(one, failing, lambda c, e: None, attempts, consecutive_limit=10)
    assert attempts[digest(one[0])] == 3

    def broken(case):
        raise ValueError("checksum mismatch")

    with pytest.raises(ValueError, match="checksum"):
        run_deferred([{"id": 0}], broken, lambda c, e: None, Counter())
    # Reused completions neither fly nor reset the consecutive count.
    calls = []

    def mixed(case):
        calls.append(case["id"])
        if case["id"] % 2:
            return False
        raise OperationalFailure("timeout")

    with pytest.raises(RuntimeError, match="consecutive"):
        run_deferred([{"id": i} for i in range(6)], mixed, lambda c, e: None, Counter())
    assert calls == [0, 1, 2, 3, 4]


def test_transfer_flight_adapter_replaces_only_search_controller():
    pytest.importorskip("fcntl")
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(
            "transfer_flight", SCRIPTS / "run_m10_transfer_flight.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SCRIPTS))
    search = ["python", str(SCRIPTS / "search_flight_pilot.py"), "run", "--protocol", "p"]
    assert module.adapted_command(search) == [
        "python",
        str(SCRIPTS / "search_recovery.py"),
        "run",
        "--protocol",
        "p",
    ]
    calibration = ["python", str(SCRIPTS / "calibration_flight.py"), "run"]
    assert module.adapted_command(calibration) == calibration


def rows():
    values = {
        ("w1", "B5"): (0.5, 0.3, 0.4),
        ("w2", "B5"): (0.7, 0.4, 0.6),
        ("w1", "PPO-17"): (0.4, 0.2, 0.3),
        ("w1", "PPO-29"): (0.6, 0.2, 0.3),
        ("w1", "PPO-43"): (0.5, 0.2, 0.3),
        ("w2", "PPO-17"): (0.3, 0.3, 0.3),
        ("w2", "PPO-29"): (0.3, 0.3, 0.3),
        ("w2", "PPO-43"): (0.3, 0.3, 0.3),
    }
    out = []
    for (world, group), triple in values.items():
        for condition, auc in zip(("cached", "live-original", "live-refit"), triple, strict=True):
            out.append({"world_id": world, "group": group, "condition": condition, "auc": auc})
    return out


def test_world_values_average_ppo_seeds_within_world():
    assert world_values(rows(), "auc", "PPO", "cached") == pytest.approx({"w1": 0.5, "w2": 0.3})
    assert world_values(rows(), "auc", "B5", "live-refit") == {"w1": 0.4, "w2": 0.6}
    incomplete = [r for r in rows() if r["group"] != "PPO-43"]
    with pytest.raises(ValueError, match="three training seeds"):
        world_values(incomplete, "auc", "PPO", "cached")


def test_paired_differences_and_recovered_fraction():
    data = rows()
    b5 = {c: world_values(data, "auc", "B5", c) for c in ("cached", "live-original", "live-refit")}
    p = paired(b5["live-original"], b5["cached"], 1, 500)
    assert p["difference"] == pytest.approx(-0.25)
    assert p["second_better_worlds"] == 2
    assert p["ci95"][0] <= -0.25 <= p["ci95"][1]
    with pytest.raises(ValueError, match="identical world sets"):
        paired({"w1": 1.0}, b5["cached"], 1, 10)
    r = recovered_fraction(b5["cached"], b5["live-original"], b5["live-refit"], 1, 500)
    assert r["defined"]
    assert r["fraction"] == pytest.approx(0.15 / 0.25)
    flat = {"w1": 0.3, "w2": 0.3}
    undefined = recovered_fraction(flat, flat, b5["live-refit"], 1, 500)
    assert undefined["defined"] is False


def test_rate_difference_pools_hits_within_resampled_worlds():
    counts = {"w1": np.array([3, 4, 4, 4]), "w2": np.array([1, 4, 2, 4])}
    result = rate_difference(counts, 1, 500)
    assert result["live_rate"] == pytest.approx(0.5)
    assert result["cached_rate"] == pytest.approx(0.75)
    low, high = result["difference_ci95"]
    assert low <= -0.25 <= high
    empty = rate_difference({"w1": np.zeros(4, dtype=int)}, 1, 10)
    assert empty["live_rate"] is None
    assert empty["difference_ci95"] is None
