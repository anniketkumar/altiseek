"""Saved-result audits and paired reporting must not turn bad data into a positive finding."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from altiseek.calibration.model import digest
from altiseek.env.config import SearchConfig
from altiseek.env.energy import EnergyModel
from altiseek.eval.baseline_review import (
    METHODS,
    audit_episode,
    candidates,
    choose,
    summarise,
    verify_grid,
)
from altiseek.policies.config import BaselineRun
from altiseek.registry.run import sha256_file

CONFIG = SearchConfig(
    energy=EnergyModel(
        horizontal_speed_mps=10,
        vertical_speed_mps=3,
        dwell_s=2,
        climb_multiplier=1.5,
        battery_s=500,
    )
)


def metrics(auc=0.996):
    return {
        "steps": 0,
        "auc": auc,
        "found_fraction": 1.0,
        "elapsed_s": 2.0,
        "energy_s": 2.0,
        "success": True,
        "first_find_s": 2.0,
        "false_declarations": 0,
        "time_per_level_s": [0, 0, 0, 0, 2.0],
    }


def record():
    return {"action": None, "found": 1, "false_declarations": 0, "time_s": 2, "energy_s": 2}


def test_auc_reconstructed_from_saved_finds_and_bad_endings_rejected():
    row = {"metrics": metrics(), "records": [record()]}
    audit_episode(row, 1, CONFIG)
    row["metrics"]["auc"] = 0.999  # plausible-looking improvement inconsistent with timing
    with pytest.raises(ValueError, match="disagree"):
        audit_episode(row, 1, CONFIG)
    row["metrics"] = {**metrics(), "success": False, "found_fraction": 0, "auc": 0}
    row["records"][0]["found"] = 0
    with pytest.raises(ValueError, match="ending"):
        audit_episode(row, 1, CONFIG)


@pytest.mark.parametrize(
    ("field", "value"), [("energy_s", 501), ("auc", float("nan")), ("steps", 20)]
)
def test_invalid_metrics_cannot_pass_review(field, value):
    row = {"metrics": {**metrics(), field: value}, "records": [record()]}
    with pytest.raises(ValueError, match=r"metrics|Non-finite"):
        audit_episode(row, 1, CONFIG)


def test_selection_keeps_declared_false_declaration_and_canonical_ties():
    rows = [
        {
            "key": {"parameters": {"level": level}},
            "metrics": {"auc": auc, "false_declarations": false},
        }
        for level, auc, false in [(0, 0.9, 4), (1, 0.9, 1), (2, 0.9, 1), (3, 0.8, 0)]
    ]
    assert choose(rows) == {"level": 1}
    assert choose(list(reversed(rows))) == {"level": 1}


def test_paired_bootstrap_preserves_negative_differences_and_world_order():
    rows, old = [], {}
    for world, base in (("z", 0.5), ("a", 0.3), ("m", 0.4)):
        for method in METHODS:
            key = {
                "world_id": world,
                "variant": "V3",
                "method": method,
                "parameters": {},
                "seed": 5,
            }
            auc = base + (0.3 if method == "oracle" else -0.2 if method == "random" else 0)
            rows.append({"key": key, "family": "f", "metrics": metrics(auc)})
            if method == "B5":
                old[digest(key)] = {"key": key, "metrics": metrics(auc + 0.1)}
    result = summarise(rows, old, ("V3",))
    pair = result["b5_paired"][0]
    assert pair["delta_auc"] == pytest.approx(-0.1)
    assert pair["delta_auc_ci95"] == pytest.approx([-0.1, -0.1])
    assert [r["world_id"] for r in pair["per_world"]] == ["a", "m", "z"]
    assert result == summarise(list(reversed(rows)), old, ("V3",))
    assert result["sanity_passed"]
    for row in rows:
        if row["key"]["method"] == "oracle":
            row["metrics"]["auc"] = 0
    assert not summarise(rows, old, ("V3",))["sanity_passed"]


def test_test_world_is_rejected_before_reading_any_artifact(tmp_path):
    world = SimpleNamespace(split="T", world_set="test")
    with pytest.raises(ValueError, match="rl-val B"):
        verify_grid(tmp_path, [world], CONFIG, BaselineRun(), [], "")


def test_full_key_grid_and_run_metadata_verified(tmp_path):
    grid = BaselineRun(variants=("V3",))
    world = SimpleNamespace(
        split="B", world_set="rl-val", world_id="world", seed=123, family="f", targets=[None]
    )
    plan = {
        "schema": 1,
        "implementation": "m7-v2-fixed-level-routing",
        "smoke": False,
        "world_set_id": "40f07c646a97efba",
        "config": grid.model_dump(mode="json"),
        "search": CONFIG.model_dump(mode="json"),
        "cache_ids": ["cache"],
        "model_sha256": "sensor",
    }
    root = tmp_path / digest(plan)
    root.mkdir()
    (root / "plan.json").write_text(json.dumps(plan))
    names = []
    for method, params in candidates(grid):
        key = {
            "world_id": "world",
            "variant": "V3",
            "method": method,
            "parameters": params,
            "seed": grid.seed + world.seed,
        }
        name = f"{digest(key)}.json"
        names.append(name)
        (root / name).write_text(
            json.dumps(
                {
                    "key": key,
                    "run_id": root.name,
                    "family": "f",
                    "metrics": metrics(),
                    "records": [record()],
                }
            )
        )
    completion = {"run_id": root.name, "files": names, "episodes": len(names)}
    (root / "complete.json").write_text(json.dumps(completion))
    metadata = {
        "outputs": {
            name: {"sha256": sha256_file(root / filename)}
            for name, filename in (("plan", "plan.json"), ("completion", "complete.json"))
        }
    }
    (root / "run.meta.json").write_text(json.dumps(metadata))
    rows, verification = verify_grid(root, [world], CONFIG, grid, ["cache"], "sensor")
    assert len(rows) == len(verification["episode_sha256"]) == 22
    broken = copy.deepcopy(completion)
    broken["files"][-1] = broken["files"][0]
    (root / "complete.json").write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="duplicate baseline grid"):
        verify_grid(root, [world], CONFIG, grid, ["cache"], "sensor")


def test_world_bootstrap_matches_paired_calculation():
    from altiseek.eval.baseline_review import interval

    new = np.array([0.2, 0.8, 0.3])
    old = np.array([0.7, 0.4, 0.5])
    draws = np.array([[0, 0, 1], [2, 1, 1], [2, 2, 0]])
    expected = np.quantile((new[draws] - old[draws]).mean(axis=1), [0.025, 0.975])
    assert interval(new - old, draws) == pytest.approx(expected)
