import numpy as np
import pytest
from pydantic import ValidationError

from altiseek.cache.jobs import write_json
from altiseek.calibration.model import digest
from altiseek.env.search import SearchEnv
from altiseek.eval import reporting
from altiseek.eval.config import EvaluationProtocol, Scenario
from altiseek.eval.reporting import curve, interval
from altiseek.eval.runner import (
    episode_keys,
    play,
    save_episode,
    validate_artifacts,
    verified_episode,
)
from altiseek.policies.config import SelectedBaselines
from altiseek.registry.run import sha256_file
from altiseek.world.sampler import prior_grid
from tests.env.test_search import built, cfg, env, library, sensor

__all__ = ["built", "cfg", "env", "library", "sensor"]


def test_prior_override_preserves_world_and_cache_identity(env):
    world = env._worlds[0]
    original = world.model_dump_json()
    env.reset(seed=17)
    scenario = Scenario(name="shift", shift_east_m=2, rotation_deg=90, spread_scale=2)
    prior = scenario.prior(world.prior_given)
    changed = SearchEnv(
        [world], env.cache_config, env.sensor, env.config, pool=env.pool, prior_override=prior
    )
    obs, _ = changed.reset(seed=17)
    expected = prior_grid(prior, world.area_m, world.cell_m)
    np.testing.assert_allclose(obs["map"][1], expected.T[::-1], rtol=1e-6)
    assert not np.array_equal(obs["map"][1], env._observation()["map"][1])
    assert changed.cache.header.cache_id == env.cache.header.cache_id
    assert changed.episode_seed == env.episode_seed
    assert changed.node == env.node
    np.testing.assert_array_equal(changed._targets, env._targets)
    assert world.model_dump_json() == original
    with pytest.raises(ValueError, match="exactly one"):
        SearchEnv([world, world], env.cache_config, env.sensor, env.config, prior_override=prior)


def test_nominal_prior_override_replays_default(env):
    world = env._worlds[0]
    nominal = Scenario(name="nominal").prior(world.prior_given)
    other = SearchEnv([world], env.cache_config, env.sensor, env.config, prior_override=nominal)
    first, _ = env.reset(seed=71)
    second, _ = other.reset(seed=71)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    for _ in range(4):
        action = int(np.flatnonzero(env.action_masks())[0])
        first, reward, terminated, truncated, _ = env.step(action)
        second, other_reward, other_terminated, other_truncated, _ = other.step(action)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        assert (reward, terminated, truncated) == (other_reward, other_terminated, other_truncated)
        if terminated or truncated:
            break


def test_protocol_rejects_ambiguous_or_nonfinite_scenarios():
    with pytest.raises(ValidationError, match="unique"):
        EvaluationProtocol(scenarios=(Scenario(name="same"), Scenario(name="same")))
    for kwargs in ({"spread_scale": 0}, {"shift_east_m": float("nan")}, {"battery_s": 2}):
        with pytest.raises(ValidationError):
            Scenario(name="invalid", **kwargs)


def test_episode_resume_rejects_tampering_and_wrong_identity(tmp_path):
    path = tmp_path / "episode.json"
    key = {"world_id": "one", "repeat": 0}
    result = {"run_id": "run", "key": key, "metrics": {"auc": 0.2}}
    assert verified_episode(path, "run", key) is None
    save_episode(path, result)
    assert verified_episode(path, "run", key) == result
    with pytest.raises(ValueError, match="mismatch"):
        verified_episode(path, "other", key)
    path.write_text(path.read_text().replace("0.2", "0.9"))
    with pytest.raises(ValueError, match="mismatch"):
        verified_episode(path, "run", key)


def test_frozen_artifacts_allow_line_endings_but_reject_checkpoint_change(tmp_path):
    baseline = SelectedBaselines(
        source_run_id="source",
        sensor_model_id="sensor",
        world_set_id="validation",
        implementation="fixed",
        settings={},
    )
    selected_path = tmp_path / "selected.yaml"
    write_json(selected_path, baseline.model_dump(mode="json"))
    checkpoints = []
    for variant in ("V0", "V3"):
        for seed in (17, 29, 43):
            path = tmp_path / f"{variant}-{seed}.zip"
            path.write_bytes(f"{variant}-{seed}".encode())
            checkpoints.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "checkpoint": str(path),
                    "sha256": sha256_file(path),
                }
            )
    selection = {"selected": checkpoints}
    selection_path = tmp_path / "selection.json"
    write_json(selection_path, selection)
    manifest = {
        "schema_version": 1,
        "baselines_id": digest(baseline.model_dump(mode="json")),
        "selection_report": str(selection_path),
        "selection_report_id": digest(selection),
        "checkpoints": checkpoints,
    }
    validate_artifacts(manifest, selected_path)
    selection_path.write_bytes(selection_path.read_bytes().replace(b"\n", b"\r\n"))
    validate_artifacts(manifest, selected_path)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint checksum"):
        validate_artifacts(manifest, selected_path)


def test_paired_jobs_and_random_episode_replay(env):
    manifest = {
        "checkpoints": [{"variant": v, "seed": s} for v in ("V0", "V3") for s in (17, 29, 43)]
    }
    protocol = EvaluationProtocol(scenarios=(Scenario(name="nominal"),))
    keys = episode_keys(protocol, manifest, env._worlds)
    assert len(keys) == 54
    assert len({k["environment_seed"] for k in keys}) == 1
    assert len({k["action_seed"] for k in keys}) == 3
    random_key = next(k for k in keys if k["method"] == "random" and k["variant"] == "V3")
    first = play(env, random_key, {}, None)
    second = play(env, random_key, {}, None)
    assert first == second
    assert 0 <= first["metrics"]["auc"] <= 1


def test_report_curve_includes_initial_find_and_censoring():
    records = [{"time_s": 2, "found": 1}, {"time_s": 50, "found": 2}]
    result = curve(records, 2, 100)
    assert result[0] == 0
    assert result[2] == 0.5
    assert result[49] == 0.5
    assert result[50] == result[-1] == 1
    protocol = EvaluationProtocol(scenarios=(Scenario(name="nominal"),))
    assert interval(np.ones(20) * 0.25, protocol) == [0.25, 0.25]
    differences = np.array([0.1, -0.1, 0.2, -0.2])
    assert interval(differences, protocol) == interval(differences, protocol)


def test_complete_report_averages_repeats_then_training_seeds(env, tmp_path, monkeypatch):
    world = env._worlds[0]
    manifest = {
        "checkpoints": [{"variant": v, "seed": s} for v in ("V0", "V3") for s in (17, 29, 43)],
        "worlds": [{"world_id": world.world_id, "seed": world.seed}],
    }
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, manifest)
    protocol = EvaluationProtocol(scenarios=(Scenario(name="nominal"),))
    plan = {
        "smoke": False,
        "protocol": protocol.model_dump(mode="json"),
        "cache": env.cache_config.model_dump(mode="json"),
        "inputs": {"manifest": sha256_file(manifest_path)},
    }
    root = tmp_path / digest(plan)
    write_json(root / "plan.json", plan)
    write_json(root / "run.meta.json", {"inputs": {"manifest": {"path": str(manifest_path)}}})
    jobs = episode_keys(protocol, manifest, [world])
    for key in jobs:
        method = key["method"]
        auc = {"random": 0.1, "oracle": 0.9, "B5": 0.4}.get(method, 0.2)
        if method == "PPO":
            auc = {17: 0.1, 29: 0.2, 43: 0.3}[key["training_seed"]] + 0.01 * key["repeat"]
        metrics = dict.fromkeys(reporting.METRICS, 0.0)
        metrics.update(auc=auc, time_per_level_s=[1.0] * 5)
        save_episode(
            root / "episodes" / f"{digest(key)}.json",
            {
                "run_id": root.name,
                "key": key,
                "family": world.family,
                "metrics": metrics,
                "records": [{"time_s": 2, "found": 0}],
            },
        )
    monkeypatch.setattr(reporting, "load_worlds", lambda *_: [world])
    monkeypatch.setattr(reporting, "figures", lambda *_: None)
    write_json(
        root / "status.json", {"completed": len(jobs), "total": len(jobs), "complete": False}
    )
    with pytest.raises(ValueError, match="incomplete"):
        reporting.report(root, tmp_path / "report")
    write_json(root / "status.json", {"completed": len(jobs), "total": len(jobs), "complete": True})
    reporting.report(root, tmp_path / "report")
    result = reporting.read(tmp_path / "report/results.json")
    row = next(
        r
        for r in result["rows"]
        if r["method"] == "PPO" and r["variant"] == "V3" and r["family"] == "all"
    )
    assert row["metrics"]["auc"]["mean"] == pytest.approx(0.21)
    assert row["auc_seed_sd"] == pytest.approx(0.1)
    assert row["normalised_auc"] == pytest.approx((0.21 - 0.1) / 0.8)
    paired = next(
        r
        for r in result["paired"]
        if r["method"] == ["nominal", "V3", "PPO"]
        and r["reference"] == ["nominal", "V3", "B5"]
        and r["family"] == "all"
    )
    assert paired["auc_difference"] == pytest.approx(-0.19)
