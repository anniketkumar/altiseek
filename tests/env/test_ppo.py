import json
import pickle
import random

import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from altiseek.env.logging import EpisodeLog
from altiseek.policies import training
from altiseek.policies.ppo import FiniteSearch, PPOConfig, SearchFeatures
from altiseek.policies.training import (
    evaluate,
    load_checkpoint,
    make_model,
    save_checkpoint,
    vector_env,
)
from tests.env.test_search import built, cfg, env, library, sensor

__all__ = ["built", "cfg", "env", "library", "sensor"]


def test_cnn_uses_map_and_zero_ablation_removes_only_belief(env):
    obs, _ = env.reset(seed=7)
    torch.manual_seed(5)
    cnn = SearchFeatures(env.observation_space, zero_belief=True)
    tensors = {k: torch.from_numpy(v[None]) for k, v in obs.items()}
    expected = cnn(tensors)
    changed = {k: v.clone() for k, v in tensors.items()}
    changed["map"][:, 0] = 0.75
    torch.testing.assert_close(expected, cnn(changed), rtol=0, atol=0)
    changed["map"][:, 1] = 0.5
    assert not torch.equal(expected, cnn(changed))
    expected.sum().backward()
    assert cnn.cnn[0].weight.grad.abs().sum() > 0
    assert expected.shape == (1, 128)
    np.testing.assert_array_equal(tensors["map"].numpy()[0], obs["map"])


def test_finite_battery_has_no_timeout_bootstrap(env):
    adapted = FiniteSearch(env)
    adapted.reset(seed=1)
    env.energy_used_s = env.config.energy.battery_s - 0.01
    action = int(np.flatnonzero(adapted.action_masks())[0])
    _, _, terminated, truncated, _ = adapted.step(action)
    assert terminated
    assert not truncated


@pytest.mark.parametrize("vector_class", [DummyVecEnv, SubprocVecEnv])
@pytest.mark.parametrize("objective", ["legacy", "auc"])
def test_checkpoint_resume_matches_uninterrupted_optimizer_and_rollout(
    env, tmp_path, vector_class, objective
):
    workers = 2 if vector_class is SubprocVecEnv else 1
    rollout = 8 * workers
    config = PPOConfig(
        objective=objective,
        gamma=1 if objective == "auc" else 0.99,
        n_envs=workers,
        n_steps=8,
        batch_size=4,
        epochs=1,
        threads=workers,
        total_steps=2 * rollout,
        eval_every=rollout,
    )
    vector = vector_class([lambda: FiniteSearch(env, objective) for _ in range(workers)])
    model = make_model(vector, config, 17)
    model.learn(rollout)
    saved_obs = {k: v.copy() for k, v in model._last_obs.items()}
    save_checkpoint(model, tmp_path / "checkpoint")
    model.learn(rollout, reset_num_timesteps=False)
    expected = {k: v.clone() for k, v in model.policy.state_dict().items()}
    expected_obs = {k: v.copy() for k, v in model._last_obs.items()}
    resumed = load_checkpoint(tmp_path / "checkpoint", vector)
    for key in saved_obs:
        np.testing.assert_array_equal(resumed._last_obs[key], saved_obs[key])
    resumed.learn(rollout, reset_num_timesteps=False)
    assert resumed.num_timesteps == 2 * rollout
    for key, value in resumed.policy.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    for key in expected_obs:
        np.testing.assert_array_equal(resumed._last_obs[key], expected_obs[key])
    (tmp_path / "checkpoint" / "state.pkl").write_bytes(b"broken")
    with pytest.raises(ValueError, match="checksum"):
        load_checkpoint(tmp_path / "checkpoint", vector)
    vector.close()


def test_auc_reward_matches_logged_integral_without_false_cost(env):
    log = EpisodeLog(env)
    adapted = FiniteSearch(log, "auc")
    _, info = adapted.reset(seed=7)
    initial = info["found"] / len(env.found) * (1 - info["time_s"] / env.config.energy.battery_s)
    assert not info["ended"]
    # Force declarations in the real transition: one true and many false.
    env.belief.log_odds[:] = 20
    action = int(np.flatnonzero(adapted.action_masks())[0])
    _, reward, terminated, truncated, info = adapted.step(action)
    assert terminated
    assert not truncated
    assert info["false_declarations"] > 0
    assert reward > 0
    assert reward + initial == pytest.approx(log.summary()["auc"])
    assert log.summary()["reward"] < reward


def test_auc_configuration_and_checkpoint_guards(env):
    with pytest.raises(ValueError, match="gamma=1"):
        PPOConfig(objective="auc")
    with pytest.raises(ValueError, match="one repeat"):
        PPOConfig(evaluation_repeats=3)
    legacy = FiniteSearch(env)
    legacy.reset(seed=1)
    with pytest.raises(ValueError, match="objective mismatch"):
        FiniteSearch(env, "auc").restore(legacy.snapshot())
    env.config = env.config.model_copy(update={"known_count": False})
    with pytest.raises(ValueError, match="known K"):
        FiniteSearch(env, "auc")


def test_auc_initial_completion_is_not_credited_to_an_action(env):
    env.config = env.config.model_copy(update={"declare_threshold": 1e-12})
    log = EpisodeLog(env)
    adapted = FiniteSearch(log, "auc")
    _, info = adapted.reset(seed=7)
    assert info["ended"]
    expected = log.summary()["auc"]
    assert expected > 0
    action = int(np.flatnonzero(adapted.action_masks())[0])
    _, reward, terminated, truncated, _ = adapted.step(action)
    assert terminated
    assert not truncated
    assert reward == 0
    assert log.summary()["auc"] == expected


def test_sampled_evaluation_is_repeatable_and_preserves_training_rng(env, tmp_path, monkeypatch):
    world = env._worlds[0].model_copy(update={"world_set": "rl-val"})
    world = world.model_copy(update={"world_id": world.content_id()})
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "family_results": [
                    {
                        "family": world.family,
                        "variant": env.config.sensor_variant,
                        "method": method,
                        "auc": auc,
                    }
                    for method, auc in (("random", 0), ("oracle", 1))
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(training, "BASELINE_PATH", baseline)
    monkeypatch.setattr(training, "SearchEnv", lambda *args, **kwargs: env)
    vector = DummyVecEnv([lambda: FiniteSearch(env, "auc")])
    model = make_model(vector, PPOConfig(objective="auc", gamma=1), 17)
    before_torch = torch.get_rng_state().clone()
    before_python = random.getstate()
    before_numpy = pickle.dumps(np.random.get_state())  # noqa: NPY002
    first = evaluate(
        model, [world], env.cache_config, env.sensor, env.config, action_rule="sampled", repeats=3
    )
    second = evaluate(
        model, [world], env.cache_config, env.sensor, env.config, action_rule="sampled", repeats=3
    )
    assert first == second
    assert len(first["episodes"]) == 3
    torch.testing.assert_close(torch.get_rng_state(), before_torch, rtol=0, atol=0)
    assert random.getstate() == before_python
    assert pickle.dumps(np.random.get_state()) == before_numpy  # noqa: NPY002
    vector.close()


def test_training_and_validation_reject_wrong_world_sets(env):
    wrong = env._worlds[0].model_copy(update={"world_set": "test"})
    with pytest.raises(ValueError, match="rl-train only"):
        vector_env([wrong], env.cache_config, env.sensor, env.config, PPOConfig(), 17)
    with pytest.raises(ValueError, match="rl-val only"):
        evaluate(None, [wrong], env.cache_config, env.sensor, env.config)
