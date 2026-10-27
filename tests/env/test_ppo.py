import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from altiseek.policies.ppo import FiniteSearch, PPOConfig, SearchFeatures
from altiseek.policies.training import load_checkpoint, make_model, save_checkpoint
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
def test_checkpoint_resume_matches_uninterrupted_optimizer_and_rollout(env, tmp_path, vector_class):
    config = PPOConfig(n_envs=1, n_steps=8, batch_size=4, epochs=1, total_steps=16, eval_every=8)
    vector = vector_class([lambda: FiniteSearch(env)])
    model = make_model(vector, config, 17)
    model.learn(8)
    saved_obs = {k: v.copy() for k, v in model._last_obs.items()}
    save_checkpoint(model, tmp_path / "checkpoint")
    model.learn(8, reset_num_timesteps=False)
    expected = {k: v.clone() for k, v in model.policy.state_dict().items()}
    expected_obs = {k: v.copy() for k, v in model._last_obs.items()}
    resumed = load_checkpoint(tmp_path / "checkpoint", vector)
    for key in saved_obs:
        np.testing.assert_array_equal(resumed._last_obs[key], saved_obs[key])
    resumed.learn(8, reset_num_timesteps=False)
    assert resumed.num_timesteps == 16
    for key, value in resumed.policy.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    for key in expected_obs:
        np.testing.assert_array_equal(resumed._last_obs[key], expected_obs[key])
    (tmp_path / "checkpoint" / "state.pkl").write_bytes(b"broken")
    with pytest.raises(ValueError, match="checksum"):
        load_checkpoint(tmp_path / "checkpoint", vector)
    vector.close()
