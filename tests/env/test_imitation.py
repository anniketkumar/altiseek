"""D66 leakage guards, masked supervision and warm-start continuation."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from altiseek.cache.build import header_for
from altiseek.calibration.model import digest
from altiseek.config import load_config
from altiseek.policies.imitation import (
    ARRAY_FIELDS,
    collect_episode,
    imitation_loss,
    load_shard,
    load_weights,
    returns_to_go,
    save_weights,
    split_worlds,
)
from altiseek.policies.ppo import FiniteSearch, PPOConfig
from altiseek.policies.training import load_checkpoint, make_model, save_checkpoint
from altiseek.registry.run import sha256_file
from tests.env.test_search import built, cfg, env, library, sensor
from tests.test_repo_configs import CONFIG_DIR, SCHEMAS

__all__ = ["built", "cfg", "env", "library", "sensor"]


def test_existing_configs_keep_their_canonical_identities():
    before = json.loads(
        Path("docs/m12-ppo-revisit/config-identities-before-phase2.json").read_text()
    )
    for name, expected in before["configs"].items():
        relative = Path(name)
        schema = SCHEMAS.get(relative.with_suffix("").as_posix()) or SCHEMAS[relative.parts[0]]
        assert (
            digest(load_config(CONFIG_DIR / name, schema).model_dump(mode="json"))
            == expected["resolved_id"]
        )


def test_world_split_is_stratified_disjoint_and_excludes_validation():
    worlds = [
        SimpleNamespace(world_id=f"{family}-{i}", split="B", world_set="rl-train", family=family)
        for family in ("a", "b")
        for i in range(20)
    ]
    split = split_worlds(worlds)
    assert len(split["holdout"]) == 4
    assert len(split["train"]) == 36
    assert not set(split["holdout"]) & set(split["train"])
    assert split == split_worlds(list(reversed(worlds)))
    worlds[0].world_set = "rl-val"
    with pytest.raises(ValueError, match="rl-train B"):
        split_worlds(worlds)


def test_collection_exposes_only_public_fields_and_correct_returns(env, tmp_path):
    env.reset(seed=7)
    world = env.world.model_copy(update={"world_set": "rl-train"}).identified()
    cache = env.cache
    cache.header = header_for(world, env.cache_config)
    cache.save(env.cache_config.out_dir / "rl-train" / f"{world.world_id}.npz")
    env._worlds = [world]

    class PublicExpert:
        def act(self, observation, mask):
            assert set(observation) == {"map", "vector"}
            return int(np.flatnonzero(mask)[0])

    arrays, metadata = collect_episode(env, PublicExpert(), 17)
    assert set(arrays) == ARRAY_FIELDS
    assert metadata["action_reward_sum"] + metadata["initial_contribution"] == pytest.approx(
        metadata["auc"]
    )
    assert arrays["return"] == pytest.approx(returns_to_go(arrays["reward"]))
    path = tmp_path / "public.npz"
    np.savez_compressed(path, **arrays)
    assert set(load_shard(path, sha256_file(path))) == ARRAY_FIELDS
    np.savez_compressed(path, **arrays, target_locations=np.zeros((1, 2)))
    with pytest.raises(ValueError, match="Prohibited"):
        load_shard(path, sha256_file(path))
    env._worlds = [world.model_copy(update={"split": "T"})]
    with pytest.raises(ValueError, match="rl-train B"):
        collect_episode(env, PublicExpert(), 17)


def make_test_model(env):
    vector = DummyVecEnv([lambda: FiniteSearch(env, "auc")])
    cfg = PPOConfig(
        objective="auc",
        gamma=1,
        n_envs=1,
        n_steps=8,
        batch_size=4,
        epochs=1,
        total_steps=16,
        eval_every=8,
        threads=1,
    )
    return make_model(vector, cfg, 17), vector


def test_masked_loss_has_zero_invalid_probability_and_trains_critic(env):
    model, vector = make_test_model(env)
    obs, _ = env.reset(seed=7)
    mask = np.array([[True, False, True, False, False, False]])
    batch = {
        **{key: value[None] for key, value in obs.items()},
        "mask": mask,
        "action": np.array([0]),
        "return": np.array([0.9], dtype=np.float32),
    }
    tensors = {key: torch.as_tensor(batch[key]) for key in ("map", "vector")}
    probabilities = model.policy.get_distribution(tensors, action_masks=mask).distribution.probs
    assert torch.equal(probabilities[0, ~torch.as_tensor(mask[0])], torch.zeros(4))
    loss, stats = imitation_loss(model, batch)
    loss.backward()
    assert model.policy.value_net.weight.grad.abs().sum() > 0
    assert model.policy.action_net.weight.grad.abs().sum() > 0
    assert stats["critic_mse"] > 0
    batch["action"] = np.array([1])
    with pytest.raises(ValueError, match="Masked action"):
        imitation_loss(model, batch)
    vector.close()


def test_warm_start_has_fresh_optimizer_and_exact_rollout_resume(env, tmp_path):
    model, vector = make_test_model(env)
    model.learn(8)
    save_weights(model, tmp_path / "clone", {"pretrain_id": "synthetic"})
    fresh = make_model(
        vector,
        PPOConfig(
            objective="auc",
            gamma=1,
            n_envs=1,
            n_steps=8,
            batch_size=4,
            epochs=1,
            total_steps=16,
            eval_every=8,
            threads=1,
        ),
        17,
    )
    assert not fresh.policy.optimizer.state
    load_weights(fresh, tmp_path / "clone")
    assert not fresh.policy.optimizer.state
    for key, value in model.policy.state_dict().items():
        torch.testing.assert_close(value, fresh.policy.state_dict()[key], rtol=0, atol=0)
    fresh.learn(8)
    save_checkpoint(fresh, tmp_path / "checkpoint")
    fresh.learn(8, reset_num_timesteps=False)
    expected = {key: value.clone() for key, value in fresh.policy.state_dict().items()}
    resumed = load_checkpoint(tmp_path / "checkpoint", vector)
    resumed.learn(8, reset_num_timesteps=False)
    for key, value in resumed.policy.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    (tmp_path / "clone/policy.pt").write_bytes(b"broken")
    with pytest.raises(ValueError, match="mismatch"):
        load_weights(fresh, tmp_path / "clone")
    vector.close()
