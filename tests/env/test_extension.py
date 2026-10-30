import json

import numpy as np
import pytest

from altiseek.cache.jobs import write_json
from altiseek.calibration.model import digest
from altiseek.env.search import SearchEnv
from altiseek.eval.extension import ExtensionProtocol, keys, play_extension, selected_run
from altiseek.eval.extension_report import aggregate
from altiseek.eval.reporting import METRICS
from altiseek.registry.run import sha256_file
from tests.env.test_search import built, cfg, env, library, sensor

__all__ = ["built", "cfg", "env", "library", "sensor"]


def test_aggregation_averages_draws_before_seeds():
    rows = []
    for seed, values in ((17, [0.0, 0.0, 0.0]), (29, [1.0])):
        for value in values:
            metrics = dict.fromkeys(METRICS, value)
            metrics["time_per_level_s"] = [0] * 5
            rows.append(
                {
                    "key": {
                        "scenario": "control",
                        "variant": "V3",
                        "method": "PPO",
                        "training_seed": seed,
                        "world_id": "w",
                    },
                    "metrics": metrics,
                    "records": [],
                    "target_count": 1,
                    "coarse_detection_opportunities": 0,
                    "immediate_descents": 0,
                }
            )
    per_seed, means = aggregate(rows)
    assert per_seed[("control", "V3", "PPO", 17)]["w"][0] == 0
    assert means[("control", "V3", "PPO")]["w"][0] == 0.5


def test_extension_pair_seeds_and_all_training_seeds():
    def dataset(wid):
        return {"worlds": [{"world_id": wid, "seed": 5000000}], "smoke_worlds": []}

    manifest = {
        "datasets": {
            "control": dataset("t"),
            "stress-low": dataset("low"),
            "stress-high": dataset("high"),
        },
        "ablations": {"control": [{"seed": s} for s in (17, 29, 43)]},
        "historical": {"checkpoints": [{"seed": s, "variant": "V3"} for s in (17, 29, 43)]},
    }
    jobs = keys(ExtensionProtocol(), manifest)
    controls = [k for k in jobs if k["scenario"] == "control"]
    assert len(controls) == 9
    assert {k["training_seed"] for k in controls} == {17, 29, 43}
    low = [k for k in jobs if k["scenario"] == "stress-low"]
    high = [k for k in jobs if k["scenario"] == "stress-high"]
    assert len(low) == len(high)
    for a, b in zip(low, high, strict=True):
        assert a["environment_seed"] == b["environment_seed"]
        assert a["action_seed"] == b["action_seed"]


def test_selection_verifies_earliest_tie_and_checkpoint_bytes(tmp_path):
    plan = {
        "smoke": False,
        "config": {"selection_metric": "mean_auc"},
        "seed": 17,
        "world_ids": {"rl-val": ["b"], "rl-train": ["train"]},
        "search": {},
    }
    root = tmp_path / digest(plan)
    write_json(root / "plan.json", plan)
    for step in range(16384, 131073, 16384):
        checkpoint = root / f"step-{step:09d}"
        checkpoint.mkdir()
        (checkpoint / "model.zip").write_bytes(str(step).encode())
        write_json(
            checkpoint / "complete.json",
            {"steps": step, "hashes": {"model.zip": sha256_file(checkpoint / "model.zip")}},
        )
        write_json(
            root / f"eval-{step:09d}.json",
            {
                "steps": step,
                "zero_belief": False,
                "action_rule": "sampled",
                "repeats": 3,
                "mean_auc": 0.5,
                "episodes": [
                    {"world_id": "b", "repeat": r, "metrics": {"auc": 0.5}} for r in range(3)
                ],
            },
        )
    write_json(
        root / "complete.json",
        {
            "run_id": root.name,
            "steps": 131072,
            "selected_step": 16384,
            "validation": "eval-000016384.json",
            "checkpoint_sha256": sha256_file(root / "step-000016384/model.zip"),
        },
    )
    assert selected_run(root)["selected_step"] == 16384
    (root / "step-000032768/model.zip").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        selected_run(root)


@pytest.mark.parametrize(
    "changes",
    [
        {"known_count": False},
        {"relook_rule": "distinct_variant"},
        {"declaration_mode": "learned"},
        {"observation_frame": "egocentric", "coordinate_channels": False},
    ],
)
def test_extension_episode_adapter_is_deterministic(env, changes):
    import torch

    config = env.config.model_copy(update=changes | {"max_steps": 4})
    test_env = SearchEnv(env._worlds, env.cache_config, env.sensor, config)

    class UniformPolicy:
        def obs_to_tensor(self, obs):
            assert test_env.observation_space.contains(obs)
            return obs, False

        def get_distribution(self, tensors, action_masks):
            class Distribution:
                distribution = torch.distributions.Categorical(
                    probs=torch.tensor(action_masks.astype(float))[None]
                )

            return Distribution()

    class Model:
        policy = UniformPolicy()

    key = {"environment_seed": 17, "action_seed": 29, "method": "PPO", "variant": "V3"}
    one = play_extension(test_env, key, {}, Model())
    two = play_extension(test_env, key, {}, Model())
    assert json.dumps(one, sort_keys=True) == json.dumps(two, sort_keys=True)
    assert 0 <= one["metrics"]["auc"] <= 1
    assert np.isfinite(one["metrics"]["energy_s"])
