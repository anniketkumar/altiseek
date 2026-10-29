import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from altiseek.belief.state import Belief, entropy
from altiseek.cache.storage import DETECTION_DTYPE
from altiseek.env.logging import EpisodeLog
from altiseek.env.search import SearchEnv
from altiseek.policies.ppo import FiniteSearch, PPOConfig
from altiseek.policies.training import training_worlds
from tests.env.test_search import built, cfg, env, library, sensor

__all__ = ["built", "cfg", "env", "library", "sensor"]


def altered(env, **changes):
    return SearchEnv(
        env._worlds,
        env.cache_config,
        env.sensor,
        env.config.model_copy(update=changes),
        pool=env.pool,
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"relook_rule": "distinct_variant"},
        {"declaration_mode": "learned"},
        {"known_count": False},
        {"shaping_weight": 0.01, "discount": 1.0},
        {"observation_frame": "egocentric", "coordinate_channels": False},
    ],
)
def test_ablation_gym_contract(env, changes):
    check_env(altered(env, **changes), skip_render_check=True)


def test_distinct_variants_update_only_once_each(env):
    env.reset(seed=17)
    belief = Belief(env.belief.prior, 2, env.level_count, env.sensor, "distinct_variant")
    detections = np.empty(0, dtype=DETECTION_DTYPE)
    bounds = (0, 0, 2, 2)
    args = (detections, bounds, 0)
    following = ("V3", np.empty((0, 2)), 1.0)
    assert belief.update(*args, 0, *following) == 4
    first = belief.log_odds.copy()
    assert belief.update(*args, 0, *following) == 0
    np.testing.assert_array_equal(first, belief.log_odds)
    assert belief.update(*args, 1, *following) == 4
    assert np.all(belief.log_odds[:2, :2] < first[:2, :2])
    belief.resolve(np.array([[0, 0]]))
    assert belief.update(*args, 2, *following) == 3


def test_learned_declaration_costs_time_and_resolves_one_cell(env):
    core = altered(env, declaration_mode="learned", declare_threshold=1e-12)
    _, info = core.reset(seed=7)
    assert info["found"] == info["false_declarations"] == 0
    assert core.action_space.n == 7
    assert core.action_masks()[6]
    x0, y0, x1, y1 = core.footprint
    core.belief.log_odds[:] = -20
    core.belief.log_odds[x0, y0] = 20
    before = (core.elapsed_s, core.energy_used_s, core.observations, core.node)
    core.step(6)
    assert core.belief.resolved.sum() == 1
    assert core.belief.resolved[x0, y0]
    assert core.elapsed_s == before[0] + core.config.energy.dwell_s
    assert core.energy_used_s == before[1] + core.config.energy.dwell_s
    assert (core.observations, core.node) == before[2:]
    core.belief.resolved[x0:x1, y0:y1] = True
    assert not core.action_masks()[6]


def test_egocentric_preserves_full_prior_and_centres_drone(env):
    other = altered(env, observation_frame="egocentric", coordinate_channels=False)
    original, _ = env.reset(seed=17)
    centred, _ = other.reset(seed=17)
    assert centred["map"].shape == (other.level_count + 5, 2 * env.grid - 1, 2 * env.grid - 1)
    position = centred["map"][other.level_count + 4]
    assert position.sum() == position[env.grid - 1, env.grid - 1] == 1
    for channel in range(other.level_count + 5):
        assert centred["map"][channel].sum() == pytest.approx(original["map"][channel].sum())
    np.testing.assert_array_equal(centred["vector"], original["vector"])


def test_unknown_count_observation_does_not_expose_unfound_count(env):
    core = altered(env, known_count=False)
    original, _ = core.reset(seed=17)
    assert original["vector"][-1] == 0
    core.found = np.zeros(len(core.found) + 1, dtype=bool)
    changed = core._observation()
    for key in original:
        np.testing.assert_array_equal(original[key], changed[key])


def test_auc_shaping_telescopes_at_finite_terminal(env):
    core = altered(env, shaping_weight=0.01, discount=1.0, max_steps=3, known_count=False)
    logger = EpisodeLog(core)
    wrapper = FiniteSearch(logger, "auc")
    _, info = wrapper.reset(seed=17)
    initial_auc = (
        info["found"] / len(core.found) * (1 - info["time_s"] / core.config.energy.battery_s)
    )
    initial_potential = -float(entropy(core.belief.probabilities).sum())
    total = 0.0
    while not info["ended"]:
        _, reward, _, _, info = wrapper.step(int(np.flatnonzero(core.action_masks())[0]))
        total += reward
    assert total + initial_auc == pytest.approx(
        logger.summary()["auc"] - core.config.shaping_weight * initial_potential
    )


def test_appearance_selection_matches_family_counts_and_rejects_test(env):
    template = env._worlds[0]
    worlds = {
        name: [
            template.model_copy(
                update={
                    "world_set": name,
                    "split": "B",
                    "seed": i,
                    "family": f"family-{i // 4}",
                    "condition": i % 3,
                }
            )
            for i in range(8)
        ]
        for name in ("rl-train", "rl-val")
    }
    holdout = training_worlds(
        worlds, PPOConfig(training_conditions=(0,), validation_conditions=(0,))
    )
    assert all(w.condition == 0 for specs in holdout.values() for w in specs)
    config = PPOConfig(match_training_condition_counts=0, validation_conditions=(0,))
    matched = training_worlds(worlds, config)
    assert matched == training_worlds(worlds, config)
    assert len(matched["rl-train"]) == len(holdout["rl-train"])
    for family in {w.family for w in worlds["rl-train"]}:
        assert sum(w.family == family for w in matched["rl-train"]) == sum(
            w.family == family for w in holdout["rl-train"]
        )
    assert matched["rl-val"] == holdout["rl-val"]
    worlds["rl-val"][0] = template.model_copy(update={"world_set": "test", "split": "T"})
    with pytest.raises(ValueError, match="B train/validation"):
        training_worlds(worlds, config)
