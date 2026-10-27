import numpy as np
import pytest

from altiseek.config import load_config
from altiseek.env.logging import EpisodeLog
from altiseek.env.navigation import Navigation
from altiseek.geometry.levels import LevelTable
from altiseek.policies.baselines import Baseline, conditional_table
from tests.env.test_search import built, cfg, env, library, sensor

__all__ = ["built", "cfg", "env", "library", "sensor"]


@pytest.mark.parametrize("name", ["random", "oracle", "B1", "B2", "B4", "B5"])
def test_baselines_only_choose_valid_moves(env, name):
    logger = EpisodeLog(env)
    obs, info = logger.reset(seed=17)
    policy = Baseline(
        name,
        env.nav,
        env.sensor,
        "V3",
        seed=17,
        parameters={"level": 0},
        oracle_cells=env._target_cells if name == "oracle" else None,
    )
    while not info["ended"]:
        action = policy.act(obs, info["action_mask"])
        assert info["action_mask"][action]
        obs, _, _, _, info = logger.step(action)
    result = logger.summary()
    assert 0 <= result["auc"] <= 1
    assert result["energy_s"] <= env.config.energy.battery_s + 1e-9


def test_normal_policy_rejects_truth(env):
    with pytest.raises(ValueError, match="only oracle"):
        Baseline("B1", env.nav, env.sensor, "V3", seed=1, oracle_cells=np.zeros((1, 2)))


def test_auc_includes_initial_and_terminal_plateau(env):
    logger = EpisodeLog(env)
    logger.reset(seed=1)
    logger.records = [{"time_s": 2, "found": 0}, {"time_s": 25, "found": 1}]
    env.found[:] = True
    assert logger.summary()["auc"] == pytest.approx(0.75)
    logger.records = [{"time_s": 2, "found": 1}]
    assert logger.summary()["auc"] == pytest.approx(0.98)


def test_second_information_is_less_than_first_at_half_prior(sensor):
    grid, conditional = conditional_table(sensor.model_dump_json(), "V2", 0, 0)
    from altiseek.belief.information import information

    first = information(np.array([0.5]), sensor, 0, "V2")[0][0]
    assert 0 < np.interp(0, grid, conditional) < first


@pytest.mark.parametrize("name", ["B1", "B2", "B3", "oracle"])
def test_same_level_travel_never_climbs_for_shortcut(env, name):
    table = load_config("configs/levels/default.yaml", LevelTable)
    nav = Navigation(table, 80, env.config.energy.model_copy(update={"vertical_speed_mps": 100}))
    policy = Baseline(
        name,
        nav,
        env.sensor,
        "V3",
        seed=1,
        oracle_cells=np.array([[100, 100]]) if name == "oracle" else None,
    )
    node, goal = 0, 120  # opposite L1 corners; make a climb shortcut attractive
    probe, left_level = node, False
    while probe != goal:
        probe = int(nav.next[probe, nav.first_actions[probe, goal]])
        left_level |= nav.nodes[probe].level != 0
    assert left_level
    while node != goal:
        action = policy._toward(node, goal)
        assert action < 4
        node = int(nav.next[node, action])
        assert nav.nodes[node].level == 0
