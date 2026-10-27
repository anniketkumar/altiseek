import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from altiseek.belief.information import information
from altiseek.belief.state import Belief
from altiseek.cache.storage import DETECTION_DTYPE
from altiseek.calibration.config import SensorConfig
from altiseek.calibration.model import BinaryRates, LevelModel, SensorModel
from altiseek.env.config import SearchConfig
from altiseek.env.energy import EnergyModel
from altiseek.env.navigation import Navigation
from altiseek.env.search import SearchEnv
from altiseek.geometry.levels import Level, LevelTable
from tests.cache.test_cache import built, cfg, library

__all__ = ["built", "cfg", "library"]


@pytest.fixture
def sensor(built):
    _, cache = built
    rates = BinaryRates(recall=0.8, false_alarm=0.1)
    lm = LevelModel(
        name="L1",
        binary=rates,
        soft=rates,
        temperature=1,
        occupied_hist=(0.2, 0.8),
        empty_hist=(0.8, 0.2),
    )
    model = SensorModel(
        detector_id="fake",
        fit_world_set_id="fit",
        fit_cache_ids=("cache",),
        cache_contract=cache.header.model_dump(mode="json", exclude={"world_id", "cache_id"}),
        fit_config=SensorConfig(histogram_bins=2),
        global_binary=rates,
        levels=(lm,),
    )
    return model.model_copy(update={"sensor_model_id": model.content_id()})


@pytest.fixture
def env(tmp_path, built, cfg, sensor):
    spec, cache = built
    cfg = cfg.model_copy(update={"out_dir": tmp_path / "caches"})
    cache.save(cfg.out_dir / spec.world_set / f"{spec.world_id}.npz")
    energy = EnergyModel(
        horizontal_speed_mps=10,
        vertical_speed_mps=3,
        dwell_s=2,
        climb_multiplier=1.5,
        battery_s=100,
    )
    config = SearchConfig(energy=energy, start_level=0, declare_threshold=0.99)
    return SearchEnv([spec], cfg, sensor, config)


def test_gym_checker_and_replay(env):
    check_env(env, skip_render_check=True)
    trajectories = []
    for _ in range(2):
        obs, info = env.reset(seed=71)
        records = [(obs["map"].copy(), info["energy_s"])]
        for _ in range(8):
            action = int(np.flatnonzero(env.action_masks())[0])
            obs, _, terminated, truncated, info = env.step(action)
            records.append((obs["map"].copy(), info["energy_s"]))
            if terminated or truncated:
                break
        trajectories.append(records)
    assert len(trajectories[0]) == len(trajectories[1])
    for (a, e), (b, f) in zip(*trajectories, strict=True):
        np.testing.assert_array_equal(a, b)
        assert e == f


def test_mask_energy_noop_and_budget(env):
    obs, _ = env.reset(seed=1, options={"start_node": 0})
    assert env.action_masks().tolist() == [True, True, False, False, False, False]
    unchanged, reward, _, _, _ = env.step(2)
    np.testing.assert_array_equal(obs["map"], unchanged["map"])
    assert reward == 0
    assert env.observations == 1
    previous = env.energy_used_s
    cost = env.nav.cost[env.node, 0]
    env.step(0)
    assert env.energy_used_s == previous + cost
    env.energy_used_s = env.config.energy.battery_s - 0.01
    count, node = env.observations, env.node
    _, _, terminated, truncated, _ = env.step(int(np.flatnonzero(env.action_masks())[0]))
    assert not terminated
    assert truncated
    assert env.observations == count
    assert env.node == node
    assert env.energy_used_s == env.config.energy.battery_s


def test_observation_has_no_hidden_locations_and_is_not_reused(env):
    obs, _ = env.reset(seed=3)
    env._targets[:] = 0
    again = env._observation()
    np.testing.assert_array_equal(obs["map"], again["map"])
    np.testing.assert_array_equal(obs["vector"], again["vector"])
    obs["map"][:] = 0
    assert env._observation()["map"].sum() > 0
    # x increases to the right, y increases toward the top of a north-up tensor.
    coords = again["map"][-2:]
    assert coords[0, 0, 0] < coords[0, 0, -1]
    assert coords[1, 0, 0] > coords[1, -1, 0]


def test_first_look_resolution_and_suppression(sensor):
    belief = Belief(np.full((2, 2), 0.25), 1, 1, sensor)
    det = np.zeros(1, dtype=DETECTION_DTYPE)
    det["score"], det["x_m"], det["y_m"] = 0.9, 1, 1
    n = belief.update(det, (0, 0, 2, 2), 0, 0, "V2", np.empty((0, 2)), 1)
    assert n == 4
    first = belief.probabilities.copy()
    assert first[0, 0] > 0.25
    assert first[1, 1] < 0.25
    assert belief.update(det, (0, 0, 2, 2), 0, 1, "V2", np.empty((0, 2)), 1) == 0
    np.testing.assert_array_equal(first, belief.probabilities)
    assert (belief.used_variants == 3).all()
    belief.resolve(np.array([[0, 0]]))
    belief.observed[:] = False
    belief.update(det, (0, 0, 2, 2), 0, 2, "V2", np.empty((0, 2)), 1)
    assert belief.probabilities[0, 0] == 0
    suppressed = Belief(np.full((2, 2), 0.25), 1, 1, sensor)
    suppressed.update(det, (0, 0, 2, 2), 0, 0, "V2", np.array([[1, 1]]), 1)
    assert suppressed.probabilities[0, 0] < 0.25


def test_declaration_resolves_and_terminates(env):
    env.reset(seed=2)
    ix, iy = env._target_cells[0]
    env.belief.log_odds[ix, iy] = 20
    # The current footprint was already consumed, so observing it cannot undo this evidence.
    result = env._observe()
    assert result["found"] == 1
    assert env.found.all()
    assert env.belief.resolved[ix, iy]
    assert env._observe()["found"] == 0


def test_multilevel_snap_charges_horizontal_and_climb():
    table = LevelTable(
        cell_m=2.5,
        grid_cells=24,
        levels=(
            Level(name="L1", footprint_cells=8, stride_cells=4),
            Level(name="L2", footprint_cells=12, stride_cells=6),
        ),
    )
    energy = EnergyModel(
        horizontal_speed_mps=10,
        vertical_speed_mps=3,
        dwell_s=2,
        climb_multiplier=1.5,
        battery_s=100,
    )
    nav = Navigation(table, 80, energy)
    target = nav.next[0, 4]
    assert nav.nodes[target].level == 1
    dx, dy = nav.positions[target] - nav.positions[0]
    _, cost = energy.move(dx, dy, nav.altitudes[1] - nav.altitudes[0])
    assert nav.cost[0, 4] == pytest.approx(cost + energy.dwell_s)
    assert nav.distance_m[0, 4] > 0
    assert np.isfinite(nav.distances).all()


def test_initial_completion_has_a_terminal_transition(env):
    env.config = env.config.model_copy(update={"declare_threshold": 1e-10})
    _, info = env.reset(seed=2)
    assert info["ended"]
    observations, energy = env.observations, env.energy_used_s
    _, reward, terminated, truncated, _ = env.step(int(np.flatnonzero(env.action_masks())[0]))
    assert terminated
    assert not truncated
    assert reward == 0
    assert env.energy_used_s == energy
    assert env.observations == observations
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)


@pytest.mark.parametrize("variant", ["V0", "V1", "V2", "V3"])
def test_information_is_nonnegative_and_zero_at_certainty(sensor, variant):
    p = np.array([0.0, 0.1, 0.5, 1.0])
    gain, found = information(p, sensor, 0, variant)
    assert (gain >= 0).all()
    assert gain[0] < 1e-8
    assert gain[-1] < 1e-8
    assert gain[1] > 0
    assert (found <= p).all()
