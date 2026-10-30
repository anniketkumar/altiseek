import numpy as np
import pytest

from altiseek.cache.storage import DETECTION_DTYPE
from altiseek.deploy.projection import GroundCamera
from altiseek.deploy.search import LiveSearchPilot, LiveView
from tests.env.test_search import built, cfg, env, library, sensor

__all__ = ["built", "cfg", "env", "library", "sensor"]


def test_live_source_never_reads_cache_and_uses_shared_state(env, built, sensor, monkeypatch):
    world, _ = built
    requests = []

    def source(goal, level):
        requests.append((goal.copy(), level))
        camera = GroundCamera(
            goal + np.array([0.1, 0, 0]),
            np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]),
            env.cache_config.render.camera.resolution_px,
            env.cache_config.render.camera.hfov_deg,
        )
        return LiveView(camera, np.zeros(0, dtype=DETECTION_DTYPE))

    live = LiveSearchPilot(world, env.cache_config, sensor, env.config, source)

    def forbidden(*args, **kwargs):
        raise AssertionError("Live deployment read a cached detection")

    monkeypatch.setattr(live.pool, "get", forbidden)
    obs, _ = live.reset(seed=71, options={"start_node": 0})
    assert live.cache is None
    np.testing.assert_array_equal(live.belief.observed[0], live.live_visible)
    np.testing.assert_array_equal(obs["map"][live.level_count + 3], live.live_visible.T[::-1])
    assert live.observations == 1
    action = int(np.flatnonzero(live.action_masks())[0])
    target = live.nav.next[live.node, action]
    cost = live.nav.cost[live.node, action]
    old_energy = live.energy_used_s
    live.step(action)
    assert live.node == target
    assert live.energy_used_s == pytest.approx(old_energy + cost)
    assert len(requests) == 2
    assert live.observations == 2
    # The source receives only a commanded pose/level, never target coordinates.
    assert requests[1][0].shape == (3,)


def test_live_pilot_rejects_test_worlds(env, built, sensor):
    world, _ = built
    with pytest.raises(ValueError, match="B only"):
        LiveSearchPilot(
            world.model_copy(update={"split": "T"}),
            env.cache_config,
            sensor,
            env.config,
            lambda *_: None,
        )
