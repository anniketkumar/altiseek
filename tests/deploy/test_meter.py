import numpy as np
import pytest

from altiseek.deploy.meter import FlightMeter
from altiseek.env.energy import EnergyModel


def meter(battery=500):
    energy = EnergyModel(
        horizontal_speed_mps=10,
        vertical_speed_mps=3,
        dwell_s=2,
        climb_multiplier=1.5,
        battery_s=battery,
    )
    return FlightMeter(energy, np.array([30, 45, 60, 72, 89]))


def test_measured_sequential_motion_agrees_with_shared_model():
    m = meter()
    m.start(100, np.array([0, 0, 30]))
    m.update(110, np.array([100, 0, 30]))
    m.update(115, np.array([100, 0, 45]))
    m.update(117, np.array([100, 0, 45]))
    expected_time, expected_energy = m.energy.move(100, 0, 15)
    state = m.snapshot()
    assert state["elapsed_s"] == expected_time + 2
    assert state["energy_s"] == expected_energy + 2
    assert state["path_m"] == 100
    assert state["climbing_s"] == 5
    assert state["time_per_level"].sum() == 17


def test_expiry_clips_time_and_prevents_landing_charges():
    m = meter(3)
    m.start(0, np.array([0, 0, 30]))
    m.update(5, np.array([10, 0, 45]))
    state = m.snapshot()
    assert state["exhausted"]
    assert state["energy_s"] == 3
    assert state["elapsed_s"] == 2
    assert state["path_m"] == 4
    m.update(50, np.array([10, 0, 0]))
    assert m.snapshot()["energy_s"] == 3
    assert m.snapshot()["elapsed_s"] == 2


def test_hover_jitter_is_not_a_climb_and_time_cannot_reverse():
    m = meter()
    m.start(0, np.array([0, 0, 30]))
    m.update(1, np.array([0, 0, 30.01]))
    assert m.snapshot()["energy_s"] == 1
    with pytest.raises(ValueError, match="backwards"):
        m.update(0.5, np.array([0, 0, 30]))
    with pytest.raises(ValueError, match="finite"):
        m.energy.measured_interval(float("nan"), False)
