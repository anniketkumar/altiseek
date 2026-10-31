"""Continuous simulated-time battery accounting independent of policy inference."""

import threading

import numpy as np

from altiseek.env.energy import EnergyModel


class BudgetExhausted(RuntimeError):
    """Normal episode termination; no further image may update belief."""


class FlightMeter:
    def __init__(self, energy: EnergyModel, altitudes: np.ndarray) -> None:
        self.energy = energy
        self.altitudes = np.asarray(altitudes, dtype=float)
        self.lock = threading.Lock()
        self.previous: tuple[float, np.ndarray] | None = None
        self.elapsed_s = self.energy_s = self.path_m = self.climbing_s = 0.0
        self.time_per_level = np.zeros(len(altitudes))
        self.finished = False

    def start(self, stamp_s: float, position: np.ndarray) -> None:
        with self.lock:
            if self.previous is not None:
                raise RuntimeError("Flight meter is already started")
            self._validate(stamp_s, position)
            self.previous = (float(stamp_s), np.asarray(position, dtype=float).copy())

    @staticmethod
    def _validate(stamp_s: float, position: np.ndarray) -> None:
        if (
            not np.isfinite(stamp_s)
            or np.shape(position) != (3,)
            or not np.isfinite(position).all()
        ):
            raise ValueError("Invalid timed flight pose")

    def update(self, stamp_s: float, position: np.ndarray) -> None:
        with self.lock:
            self._validate(stamp_s, position)
            if self.previous is None or self.finished:
                return
            previous_stamp, previous_position = self.previous
            dt = stamp_s - previous_stamp
            if dt < -1e-9:
                raise ValueError("Simulator time moved backwards")
            if dt <= 0:
                return
            position = np.asarray(position, dtype=float)
            delta = position - previous_position
            # A 0.2 m/s deadband excludes hover jitter from climb-power accounting.
            climbing = delta[2] / dt > 0.2
            cost = self.energy.measured_interval(dt, climbing)
            remaining = max(0.0, self.energy.battery_s - self.energy_s)
            fraction = min(1.0, remaining / cost) if cost else 1.0
            charged = dt * fraction
            self.elapsed_s += charged
            self.energy_s = min(self.energy.battery_s, self.energy_s + cost)
            self.climbing_s += charged if climbing else 0
            self.path_m += float(np.linalg.norm(delta[:2])) * fraction
            level = int(np.argmin(abs(self.altitudes - (previous_position[2] + position[2]) / 2)))
            self.time_per_level[level] += charged
            self.previous = (float(stamp_s), position.copy())
            if self.energy_s >= self.energy.battery_s - 1e-9:
                self.finished = True

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "elapsed_s": self.elapsed_s,
                "energy_s": self.energy_s,
                "path_m": self.path_m,
                "climbing_s": self.climbing_s,
                "time_per_level": self.time_per_level.copy(),
                "started": self.previous is not None,
                "exhausted": self.energy_s >= self.energy.battery_s - 1e-9,
            }

    def stop(self) -> None:
        with self.lock:
            self.finished = True
