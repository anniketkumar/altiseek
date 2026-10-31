"""The energy and time model (CLAUDE.md §14). One implementation, shared by the Python
simulator, the classical baselines, level design (M3) and the Gazebo supervisor (M10).

Time: horizontal distance at ``horizontal_speed_mps``, then the altitude change at
``vertical_speed_mps``, plus ``dwell_s`` per observation. Energy is time in hover-equivalent
seconds, except that climbing time costs ``climb_multiplier`` times as much. The battery is an
energy budget in the same unit.
"""

import math
from collections.abc import Sequence

from altiseek.config import Config


class EnergyModel(Config):
    """Speeds in m/s, times and energy in (hover-equivalent) seconds."""

    horizontal_speed_mps: float
    vertical_speed_mps: float
    dwell_s: float
    climb_multiplier: float
    battery_s: float

    def move(self, dx_m: float, dy_m: float, dz_m: float) -> tuple[float, float]:
        """Time and energy of flying by ``(dx, dy)`` horizontally and ``dz`` vertically (up > 0)."""
        horizontal = math.hypot(dx_m, dy_m) / self.horizontal_speed_mps
        vertical = abs(dz_m) / self.vertical_speed_mps
        energy = horizontal + vertical * (self.climb_multiplier if dz_m > 0 else 1.0)
        return horizontal + vertical, energy

    def observe(self) -> tuple[float, float]:
        """Time and energy of one observation (hovering for the dwell time)."""
        return self.dwell_s, self.dwell_s

    def measured_interval(self, elapsed_s: float, climbing: bool) -> float:
        """Live hover-equivalent energy from simulator time and measured climb state.

        Sequential constant-speed motion reproduces ``move``. Concurrent flight
        is charged for actual elapsed time; climbing retains the same multiplier.
        """
        if not math.isfinite(elapsed_s) or elapsed_s < 0:
            raise ValueError("Measured elapsed time must be finite and nonnegative")
        return elapsed_s * (self.climb_multiplier if climbing else 1.0)

    def tour(
        self, points_m: Sequence[tuple[float, float]], altitude_m: float, start_altitude_m: float
    ) -> tuple[float, float]:
        """Time and energy of climbing to ``altitude_m`` over the first point, then visiting
        every point in order with one observation each."""
        time_s, energy_s = self.move(0.0, 0.0, altitude_m - start_altitude_m)
        for k, (x, y) in enumerate(points_m):
            if k:
                px, py = points_m[k - 1]
                t, e = self.move(x - px, y - py, 0.0)
                time_s, energy_s = time_s + t, energy_s + e
            t, e = self.observe()
            time_s, energy_s = time_s + t, energy_s + e
        return time_s, energy_s
