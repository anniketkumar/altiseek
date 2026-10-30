"""World/ROS ENU and PX4 NED conversions, with a shared local origin.

Positions and velocities use SI units. These functions change axes only; callers
must subtract the PX4 origin in world coordinates before converting positions.
Yaw is in radians: ENU zero faces east, NED zero faces north.
"""

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


def enu_to_ned(vector: ArrayLike) -> NDArray[np.float64]:
    """Convert vectors (..., 3) from east/north/up to north/east/down."""
    values = np.asarray(vector, dtype=np.float64)
    if values.ndim == 0 or values.shape[-1] != 3 or not np.isfinite(values).all():
        raise ValueError("Expected finite vectors with final dimension 3")
    converted = values[..., [1, 0, 2]].copy()
    converted[..., 2] *= -1
    return converted


def ned_to_enu(vector: ArrayLike) -> NDArray[np.float64]:
    """Convert vectors (..., 3) from north/east/down to east/north/up."""
    return enu_to_ned(vector)


def enu_yaw_to_ned(yaw_rad: float) -> float:
    """Convert ENU heading to NED heading, wrapped to [-pi, pi)."""
    if not math.isfinite(yaw_rad):
        raise ValueError("Yaw must be finite")
    return (math.pi / 2 - yaw_rad + math.pi) % (2 * math.pi) - math.pi


def ned_yaw_to_enu(yaw_rad: float) -> float:
    """Convert NED heading to ENU heading, wrapped to [-pi, pi)."""
    return enu_yaw_to_ned(yaw_rad)
