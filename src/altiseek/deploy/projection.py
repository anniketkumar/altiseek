"""Live-camera ground projection using the measured optical-link pose.

Gazebo camera axes are forward +X, left +Y, up +Z. World coordinates are ENU.
Pixel centres have integer coordinates, matching the shared camera geometry.
This module contains no detector, truth, policy or belief-update logic.
"""

from dataclasses import dataclass

import numpy as np

from altiseek.geometry.camera import focal_length_px


def quaternion_matrix(wxyz: np.ndarray) -> np.ndarray:
    """Active local-to-world rotation; reject malformed or non-unit poses."""
    q = np.asarray(wxyz, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > 1e-3:
        raise ValueError("Expected a finite unit wxyz quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


@dataclass(frozen=True)
class GroundCamera:
    """Measured camera origin and rotation, not a commanded vehicle setpoint."""

    position_enu_m: np.ndarray
    rotation: np.ndarray
    resolution_px: int
    hfov_deg: float

    def __post_init__(self) -> None:
        position = np.asarray(self.position_enu_m, dtype=float).copy()
        rotation = np.asarray(self.rotation, dtype=float).copy()
        if position.shape != (3,) or not np.isfinite(position).all() or position[2] <= 0:
            raise ValueError("Camera must be finite and above the sea")
        if (
            rotation.shape != (3, 3)
            or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6)
        ):
            raise ValueError("Expected a proper camera-to-ENU rotation")
        if self.resolution_px < 2 or not 0 < self.hfov_deg < 180:
            raise ValueError("Invalid camera intrinsics")
        position.flags.writeable = rotation.flags.writeable = False
        object.__setattr__(self, "position_enu_m", position)
        object.__setattr__(self, "rotation", rotation)

    @property
    def focal_px(self) -> float:
        return focal_length_px(self.hfov_deg, self.resolution_px)

    def pixel_to_ground(self, pixels: np.ndarray) -> np.ndarray:
        """Project (...,2) column/row centres onto z=0; reject horizon rays."""
        uv = np.asarray(pixels, dtype=float)
        if uv.ndim == 0 or uv.shape[-1] != 2 or not np.isfinite(uv).all():
            raise ValueError("Expected finite pixel coordinates (...,2)")
        offset = (uv - (self.resolution_px - 1) / 2) / self.focal_px
        local = np.concatenate((np.ones_like(offset[..., :1]), -offset), axis=-1)
        rays = local @ self.rotation.T
        if np.any(rays[..., 2] >= -1e-9):
            raise ValueError("Camera ray does not intersect the sea in front")
        scale = -self.position_enu_m[2] / rays[..., 2]
        return (self.position_enu_m + rays * scale[..., None])[..., :2]

    def ground_to_pixel(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project sea points, returning pixels and a positive-depth mask."""
        xy = np.asarray(xy, dtype=float)
        if xy.ndim == 0 or xy.shape[-1] != 2 or not np.isfinite(xy).all():
            raise ValueError("Expected finite sea coordinates (...,2)")
        points = np.concatenate((xy, np.zeros_like(xy[..., :1])), axis=-1)
        local = (points - self.position_enu_m) @ self.rotation
        positive = local[..., 0] > 1e-9
        depth = np.where(positive, local[..., 0], 1)
        uv = -local[..., 1:] / depth[..., None] * self.focal_px
        return uv + (self.resolution_px - 1) / 2, positive

    def footprint(self) -> np.ndarray:
        """Four image-boundary corners on the sea, clockwise from top-left."""
        lo, hi = -0.5, self.resolution_px - 0.5
        return self.pixel_to_ground(np.array([[lo, lo], [hi, lo], [hi, hi], [lo, hi]]))

    def fully_visible_cells(self, grid_cells: int, cell_m: float) -> np.ndarray:
        """Mask [ix,iy]; all four cell corners must fall inside the actual image.

        This intentionally excludes partial cells for live deployment. Cached M7
        centre-based observation bounds and historical results remain unchanged.
        """
        if grid_cells < 1 or not np.isfinite(cell_m) or cell_m <= 0:
            raise ValueError("Invalid world grid")
        ix, iy = np.meshgrid(np.arange(grid_cells), np.arange(grid_cells), indexing="ij")
        origins = np.stack((ix, iy), axis=-1) * cell_m
        corners = origins[..., None, :] + np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) * cell_m
        pixels, positive = self.ground_to_pixel(corners)
        inside = ((pixels >= -0.5 - 1e-7) & (pixels <= self.resolution_px - 0.5 + 1e-7)).all(-1)
        return (positive & inside).all(-1)
