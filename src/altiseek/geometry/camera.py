"""Pinhole camera above a flat sea: focal length, GSD and the sea horizon.

The camera is pitched down from the horizontal by ``pitch_deg`` (90 is nadir, the SeaDronesSee
convention) with no roll, at ``altitude_m`` above the water. Image offsets ``x_px`` (right) and
``y_px`` (down) are measured from the principal point, assumed to be the image centre, and
pixels are square. Lens distortion is ignored.

:class:`TiltedCamera` does the same geometry for whole images. Its ground frame has its origin
on the water directly below the camera, X to the right of the view and Y forward along the
horizontal viewing direction, in metres. Image pixel centres sit at integer (column, row).
"""

import math
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray

EARTH_RADIUS_M = 6_371_000.0


def focal_length_px(hfov_deg: float, width_px: float) -> float:
    """Focal length in pixels of an image ``width_px`` wide with horizontal FOV ``hfov_deg``."""
    return width_px / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))


def hfov_deg(focal_px: float, width_px: float) -> float:
    """Horizontal field of view in degrees of an image ``width_px`` wide."""
    return math.degrees(2.0 * math.atan(width_px / (2.0 * focal_px)))


def nadir_gsd_mpp(altitude_m: float, focal_px: float) -> float:
    """Ground sample distance of a nadir view, in metres per pixel (uniform over the image)."""
    return altitude_m / focal_px


def pixel_gsd_mpp(
    x_px: float, y_px: float, pitch_deg: float, altitude_m: float, focal_px: float
) -> tuple[float, float]:
    """Ground metres spanned by one image column step and one row step at ``(x_px, y_px)``.

    The first value is the across-view GSD (moving one pixel right), the second the along-view
    GSD (moving one pixel down). Both equal :func:`nadir_gsd_mpp` for a nadir camera.
    Raises ValueError for pixels at or above the horizon.
    """
    pitch = math.radians(pitch_deg)
    depth = y_px * math.cos(pitch) + focal_px * math.sin(pitch)
    if depth <= 0:
        raise ValueError(f"pixel ({x_px}, {y_px}) does not see the water")
    across = altitude_m / depth
    along = altitude_m * math.hypot(focal_px, x_px * math.cos(pitch)) / depth**2
    return across, along


def horizon_dip_deg(altitude_m: float) -> float:
    """Angle of the visible sea horizon below the horizontal, in degrees (no refraction)."""
    return math.degrees(math.acos(EARTH_RADIUS_M / (EARTH_RADIUS_M + altitude_m)))


def horizon_offset_px(pitch_deg: float, altitude_m: float, focal_px: float) -> float:
    """How far the visible sea horizon lies above the principal point, in pixels."""
    return focal_px * math.tan(math.radians(pitch_deg - horizon_dip_deg(altitude_m)))


def apply_homography(
    hom: NDArray[np.float64], x: ArrayLike, y: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Map points through a 3 x 3 homography; the flag marks a positive homogeneous scale."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    w = hom[2, 0] * x + hom[2, 1] * y + hom[2, 2]
    positive = w > 0
    safe = np.where(positive, w, 1.0)
    u = (hom[0, 0] * x + hom[0, 1] * y + hom[0, 2]) / safe
    v = (hom[1, 0] * x + hom[1, 1] * y + hom[1, 2]) / safe
    return u, v, positive


@dataclass(frozen=True)
class TiltedCamera:
    """A pitched pinhole camera over a flat sea, with array-friendly projections.

    ``pitch_deg`` is the true pitch (logged pitch plus the fitted gimbal bias, D12).
    """

    width_px: int
    height_px: int
    focal_px: float
    pitch_deg: float
    altitude_m: float
    cx_px: float
    cy_px: float

    @classmethod
    def centred(
        cls, width_px: int, height_px: int, focal_px: float, pitch_deg: float, altitude_m: float
    ) -> "TiltedCamera":
        """A camera whose principal point is the image centre."""
        return cls(
            width_px,
            height_px,
            focal_px,
            pitch_deg,
            altitude_m,
            (width_px - 1) / 2.0,
            (height_px - 1) / 2.0,
        )

    def ground_to_image_matrix(self) -> NDArray[np.float64]:
        """3 x 3 homography H with ``[u w, v w, w] = H [X, Y, 1]`` for points on the water."""
        pitch = math.radians(self.pitch_deg)
        h = self.altitude_m
        intrinsics = np.array(
            [[self.focal_px, 0.0, self.cx_px], [0.0, self.focal_px, self.cy_px], [0.0, 0.0, 1.0]]
        )
        to_camera = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, -math.sin(pitch), h * math.cos(pitch)],
                [0.0, math.cos(pitch), h * math.sin(pitch)],
            ]
        )
        return intrinsics @ to_camera

    def ground_to_image(
        self, x_m: ArrayLike, y_m: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        """Column, row and in-front-of-camera flag of ground points."""
        return apply_homography(self.ground_to_image_matrix(), x_m, y_m)

    def image_to_ground(
        self, u_px: ArrayLike, v_px: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        """Ground X, Y of pixels, and whether each pixel sees the water (is below the horizon)."""
        pitch = math.radians(self.pitch_deg)
        x = np.asarray(u_px, dtype=np.float64) - self.cx_px
        y = np.asarray(v_px, dtype=np.float64) - self.cy_px
        depth = y * math.cos(pitch) + self.focal_px * math.sin(pitch)
        water = depth > 0
        scale = self.altitude_m / np.where(water, depth, 1.0)
        return scale * x, scale * (self.focal_px * math.cos(pitch) - y * math.sin(pitch)), water

    def off_nadir_deg(self, u_px: ArrayLike, v_px: ArrayLike) -> NDArray[np.float64]:
        """Angle between each pixel's ray and straight down, in degrees (above 90 at the sky)."""
        pitch = math.radians(self.pitch_deg)
        x = np.asarray(u_px, dtype=np.float64) - self.cx_px
        y = np.asarray(v_px, dtype=np.float64) - self.cy_px
        depth = y * math.cos(pitch) + self.focal_px * math.sin(pitch)
        return np.degrees(
            np.arccos(np.clip(depth / np.sqrt(x**2 + y**2 + self.focal_px**2), -1, 1))
        )

    def native_gsd_mpp(self, u_px: ArrayLike, v_px: ArrayLike) -> NDArray[np.float64]:
        """The coarser of the across- and along-view GSD at each pixel; inf above the horizon."""
        pitch = math.radians(self.pitch_deg)
        x = np.asarray(u_px, dtype=np.float64) - self.cx_px
        y = np.asarray(v_px, dtype=np.float64) - self.cy_px
        depth = y * math.cos(pitch) + self.focal_px * math.sin(pitch)
        water = depth > 0
        d = np.where(water, depth, 1.0)
        across = self.altitude_m / d
        along = self.altitude_m * np.hypot(self.focal_px, x * math.cos(pitch)) / d**2
        return np.where(water, np.maximum(across, along), np.inf)

    def downscaled(self, factor: int) -> "TiltedCamera":
        """The camera of this image cropped at the top-left to multiples of ``factor`` and
        area-averaged by ``factor``: pixel centre ``i`` of the result spans old pixels
        ``i * factor ... i * factor + factor - 1``."""
        return replace(
            self,
            width_px=self.width_px // factor,
            height_px=self.height_px // factor,
            focal_px=self.focal_px / factor,
            cx_px=(self.cx_px + 0.5) / factor - 0.5,
            cy_px=(self.cy_px + 0.5) / factor - 0.5,
        )
