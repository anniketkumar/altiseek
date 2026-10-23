"""Pinhole camera above a flat sea: focal length, GSD and the sea horizon.

The camera is pitched down from the horizontal by ``pitch_deg`` (90 is nadir, the SeaDronesSee
convention) with no roll, at ``altitude_m`` above the water. Image offsets ``x_px`` (right) and
``y_px`` (down) are measured from the principal point, assumed to be the image centre, and
pixels are square. Lens distortion is ignored.
"""

import math

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
