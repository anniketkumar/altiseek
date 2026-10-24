"""Ground-plane rectification of tilted frames into nadir-equivalent views (D14).

A view is a square window on the water, ``size_px`` pixels of ``gsd_mpp`` metres, axis-aligned in
the camera's ground frame (``altiseek.geometry.camera``) with its top row on the far side. To fill
it without aliasing:

1. the frame is area-averaged by an integer factor m, so that everywhere in the window its pixels
   are at least as coarse as an intermediate grid of ``gsd_mpp / supersample``;
2. it is warped onto that grid by bilinear interpolation, which is then an upsampling and cannot
   alias;
3. the grid is reduced to the view by :func:`altiseek.imaging.pipeline.render_view`, the shared
   resampling path.

A view pixel is valid when all four of its corners come from source points that are inside the
frame, see the water, lie within ``max_off_nadir_deg`` of straight down, and are at least as
fine as the view (native GSD <= ``gsd_mpp``). Each constraint carves a convex region of the
water, so checking corners suffices. Invalid pixels are filled with letterbox grey.
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from altiseek.geometry.camera import TiltedCamera, apply_homography
from altiseek.imaging.pipeline import CameraPipeline, area_resize, render_view

#: Grey for pixels without valid source, the usual YOLO letterbox colour.
FILL = (114, 114, 114)
# Pillow's transforms address pixels by their corners; ours by their centres.
_TO_CORNERS = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
_TO_CENTRES = np.linalg.inv(_TO_CORNERS)


@dataclass(frozen=True)
class ViewWindow:
    """A square window on the water; ``(x0_m, y0_m)`` is its left, near corner."""

    x0_m: float
    y0_m: float
    gsd_mpp: float
    size_px: int

    @property
    def side_m(self) -> float:
        return self.gsd_mpp * self.size_px

    def grid_to_ground(self, scale: int = 1) -> NDArray[np.float64]:
        """Homography from pixel centres of a grid ``scale`` times finer than the view to ground."""
        g = self.gsd_mpp / scale
        top = self.y0_m + self.side_m
        return np.array([[g, 0.0, self.x0_m + 0.5 * g], [0.0, -g, top - 0.5 * g], [0, 0, 1.0]])

    def ground_to_view(
        self, x_m: NDArray[np.float64], y_m: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """View column and row (pixel centres at integers) of ground points."""
        col = (x_m - self.x0_m) / self.gsd_mpp - 0.5
        row = (self.y0_m + self.side_m - y_m) / self.gsd_mpp - 0.5
        return col, row


@dataclass(frozen=True)
class RectifiedView:
    """A rendered view: ``size x size x 3`` image and a ``size x size`` validity mask."""

    image: NDArray[np.uint8]
    valid: NDArray[np.bool_]
    #: The integer pre-averaging factor m that was applied to the frame.
    downsample: int


def source_valid(
    camera: TiltedCamera,
    u_px: NDArray[np.float64],
    v_px: NDArray[np.float64],
    gsd_mpp: float,
    max_off_nadir_deg: float,
) -> NDArray[np.bool_]:
    """Frame points that may feed a view of ``gsd_mpp``: inside, on the water, near enough to
    nadir and at least as fine as the view."""
    inside = (
        (u_px >= -0.5)
        & (u_px <= camera.width_px - 0.5)
        & (v_px >= -0.5)
        & (v_px <= camera.height_px - 0.5)
    )
    fine = camera.native_gsd_mpp(u_px, v_px) <= gsd_mpp
    return inside & fine & (camera.off_nadir_deg(u_px, v_px) <= max_off_nadir_deg)


def valid_pixels(
    camera: TiltedCamera, window: ViewWindow, max_off_nadir_deg: float
) -> tuple[NDArray[np.bool_], float]:
    """Validity of each view pixel, and the finest native GSD over the valid corners (inf if
    none)."""
    n = window.size_px
    rows, cols = np.mgrid[0 : n + 1, 0 : n + 1].astype(np.float64) - 0.5  # pixel corners
    u, v, ahead = apply_homography(
        camera.ground_to_image_matrix() @ window.grid_to_ground(), cols, rows
    )
    ok = ahead & source_valid(camera, u, v, window.gsd_mpp, max_off_nadir_deg)
    valid = ok[:-1, :-1] & ok[1:, :-1] & ok[:-1, 1:] & ok[1:, 1:]
    finest = float(camera.native_gsd_mpp(u[ok], v[ok]).min()) if ok.any() else math.inf
    return valid, finest


def rectify(
    frame: NDArray[np.uint8],
    camera: TiltedCamera,
    window: ViewWindow,
    *,
    max_off_nadir_deg: float,
    pipeline: CameraPipeline,
    supersample: int = 4,
) -> RectifiedView:
    """Render the view of ``window`` from ``frame`` (``H x W x 3``, the camera's image)."""
    if frame.shape[:2] != (camera.height_px, camera.width_px):
        raise ValueError(f"frame {frame.shape[:2]} does not match the camera")
    n, k = window.size_px, supersample
    valid, finest = valid_pixels(camera, window, max_off_nadir_deg)
    if not valid.any():
        return RectifiedView(np.full((n, n, 3), FILL, dtype=np.uint8), valid, 1)

    m = max(1, math.ceil(window.gsd_mpp / (k * finest) - 1e-9))
    rows, cols = camera.height_px // m, camera.width_px // m
    small = area_resize(frame[: rows * m, : cols * m], rows, cols)
    to_small = camera.downscaled(m).ground_to_image_matrix() @ window.grid_to_ground(k)
    corner_map = _TO_CORNERS @ to_small @ _TO_CENTRES
    corner_map /= corner_map[2, 2]
    warped = Image.fromarray(small).transform(
        (n * k, n * k),
        Image.Transform.PERSPECTIVE,
        data=tuple(corner_map.ravel()[:8]),
        resample=Image.Resampling.BILINEAR,
        fillcolor=FILL,
    )
    fine = np.array(warped)
    fine[~np.repeat(np.repeat(valid, k, axis=0), k, axis=1)] = FILL
    image = render_view(fine, n, n, pipeline)
    image[~valid] = FILL  # blur and JPEG can bleed into the fill
    return RectifiedView(image, valid, m)


def project_boxes(
    boxes_xywh_px: NDArray[np.float64], camera: TiltedCamera, window: ViewWindow
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Image boxes (COCO ``x, y, w, h``, pixel-corner coordinates) as view boxes.

    Returns ``x_min, y_min, x_max, y_max`` in view pixel-corner coordinates (unclipped) and
    whether all four box corners see the water. The view box bounds the projected quadrilateral.
    """
    boxes = np.asarray(boxes_xywh_px, dtype=np.float64).reshape(-1, 4)
    x0, y0 = boxes[:, 0] - 0.5, boxes[:, 1] - 0.5  # to pixel-centre coordinates
    x1, y1 = x0 + boxes[:, 2], y0 + boxes[:, 3]
    us = np.stack([x0, x1, x0, x1], axis=1)
    vs = np.stack([y0, y0, y1, y1], axis=1)
    gx, gy, water = camera.image_to_ground(us, vs)
    col, row = window.ground_to_view(gx, gy)
    out = (
        np.stack([col.min(axis=1), row.min(axis=1), col.max(axis=1), row.max(axis=1)], axis=1) + 0.5
    )  # back to corner coordinates
    return out, water.all(axis=1)
