"""The world frame on a square-pixel raster (CLAUDE.md §5).

World frame: origin at the search area's south-west corner, x east, y north, metres. A raster of
``gsd_mpp`` covers ``[0, width) x [0, height)`` north-up: row 0 is the north edge and rows grow
southward, columns eastward, so converting a row to y needs a flip. Pixel centres sit at integer
``(col, row)``. Composed worlds live on such a raster at the canonical GSD g0 (CLAUDE.md §8).
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class Raster:
    """A north-up raster of ``height_px`` x ``width_px`` pixels of ``gsd_mpp`` metres."""

    gsd_mpp: float
    width_px: int
    height_px: int

    @classmethod
    def covering(cls, width_m: float, height_m: float, gsd_mpp: float) -> "Raster":
        """The raster of an area whose sides are whole numbers of pixels."""
        cols, rows = round(width_m / gsd_mpp), round(height_m / gsd_mpp)
        if not (math.isclose(cols * gsd_mpp, width_m) and math.isclose(rows * gsd_mpp, height_m)):
            raise ValueError(
                f"{width_m} x {height_m} m is not a whole number of {gsd_mpp} m pixels"
            )
        return cls(gsd_mpp, cols, rows)

    @property
    def height_m(self) -> float:
        return self.height_px * self.gsd_mpp

    def to_pixel(
        self, x_m: ArrayLike, y_m: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Column and row (pixel centres at integers) of world points."""
        col = np.asarray(x_m, dtype=np.float64) / self.gsd_mpp - 0.5
        row = (self.height_m - np.asarray(y_m, dtype=np.float64)) / self.gsd_mpp - 0.5
        return col, row

    def to_world(
        self, col: ArrayLike, row: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """World x, y of raster points (pixel centres at integers)."""
        x = (np.asarray(col, dtype=np.float64) + 0.5) * self.gsd_mpp
        y = self.height_m - (np.asarray(row, dtype=np.float64) + 0.5) * self.gsd_mpp
        return x, y

    def window(self, x0_m: float, y0_m: float, side_m: float) -> tuple[int, int, int]:
        """``(row0, col0, size_px)`` of the square ``[x0, x0 + side) x [y0, y0 + side)``: its
        north-west pixel and side. The square must lie on pixel boundaries."""
        col0 = x0_m / self.gsd_mpp
        row0 = (self.height_m - y0_m - side_m) / self.gsd_mpp
        size = side_m / self.gsd_mpp
        whole = [round(v) for v in (row0, col0, size)]
        if not all(
            math.isclose(v, w, abs_tol=1e-6) for v, w in zip((row0, col0, size), whole, strict=True)
        ):
            raise ValueError(f"square at ({x0_m}, {y0_m}) m, {side_m} m wide is off the pixel grid")
        return whole[0], whole[1], whole[2]
