"""The simulated nadir camera's footprints and the altitude-level lattice (CLAUDE.md §8, §12).

For a square ``R x R`` camera with horizontal field of view theta at altitude h, the footprint is
``W(h) = 2 h tan(theta / 2)`` metres wide and the simulated GSD is ``W(h) / R``. Levels are defined
by footprint width in whole cells, so views tile the grid exactly. At each level, view centres
lie on a lattice with that level's stride (in cells), and every footprint lies fully inside the
square search area of ``grid_cells x grid_cells`` cells.
"""

import math

from pydantic import model_validator

from altiseek.config import Config


def footprint_m(altitude_m: float, hfov_deg: float) -> float:
    """Ground width covered by a nadir view from ``altitude_m``."""
    return 2.0 * altitude_m * math.tan(math.radians(hfov_deg) / 2.0)


def altitude_for_footprint_m(width_m: float, hfov_deg: float) -> float:
    """The altitude whose nadir footprint is ``width_m`` wide."""
    return width_m / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))


def sim_gsd_mpp(width_m: float, resolution_px: int) -> float:
    """Simulated GSD of a footprint ``width_m`` wide imaged at ``resolution_px`` pixels."""
    return width_m / resolution_px


class Level(Config):
    """One altitude level: footprint and lattice stride, in cells."""

    name: str
    footprint_cells: int
    stride_cells: int


class LevelTable(Config):
    """Grid and levels, finest first; validated so every lattice ends exactly at the far edge."""

    cell_m: float
    grid_cells: int
    levels: tuple[Level, ...]

    @model_validator(mode="after")
    def _check(self) -> "LevelTable":
        previous = 0
        for lvl in self.levels:
            if not 0 < lvl.footprint_cells <= self.grid_cells:
                raise ValueError(f"{lvl.name}: footprint must be 1..{self.grid_cells} cells")
            if lvl.stride_cells < 1 or (self.grid_cells - lvl.footprint_cells) % lvl.stride_cells:
                raise ValueError(
                    f"{lvl.name}: stride {lvl.stride_cells} does not step a "
                    f"{lvl.footprint_cells}-cell footprint across {self.grid_cells} cells"
                )
            if lvl.footprint_cells <= previous:
                raise ValueError("levels must be listed finest (smallest footprint) first")
            previous = lvl.footprint_cells
        return self

    @property
    def area_m(self) -> float:
        return self.cell_m * self.grid_cells

    def footprint_m(self, level: Level) -> float:
        return level.footprint_cells * self.cell_m

    def lattice(self, level: Level) -> list[tuple[int, int]]:
        """Lower-left cells ``(ix, iy)`` of every footprint at ``level``, rows from the south."""
        starts = range(0, self.grid_cells - level.footprint_cells + 1, level.stride_cells)
        return [(ix, iy) for iy in starts for ix in starts]

    def views_per_world(self, level: Level) -> int:
        side = (self.grid_cells - level.footprint_cells) // level.stride_cells + 1
        return side * side

    def centre_m(self, level: Level, cell: tuple[int, int]) -> tuple[float, float]:
        """World x, y (metres, origin at the south-west corner) of the view centre at ``cell``."""
        half = self.footprint_m(level) / 2.0
        return cell[0] * self.cell_m + half, cell[1] * self.cell_m + half

    def lawnmower(self, level: Level) -> list[tuple[float, float]]:
        """View centres (metres) in boustrophedon order: east along the first row, west along
        the next, and so on northwards. Covers every lattice point once."""
        starts = list(range(0, self.grid_cells - level.footprint_cells + 1, level.stride_cells))
        path = []
        for row, iy in enumerate(starts):
            for ix in starts if row % 2 == 0 else starts[::-1]:
                path.append(self.centre_m(level, (ix, iy)))
        return path
