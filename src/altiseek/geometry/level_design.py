"""Level design (CLAUDE.md §8, M3): from a measured recall-vs-GSD curve to a level table,
cell size and battery.

Footprints are whole numbers of cells whose lattice ends exactly at the far edge, with a stride
of the full or half footprint. Recall at a level's simulated GSD is interpolated in log-GSD from
the measured curve, never extrapolated. A candidate has 3 to 5 levels with recall at least
``finest_min`` at the finest and at most ``coarsest_max`` at the coarsest; candidates are ranked
by their largest recall step between neighbouring levels (smaller is better, i.e. evenly spread),
then by fewer views at the finest level. The battery must make full coverage at the finest level
infeasible while full coverage at the coarsest costs at most ``coarse_share`` of it.
"""

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from altiseek.env.energy import EnergyModel
from altiseek.geometry.levels import Level, LevelTable, altitude_for_footprint_m, sim_gsd_mpp


def recall_at(gsd_mpp: float, curve: Sequence[tuple[float, float]]) -> float | None:
    """Recall at ``gsd_mpp`` by linear interpolation in log-GSD; None outside the measured range."""
    gsds = np.array([g for g, _ in curve])
    recalls = np.array([r for _, r in curve])
    order = np.argsort(gsds)
    gsds, recalls = gsds[order], recalls[order]
    if not gsds[0] <= gsd_mpp <= gsds[-1]:
        return None
    return float(np.interp(math.log(gsd_mpp), np.log(gsds), recalls))


def footprint_options(grid_cells: int) -> list[tuple[int, int]]:
    """``(footprint, stride)`` pairs, in cells, whose lattices end exactly at the far edge."""
    options = []
    for n in range(1, grid_cells + 1):
        for stride in (n, n // 2) if n % 2 == 0 else (n,):
            if stride >= 1 and (grid_cells - n) % stride == 0:
                options.append((n, stride))
    return options


@dataclass(frozen=True)
class Candidate:
    """A level table with its expected recall, simulated GSDs and full-coverage costs."""

    table: LevelTable
    altitude_m: tuple[float, ...]
    gsd_mpp: tuple[float, ...]
    recall: tuple[float, ...]
    coverage_energy_s: tuple[float, ...]
    battery_s: float
    largest_recall_step: float


def coverage_energy_s(
    table: LevelTable, level: Level, hfov_deg: float, energy: EnergyModel
) -> float:
    """Energy of a lawnmower over every view of ``level``, starting at that level's altitude."""
    altitude = altitude_for_footprint_m(table.footprint_m(level), hfov_deg)
    return energy.tour(table.lawnmower(level), altitude, altitude)[1]


def choose_battery(
    finest_s: float, coarsest_s: float, preferred_s: float, coarse_share: float
) -> float | None:
    """A battery in ``[coarsest_s / coarse_share, finest_s)``: ``preferred_s`` if it fits, else
    the nearest end of that interval; None if the interval is empty."""
    low, high = coarsest_s / coarse_share, finest_s
    if low >= high:
        return None
    if low <= preferred_s < high:
        return preferred_s
    return low if preferred_s < low else math.nextafter(high, 0.0)


def candidates(
    *,
    area_m: float,
    cell_options_m: Sequence[float],
    level_counts: Sequence[int],
    curve: Sequence[tuple[float, float]],
    resolution_px: int,
    hfov_deg: float,
    energy: EnergyModel,
    finest_min: float,
    coarsest_max: float,
    coarse_share: float,
) -> list[Candidate]:
    """Every feasible design, best first (see the module docstring for the ranking)."""
    found = []
    for cell_m in cell_options_m:
        grid = round(area_m / cell_m)
        if not math.isclose(grid * cell_m, area_m):
            continue
        usable = []
        for n, stride in footprint_options(grid):
            gsd = sim_gsd_mpp(n * cell_m, resolution_px)
            recall = recall_at(gsd, curve)
            if recall is not None:
                usable.append((n, stride, gsd, recall))
        # One stride per footprint: the finer (overlapping) option.
        by_size: dict[int, tuple[int, int, float, float]] = {}
        for option in usable:
            if option[0] not in by_size or option[1] < by_size[option[0]][1]:
                by_size[option[0]] = option
        sizes = sorted(by_size)
        for k in level_counts:
            for combo in itertools.combinations(sizes, k):
                opts = [by_size[n] for n in combo]
                recalls = [o[3] for o in opts]
                if recalls[0] < finest_min or recalls[-1] > coarsest_max:
                    continue
                if any(b >= a for a, b in itertools.pairwise(recalls)):
                    continue  # recall must fall with every level
                table = LevelTable(
                    cell_m=cell_m,
                    grid_cells=grid,
                    levels=tuple(
                        Level(name=f"L{i + 1}", footprint_cells=o[0], stride_cells=o[1])
                        for i, o in enumerate(opts)
                    ),
                )
                costs = [coverage_energy_s(table, lvl, hfov_deg, energy) for lvl in table.levels]
                battery = choose_battery(costs[0], costs[-1], energy.battery_s, coarse_share)
                if battery is None:
                    continue
                found.append(
                    Candidate(
                        table=table,
                        altitude_m=tuple(
                            altitude_for_footprint_m(table.footprint_m(lvl), hfov_deg)
                            for lvl in table.levels
                        ),
                        gsd_mpp=tuple(o[2] for o in opts),
                        recall=tuple(recalls),
                        coverage_energy_s=tuple(costs),
                        battery_s=battery,
                        largest_recall_step=max(a - b for a, b in itertools.pairwise(recalls)),
                    )
                )
    return sorted(
        found,
        key=lambda c: (c.largest_recall_step, c.table.views_per_world(c.table.levels[0])),
    )


def unmet_recall_rules(
    curve: Sequence[tuple[float, float]], finest_min: float, coarsest_max: float
) -> list[str]:
    """The recall rules no level table can meet on ``curve``, one sentence each.

    Interpolated recall stays within the measured values, so the finest level cannot beat the
    best measured recall and the coarsest cannot go below the lowest. An empty list means both
    ends are reachable, and only the footprint, ordering or battery rules can have failed.
    """
    best_gsd, best = max(curve, key=lambda p: p[1])
    low_gsd, low = min(curve, key=lambda p: p[1])
    reasons = []
    if best < finest_min:
        reasons.append(
            f"finest-level recall >= {finest_min:g}: the best measured is {best:.2f} "
            f"(at {100 * best_gsd:g} cm/px)"
        )
    if low > coarsest_max:
        reasons.append(
            f"coarsest-level recall <= {coarsest_max:g}: the lowest measured is {low:.2f} "
            f"(at {100 * low_gsd:g} cm/px), and the curve ends at "
            f"{100 * max(g for g, _ in curve):g} cm/px (recall is never extrapolated)"
        )
    return reasons
