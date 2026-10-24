"""Scenario sampler (CLAUDE.md §11): the prior, targets, distractors and background layout of a
world, drawn from one split's library with an explicit seed.

- **Prior:** one or two anisotropic Gaussian components with means inside the area, truncated to
  it. :func:`prior_grid` discretises it to cells, the per-target cell distribution q of the belief
  map (CLAUDE.md §13).
- **Targets:** K swimmers. One target is drawn from the prior; several form a group, whose centre
  is drawn from the prior and whose members lie within ``group_radius_m`` of it. Targets are
  drawn over the area minus their own radius, a sliver tighter than the grid's truncation.
- **Distractors:** crops of the other classes, uniform over the area or along a random lane.
- **Spacing:** every object's whole crop lies inside the area, and opaque regions keep
  ``clearance_m`` apart; rotation is uniform, so radii are those of the crops' bounding circles.
- **Background:** patches of one appearance condition on a grid, each with a random quarter turn
  and mirror.
"""

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from altiseek.world.config import PriorSettings, WorldConfig, WorldSet
from altiseek.world.library import Library, ObjectRow
from altiseek.world.spec import Gaussian, Placement, Prior, WorldObject, WorldSpec

SWIMMER = "swimmer"
_TRIES = 500


def density(component: Gaussian, x_m: NDArray[np.float64], y_m: NDArray[np.float64]) -> NDArray:
    """The component's normalised probability density at ground points."""
    a = math.radians(component.angle_deg)
    dx, dy = np.asarray(x_m) - component.x_m, np.asarray(y_m) - component.y_m
    along = dx * math.cos(a) + dy * math.sin(a)
    across = -dx * math.sin(a) + dy * math.cos(a)
    su, sv = component.sigma_major_m, component.sigma_minor_m
    return np.exp(-0.5 * ((along / su) ** 2 + (across / sv) ** 2)) / (2 * math.pi * su * sv)


def prior_grid(prior: Prior, area_m: float, cell_m: float, sub: int = 4) -> NDArray[np.float64]:
    """Per-cell probability ``q[ix, iy]`` of the mixture truncated to the area: densities at
    ``sub x sub`` points in each cell, normalised to sum to one."""
    g = round(area_m / cell_m)
    offsets = (np.arange(sub) + 0.5) / sub * cell_m
    coords = (np.arange(g)[:, None] * cell_m + offsets[None, :]).ravel()
    x, y = np.meshgrid(coords, coords, indexing="ij")
    mass = sum(c.weight * density(c, x, y) for c in prior.components)
    q = np.asarray(mass).reshape(g, sub, g, sub).sum(axis=(1, 3))
    return q / q.sum()


def sample_prior(
    rng: np.random.Generator,
    sigma_range_m: tuple[float, float],
    settings: PriorSettings,
    area_m: float,
) -> Prior:
    """One or (with probability ``mixture_share``) two components, means inside the area."""
    if rng.random() < settings.mixture_share:
        w = float(rng.uniform(0.3, 0.7))
        weights = (w, 1.0 - w)
    else:
        weights = (1.0,)
    lo, hi = settings.mean_margin_m, area_m - settings.mean_margin_m
    components = []
    for weight in weights:
        major = float(rng.uniform(*sigma_range_m))
        components.append(
            Gaussian(
                x_m=float(rng.uniform(lo, hi)),
                y_m=float(rng.uniform(lo, hi)),
                sigma_major_m=major,
                sigma_minor_m=major * float(rng.uniform(*settings.aspect)),
                angle_deg=float(rng.uniform(0.0, 180.0)),
                weight=weight,
            )
        )
    return Prior(components=tuple(components))


def draw_point(
    prior: Prior, rng: np.random.Generator, lo_m: float, hi_m: float
) -> tuple[float, float] | None:
    """A point from the prior truncated to the square ``[lo, hi]^2``, by rejection."""
    weights = np.array([c.weight for c in prior.components])
    for _ in range(_TRIES):
        c = prior.components[int(rng.choice(len(weights), p=weights / weights.sum()))]
        along, across = rng.normal(0.0, c.sigma_major_m), rng.normal(0.0, c.sigma_minor_m)
        a = math.radians(c.angle_deg)
        x = c.x_m + along * math.cos(a) - across * math.sin(a)
        y = c.y_m + along * math.sin(a) + across * math.cos(a)
        if lo_m <= x <= hi_m and lo_m <= y <= hi_m:
            return float(x), float(y)
    return None


class _Layout:
    """Objects placed so far, with the spacing and inside-the-area rules."""

    def __init__(self, area_m: float, clearance_m: float, library: Library) -> None:
        self.area_m, self.clearance_m, self.library = area_m, clearance_m, library
        self.placed: list[tuple[float, float, float]] = []  # x, y, opaque radius

    def radii(self, crop: ObjectRow, scale: float) -> tuple[float, float]:
        """Bounding-circle radii of the crop's opaque region and of the whole crop, metres."""
        g = self.library.gsd_mpp * scale * math.sqrt(2.0)
        return max(crop.size_px / 2 - self.library.ring_px, 0) * g, crop.size_px / 2 * g

    def fits(self, x: float, y: float, opaque_m: float, full_m: float) -> bool:
        if not (full_m <= x <= self.area_m - full_m and full_m <= y <= self.area_m - full_m):
            return False
        return all(
            math.hypot(x - px, y - py) >= opaque_m + pr + self.clearance_m
            for px, py, pr in self.placed
        )

    def add(self, x: float, y: float, opaque_m: float) -> None:
        self.placed.append((x, y, opaque_m))


def _pick(rng: np.random.Generator, crops: Sequence[ObjectRow]) -> ObjectRow:
    return crops[int(rng.integers(len(crops)))]


def _targets(
    rng: np.random.Generator,
    k: int,
    prior: Prior,
    swimmers: Sequence[ObjectRow],
    layout: _Layout,
    cfg: WorldConfig,
) -> list[WorldObject]:
    """K targets: one drawn from the prior, or a group around a centre drawn from it. A group
    that does not fit is redrawn whole."""
    for _ in range(_TRIES):
        chosen = [_pick(rng, swimmers) for _ in range(k)]
        poses = [
            (float(rng.uniform(0.0, 360.0)), 1.0 + float(rng.uniform(-1, 1)) * cfg.scale_jitter)
            for _ in range(k)
        ]
        radii = [layout.radii(c, s) for c, (_, s) in zip(chosen, poses, strict=True)]
        edge = max(full for _, full in radii) + (cfg.group_radius_m if k > 1 else 0.0)
        centre = draw_point(prior, rng, edge, layout.area_m - edge)
        if centre is None:
            continue
        trial = _Layout(layout.area_m, layout.clearance_m, layout.library)
        trial.placed = list(layout.placed)
        objects = []
        for crop, (rotation, scale), (opaque, full) in zip(chosen, poses, radii, strict=True):
            spot = None
            for _ in range(_TRIES if k > 1 else 1):
                if k == 1:
                    x, y = centre
                else:
                    r = cfg.group_radius_m * math.sqrt(rng.random())
                    t = rng.uniform(0.0, 2 * math.pi)
                    x, y = centre[0] + r * math.cos(t), centre[1] + r * math.sin(t)
                if trial.fits(x, y, opaque, full):
                    spot = (float(x), float(y))
                    break
            if spot is None:
                break
            trial.add(*spot, opaque)
            objects.append(
                WorldObject(
                    category=crop.category,
                    crop_id=crop.crop_id,
                    x_m=spot[0],
                    y_m=spot[1],
                    rotation_deg=rotation,
                    scale=scale,
                    target=True,
                )
            )
        if len(objects) == k:
            layout.placed = trial.placed
            return objects
    raise RuntimeError(f"could not place {k} targets")


def _distractors(
    rng: np.random.Generator, n: int, crops: Sequence[ObjectRow], layout: _Layout, cfg: WorldConfig
) -> list[WorldObject]:
    """Up to ``n`` distractors, uniform over the area or along a lane; one that finds no room
    after many tries is dropped."""
    if not crops or n == 0:
        return []
    area = layout.area_m
    lane = rng.random() < cfg.lane_share
    origin = rng.uniform(0.0, area, 2)
    heading = rng.uniform(0.0, math.pi)
    out = []
    for _ in range(n):
        crop = _pick(rng, crops)
        rotation = float(rng.uniform(0.0, 360.0))
        scale = 1.0 + float(rng.uniform(-1, 1)) * cfg.scale_jitter
        opaque, full = layout.radii(crop, scale)
        for _ in range(_TRIES):
            if lane:
                along = rng.uniform(-area, area)
                side = rng.uniform(-cfg.lane_half_width_m, cfg.lane_half_width_m)
                x = origin[0] + along * math.cos(heading) - side * math.sin(heading)
                y = origin[1] + along * math.sin(heading) + side * math.cos(heading)
            else:
                x, y = rng.uniform(0.0, area, 2)
            if layout.fits(float(x), float(y), opaque, full):
                layout.add(float(x), float(y), opaque)
                out.append(
                    WorldObject(
                        category=crop.category,
                        crop_id=crop.crop_id,
                        x_m=float(x),
                        y_m=float(y),
                        rotation_deg=rotation,
                        scale=scale,
                        target=False,
                    )
                )
                break
    return out


def background_layout(
    rng: np.random.Generator, pool: Sequence[str], area_m: float, step_m: float, patch_m: float
) -> tuple[Placement, ...]:
    """Patches from ``pool`` centred on a grid of ``step_m`` covering the area."""
    if patch_m < step_m:
        raise ValueError(f"{patch_m} m patches leave gaps on a {step_m} m grid")
    centres = step_m / 2 + step_m * np.arange(math.ceil(area_m / step_m))
    return tuple(
        Placement(
            patch_id=pool[int(rng.integers(len(pool)))],
            x_m=float(x),
            y_m=float(y),
            turns=int(rng.integers(4)),
            flip=bool(rng.integers(2)),
        )
        for y in centres
        for x in centres
    )


def sample_world(
    seed: int,
    family: tuple[str, str, str],
    world_set: WorldSet,
    cfg: WorldConfig,
    library: Library,
) -> WorldSpec:
    """The world of ``seed`` in ``family`` (prior, targets, distractors) from ``library``."""
    if library.split != world_set.split:
        raise ValueError(f"world set {world_set.name} needs split {world_set.split} imagery")
    rng = np.random.default_rng(seed)
    area = cfg.levels.area_m
    prior_name, target_name, distractor_name = family
    usable = [
        c for c, s in enumerate(library.conditions) if s["count"] >= cfg.min_condition_patches
    ]
    condition = usable[int(rng.integers(len(usable)))]
    pool = [p.patch_id for p in library.backgrounds if p.condition == condition]
    background = background_layout(
        rng, pool, area, cfg.patch_step_m, library.patch_px * library.gsd_mpp
    )
    prior = sample_prior(rng, cfg.grid.prior_sigma_m[prior_name], cfg.prior, area)
    layout = _Layout(area, cfg.clearance_m, library)
    swimmers = [o for o in library.objects if o.category == SWIMMER]
    others = [o for o in library.objects if o.category != SWIMMER]
    lo, hi = cfg.grid.targets[target_name]
    objects = _targets(rng, int(rng.integers(lo, hi + 1)), prior, swimmers, layout, cfg)
    lo, hi = cfg.grid.distractors[distractor_name]
    objects += _distractors(rng, int(rng.integers(lo, hi + 1)), others, layout, cfg)
    return WorldSpec(
        world_id="",
        seed=seed,
        split=library.split,
        world_set=world_set.name,
        family="-".join(family),
        library_id=library.library_id,
        area_m=area,
        cell_m=cfg.levels.cell_m,
        gsd_mpp=library.gsd_mpp,
        condition=condition,
        prior_given=prior,
        prior_true=prior,
        background=background,
        objects=tuple(objects),
    ).identified()
