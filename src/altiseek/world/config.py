"""Schemas for ``configs/world/``: libraries, world generation and rendering (CLAUDE.md §11,
D26)."""

from pathlib import Path

from altiseek.config import Config
from altiseek.data.config import DataConfig
from altiseek.geometry.levels import LevelTable
from altiseek.imaging.config import CameraSpec


class BackgroundSettings(Config):
    """Empty-sea patches."""

    #: Side of a patch at g0, pixels.
    patch_px: int
    #: Side of the rectified tiles that patches are cut from, pixels at g0.
    tile_px: int
    #: Least share of a tile with valid source for it to be rendered at all.
    tile_coverage: float
    #: Patches keep at least this far from every annotated object's ground footprint, metres.
    #: It absorbs offset boxes (docs/data_notes.md).
    object_margin_m: float
    #: ... and this far from blackened (ignore-region) pixels, pixels at g0.
    black_margin_px: int


class ObjectSettings(Config):
    """Object crops (D27)."""

    classes: tuple[str, ...]
    #: The pasted region is the box footprint grown by
    #: ``max(allowance_min_m, allowance_box_factor x box size)``, so it holds the object even when
    #: its box is offset.
    allowance_min_m: float
    allowance_box_factor: float
    #: Around it, a ring of the object's own water fades the paste into the background and gives
    #: the water's colour, pixels at g0.
    ring_px: int


class ConditionSettings(Config):
    """Appearance conditions: k-means over background patch statistics."""

    clusters: int
    #: Only this split's patches shape the clusters; the others are assigned to them.
    fit_split: str
    seed: int


class LibraryConfig(Config):
    """M4 libraries. Relative paths are relative to the working directory."""

    data: DataConfig
    camera_groups: tuple[str, ...]
    #: Built in this order; ``conditions.fit_split`` must come first.
    splits: tuple[str, ...]
    #: theta_lib: source pixels farther than this from straight down are not used (D26).
    max_off_nadir_deg: float
    #: g0: the canonical world GSD, metres per pixel (D26).
    gsd_mpp: float
    supersample: int
    #: Every ``frame_stride``-th frame of a video, so consecutive near-copies are skipped.
    frame_stride: int
    background: BackgroundSettings
    objects: ObjectSettings
    conditions: ConditionSettings
    out_dir: Path


class PriorSettings(Config):
    """Anisotropic Gaussian priors over the area (CLAUDE.md §11)."""

    #: Component means lie at least this far inside the area, metres.
    mean_margin_m: float
    #: Minor over major spread, drawn uniformly from this range.
    aspect: tuple[float, float]
    #: Share of priors with a second component.
    mixture_share: float


class FamilyGrid(Config):
    """The main grid of scenario families (CLAUDE.md §16): prior sharpness x targets x
    distractor density, named ``<prior>-<targets>-<distractors>``."""

    #: Major spread of the prior, metres: a range per sharpness.
    prior_sigma_m: dict[str, tuple[float, float]]
    #: Number of targets: a range per setting. More than one target forms a group.
    targets: dict[str, tuple[int, int]]
    #: Number of distractors: a range per density.
    distractors: dict[str, tuple[int, int]]

    def families(self) -> list[tuple[str, str, str]]:
        return [
            (p, t, d) for p in self.prior_sigma_m for t in self.targets for d in self.distractors
        ]


class WorldSet(Config):
    """A set of worlds from one split's library (CLAUDE.md §16)."""

    name: str
    split: str
    per_family: int
    #: World seeds are ``seed_base + 10_000 x family index + i``; the bases keep sets disjoint.
    seed_base: int


class WorldConfig(Config):
    """M4 world generation. Relative paths are relative to the working directory."""

    levels: LevelTable
    #: Library directory per split.
    libraries: dict[str, Path]
    prior: PriorSettings
    grid: FamilyGrid
    #: Members of a group lie within this radius of its centre, metres.
    group_radius_m: float
    #: Share of worlds whose distractors follow a lane (a random line), and the lane's half-width.
    lane_share: float
    lane_half_width_m: float
    #: Appearance conditions with fewer patches than this are not used for worlds.
    min_condition_patches: int
    #: Background patch centres lie on a grid of this step; neighbours overlap by the patch side
    #: minus the step, metres.
    patch_step_m: float
    #: Clearance between objects' opaque regions, metres.
    clearance_m: float
    #: Object crops are rotated uniformly and scaled by a factor in ``1 +- scale_jitter``.
    scale_jitter: float
    world_sets: tuple[WorldSet, ...]
    out_dir: Path


class VariantSettings(Config):
    """Variants of a view (CLAUDE.md §11): a seeded shift of less than one view pixel, in whole
    g0 pixels, and photometric jitter."""

    #: Variants rendered (and cached, M5) per view.
    count: int
    #: The largest shift, in view pixels.
    shift_view_px: float
    #: Exposure gain drawn from ``1 +- exposure``; white-balance gain per channel likewise.
    exposure: float
    white_balance: float
    #: Highlights above ``glint_threshold`` (0-255) brightened by a factor in ``+- glint``.
    glint: float
    glint_threshold: float


class RenderConfig(Config):
    """The simulated camera over composed worlds."""

    camera: CameraSpec
    levels: LevelTable
    variants: VariantSettings
