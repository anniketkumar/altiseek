"""Schemas for ``configs/world/``: libraries (CLAUDE.md §11, D26)."""

from pathlib import Path

from altiseek.config import Config
from altiseek.data.config import DataConfig


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
