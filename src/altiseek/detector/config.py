"""Schema for ``configs/detector/``: training views, training and evaluation (CLAUDE.md §9)."""

from pathlib import Path

import numpy as np

from altiseek.config import Config
from altiseek.data.config import DataConfig
from altiseek.imaging.config import CameraSpec


class ViewSettings(Config):
    """How frames become nadir-equivalent views (``altiseek.detector.views``)."""

    #: Frames with a fitted camera model.
    camera_groups: tuple[str, ...]
    #: theta_train: source pixels farther than this from straight down are not used (D14).
    max_off_nadir_deg: float
    #: A training view is used only if at least this share of it has valid source.
    min_coverage: float
    supersample: int


class TrainViews(Config):
    """Random training views (split A), stratified by GSD (D23)."""

    #: The GSD range, metres per pixel, cut into ``gsd_bins`` bins of equal width in log-GSD.
    #: A view's GSD is drawn log-uniformly within its bin.
    gsd_range_mpp: tuple[float, float]
    gsd_bins: int
    #: Views per bin for the train and the internal-validation subsets ...
    views_per_bin: int
    val_views_per_bin: int
    #: ... dealt over the frames that can serve the bin's centre GSD, at most this many each.
    max_views_per_frame: int
    #: Share of views centred near an object.
    object_share: float
    #: Camera-pipeline augmentation: blur sigma and JPEG quality drawn uniformly per view.
    blur_sigma_px: tuple[float, float]
    jpeg_quality: tuple[int, int]
    seed: int

    @property
    def gsd_edges_mpp(self) -> tuple[float, ...]:
        """The ``gsd_bins + 1`` bin edges, metres per pixel."""
        return tuple(float(e) for e in np.geomspace(*self.gsd_range_mpp, self.gsd_bins + 1))


class EvalViews(Config):
    """Tiled evaluation views (splits B and T), one set per GSD bin."""

    gsd_bins_mpp: tuple[float, ...]
    #: A tile is used only if at least this share of it has valid source (D25: lower than for
    #: training, so frames from more videos can serve the coarse bins).
    min_coverage: float
    #: Per split and bin: at most this many of the frames that can serve the bin, spread evenly
    #: over them by image id ...
    frames_per_bin: int
    #: ... and at most this many tiles per frame (a seeded random subset).
    views_per_frame: int
    seed: int


class ModelSettings(Config):
    """Ultralytics training settings (D19)."""

    weights: str
    epochs: int
    batch: int
    patience: int
    workers: int
    #: Ultralytics augmentation switches; copy-paste stays off (CLAUDE.md §9).
    mosaic: float
    fliplr: float
    flipud: float
    degrees: float
    hsv_h: float
    hsv_s: float
    hsv_v: float
    seed: int


class DetectorConfig(Config):
    """M2 settings. Relative paths are relative to the working directory."""

    data: DataConfig
    camera: CameraSpec
    views: ViewSettings
    train_views: TrainViews
    eval_views: EvalViews
    #: Sessions of split A held out for early stopping and model selection only.
    internal_val_sessions: tuple[str, ...]
    model: ModelSettings
    #: Rendered views (regenerable, bulky; ``/tmp`` on Kaggle).
    views_dir: Path
    #: Checkpoints, metrics and metadata (kept).
    out_dir: Path
