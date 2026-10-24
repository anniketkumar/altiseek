"""Schema for ``configs/detector/``: training views, training and evaluation (CLAUDE.md §9)."""

from pathlib import Path

from altiseek.config import Config
from altiseek.data.config import DataConfig
from altiseek.imaging.config import CameraSpec


class ViewSettings(Config):
    """How frames become nadir-equivalent views (``altiseek.detector.views``)."""

    #: Frames with a fitted camera model.
    camera_groups: tuple[str, ...]
    #: theta_train: source pixels farther than this from straight down are not used (D14).
    max_off_nadir_deg: float
    #: A view is used only if at least this share of it has valid source.
    min_coverage: float
    supersample: int


class TrainViews(Config):
    """Random training views (split A)."""

    #: Target GSDs are drawn log-uniformly from this range, metres per pixel.
    gsd_range_mpp: tuple[float, float]
    gsds_per_frame: int
    views_per_gsd: int
    #: Share of views centred near an object.
    object_share: float
    #: Camera-pipeline augmentation: blur sigma and JPEG quality drawn uniformly per view.
    blur_sigma_px: tuple[float, float]
    jpeg_quality: tuple[int, int]
    seed: int


class EvalViews(Config):
    """Tiled evaluation views (splits B and T), one set per GSD bin."""

    gsd_bins_mpp: tuple[float, ...]


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
