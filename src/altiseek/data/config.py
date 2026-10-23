"""Schema for ``configs/data/``: the M1 data stages (CLAUDE.md §7)."""

from pathlib import Path
from typing import Literal

from altiseek.config import Config


class CameraGroupRule(Config):
    """One camera group: images from ``drone`` whose size lies in the inclusive pixel ranges."""

    name: str
    drone: str
    width_px: tuple[int, int]
    height_px: tuple[int, int]

    def matches(self, drone: str, width_px: int, height_px: int) -> bool:
        """Whether an image from ``drone`` of the given size belongs to this group."""
        return (
            drone == self.drone
            and self.width_px[0] <= width_px <= self.width_px[1]
            and self.height_px[0] <= height_px <= self.height_px[1]
        )


class ImageSource(Config):
    """Local image root and the public WebDAV share that missing images are fetched from."""

    version: Literal["compressed", "uncompressed"]
    #: Images live at ``root/<official split>/<file_name>``.
    root: Path
    webdav_url: str
    share_token: str


class HorizonFitConfig(Config):
    """Focal-length calibration from the open-sea horizon (fit-camera)."""

    camera_groups: tuple[str, ...]
    #: Frames inside this latitude range see an open-sea horizon (not a shoreline).
    latitude_deg: tuple[float, float]
    #: Pitches that put the horizon inside the frame with a usable lever arm.
    pitch_deg: tuple[float, float]
    #: The logged pitch must stay within ``steady_tol_deg`` for ``steady_window_s`` either side.
    steady_window_s: float
    steady_tol_deg: float
    frames_per_video: int
    #: Acceptance: enough strips agree, and they are most of the strips that can see the line.
    min_inlier_strips: int
    min_inlier_fraction: float
    max_roll_deg: float
    bootstrap_reps: int
    seed: int


class DataConfig(Config):
    """M1 data settings. Relative paths are relative to the working directory."""

    #: ODv2 COCO annotation file per official split; only labelled splits belong here.
    annotations: dict[str, Path]
    #: Where tables, fitted camera models and metadata records are written.
    out_dir: Path
    #: Committed figures (no dataset imagery).
    figures_dir: Path
    #: Tried in order; every labelled image must match exactly one rule.
    camera_groups: tuple[CameraGroupRule, ...]
    images: ImageSource
    horizon_fit: HorizonFitConfig
