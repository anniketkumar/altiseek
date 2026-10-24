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


class Site(Config):
    """A capture site, as a latitude/longitude box in degrees."""

    name: str
    latitude_deg: tuple[float, float]
    longitude_deg: tuple[float, float]

    def contains(self, latitude_deg: float, longitude_deg: float) -> bool:
        return (
            self.latitude_deg[0] <= latitude_deg <= self.latitude_deg[1]
            and self.longitude_deg[0] <= longitude_deg <= self.longitude_deg[1]
        )


class SessionConfig(Config):
    """Sessions are capture days across camera groups (``altiseek.data.sessions``)."""

    sites: tuple[Site, ...]
    #: Campaign of each undated session, by camera group.
    undated_campaign: dict[str, str]


class SplitConfig(Config):
    """Session-level A/B/T split (``altiseek.data.splits``)."""

    #: Target share of the usable frames in each split; must sum to 1.
    target_share: dict[Literal["A", "B", "T"], float]
    #: Frames that drive the balance: groups with a camera model, hence a GSD.
    usable_camera_groups: tuple[str, ...]
    #: Altitude band edges in metres, for stratification.
    altitude_bands_m: tuple[float, ...]
    #: Off-nadir limit for frames that M4 libraries can use.
    near_nadir_deg: float
    #: B and T need near-nadir swimmer boxes for libraries and the fidelity check.
    min_near_nadir_swimmers: int
    #: Undated sessions share a split with every session of their campaign.
    tie_undated_to_campaign: bool
    #: Campaigns of the target domain; B and T sessions must come from these.
    eval_campaigns: tuple[str, ...]


class LeakageConfig(Config):
    """check-leakage: thresholds come from known duplicates, never from the cross-split result."""

    #: Thresholds flag this share of known duplicates (consecutive frames of one video).
    calibration_quantile: float
    #: Consecutive frames at most this far apart count as known duplicates.
    positive_max_gap_s: float
    #: Known non-duplicates: pairs across these two campaigns.
    negative_campaigns: tuple[str, str]
    negative_samples: int
    seed: int


class KeepRule(Config):
    """Which scanned images to keep on disk for later stages (the rest are discarded)."""

    camera_groups: tuple[str, ...]
    #: Keep frames whose optical axis is at most this far from straight down.
    max_off_nadir_deg: float


class ScanConfig(Config):
    """scan-images: one streaming pass over every labelled image."""

    workers: int
    chunk_size: int
    keep: KeepRule


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
    scan: ScanConfig
    sessions: SessionConfig
    splits: SplitConfig
    leakage: LeakageConfig
