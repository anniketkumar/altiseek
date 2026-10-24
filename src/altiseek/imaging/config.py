"""Schema for ``configs/camera/``: the simulated camera (CLAUDE.md §8)."""

from altiseek.config import Config
from altiseek.imaging.pipeline import CameraPipeline


class CameraSpec(Config):
    """The simulated nadir camera; provisional until M3 fixes it from the recall curve (D7)."""

    #: Square views of ``resolution_px`` x ``resolution_px`` (the detector input, R).
    resolution_px: int
    #: Horizontal field of view in degrees (theta).
    hfov_deg: float
    pipeline: CameraPipeline
