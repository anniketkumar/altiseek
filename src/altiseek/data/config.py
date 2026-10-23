"""Schema for ``configs/data/``: the prepare-data stage (CLAUDE.md §7)."""

from pathlib import Path

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


class DataConfig(Config):
    """prepare-data settings. Relative paths are relative to the working directory."""

    #: ODv2 COCO annotation file per official split; only labelled splits belong here.
    annotations: dict[str, Path]
    #: Where the frame and box tables and the metadata record are written.
    out_dir: Path
    #: Tried in order; every labelled image must match exactly one rule.
    camera_groups: tuple[CameraGroupRule, ...]
