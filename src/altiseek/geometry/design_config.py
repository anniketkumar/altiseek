"""Schema for ``configs/levels/design.yaml``: the design-levels search (CLAUDE.md §8)."""

from pathlib import Path

from altiseek.config import Config
from altiseek.env.energy import EnergyModel
from altiseek.imaging.config import CameraSpec


class LevelDesign(Config):
    """Search space and rules for choosing levels, cell size and battery."""

    camera: CameraSpec
    energy: EnergyModel
    #: eval-detector's per-bin metrics; recall is read from rows of ``recall_split``.
    recall_csv: Path
    recall_split: str
    area_m: float
    cell_options_m: tuple[float, ...]
    level_counts: tuple[int, ...]
    finest_min_recall: float
    coarsest_max_recall: float
    coarse_share: float
