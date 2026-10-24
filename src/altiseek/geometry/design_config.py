"""Schema for ``configs/levels/design.yaml``: the design-levels search (CLAUDE.md §8)."""

from pathlib import Path
from typing import Literal

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
    #: The recall column: IoU matching (``recall``) or centre distance (``recall_centre``), which
    #: is closer to the cell-level recall of the sensor model (§10).
    recall_metric: Literal["recall", "recall_centre"]
    area_m: float
    cell_options_m: tuple[float, ...]
    level_counts: tuple[int, ...]
    finest_min_recall: float
    coarsest_max_recall: float
    coarse_share: float
