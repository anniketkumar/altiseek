"""Cache generation and inference settings; no untracked inference defaults."""

from pathlib import Path

from pydantic import Field

from altiseek.config import Config
from altiseek.world.config import RenderConfig


class InferenceSettings(Config):
    iou: float = Field(default=0.7, gt=0, le=1)
    max_det: int = Field(default=300, gt=0)


class CacheConfig(Config):
    render: RenderConfig
    detector_id: str
    checkpoint: Path
    tau_min: float = Field(default=0.05, gt=0, lt=1)
    inference: InferenceSettings = InferenceSettings()
    batch_size: int = Field(default=16, gt=0)
    render_workers: int = Field(default=2, gt=0)
    worlds_dir: Path = Path("data/worlds")
    libraries: dict[str, Path]
    library_ids: dict[str, str]
    world_set_ids: dict[str, str]
    out_dir: Path = Path("data/caches")
