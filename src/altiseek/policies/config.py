"""Predeclared validation-only baseline candidate grid."""

from typing import Any, Literal

from pydantic import Field

from altiseek.config import Config


class BaselineRun(Config):
    seed: int = Field(default=20261027, ge=0)
    variants: tuple[Literal["V0", "V1", "V2", "V3"], ...] = ("V0", "V1", "V2", "V3")
    levels: tuple[int, ...] = (0, 1, 2, 3, 4)
    coarse_levels: tuple[int, ...] = (3, 4)
    confirm_levels: tuple[int, ...] = (0, 1)
    candidate_thresholds: tuple[float, ...] = (0.01, 0.05)
    top_share: float = Field(default=0.25, gt=0, le=1)


class SelectedBaselines(Config):
    schema_version: Literal[1] = 1
    source_run_id: str
    sensor_model_id: str
    world_set_id: str
    implementation: str
    # family -> sensor variant -> method -> fixed parameters
    settings: dict[str, dict[str, dict[str, dict[str, Any]]]]
