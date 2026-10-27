"""Search-loop settings, separate from the frozen sensor fit."""

from typing import Literal

from pydantic import Field

from altiseek.config import Config
from altiseek.env.energy import EnergyModel


class SearchConfig(Config):
    energy: EnergyModel
    sensor_variant: Literal["V0", "V1", "V2", "V3"] = "V3"
    start_level: int = Field(default=4, ge=0)
    max_steps: int = Field(default=300, ge=1)
    declare_threshold: float = Field(default=0.5, gt=0, lt=1)
    suppression_radius_m: float = Field(default=1.0, ge=0)
    false_declaration_cost: float = Field(default=0.5, ge=0)
    known_count: bool = True
    assumed_count: float = Field(default=2.0, gt=0)
    max_targets: int = Field(default=16, ge=1)
    coordinate_channels: bool = True
    shaping_weight: float = Field(default=0.0, ge=0)
    discount: float = Field(default=0.99, gt=0, le=1)
