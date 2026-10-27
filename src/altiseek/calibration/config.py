"""Frozen M6 fit settings; test observations never select these values."""

from typing import Literal

from pydantic import Field, model_validator

from altiseek.config import Config


class SensorConfig(Config):
    schema_version: Literal[1] = 1
    boundary_rule: Literal["centre-cell"] = "centre-cell"
    calibration_unit: Literal["maximum-score-cell"] = "maximum-score-cell"
    tau: float = Field(default=0.37650254368782043, gt=0, lt=1)
    tau_min: float = Field(default=0.05, gt=0, lt=1)
    histogram_bins: int = Field(default=10, ge=2)
    reliability_bins: int = Field(default=15, ge=2)
    pseudocount: float = Field(default=0.5, gt=0)
    bootstrap_samples: int = Field(default=2000, ge=100)
    seed: int = Field(default=20261027, ge=0)
    temperature_min: float = Field(default=0.05, gt=0)
    temperature_max: float = Field(default=20.0, gt=0)
    naive_recall: float = Field(default=0.9, gt=0, lt=1)
    log_odds_limit: float = Field(default=20.0, gt=0, le=35)
    relook_rule: Literal["first-cell-level"] = "first-cell-level"

    @model_validator(mode="after")
    def check_ranges(self) -> "SensorConfig":
        if self.tau < self.tau_min or self.temperature_min >= self.temperature_max:
            raise ValueError("invalid threshold/temperature range")
        return self
