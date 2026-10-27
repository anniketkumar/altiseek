"""Predeclared validation-only baseline candidate grid."""

from typing import Literal

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
