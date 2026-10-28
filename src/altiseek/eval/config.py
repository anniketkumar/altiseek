"""M9 scenario changes affect public priors or battery, never physical cache IDs."""

from typing import Literal

from pydantic import Field, model_validator

from altiseek.config import Config
from altiseek.world.spec import Prior


class Scenario(Config):
    name: str = Field(pattern=r"^[a-z][a-z0-9_-]+$")
    battery_s: float = Field(default=500, gt=2, allow_inf_nan=False)
    shift_east_m: float = Field(default=0, allow_inf_nan=False)
    rotation_deg: float = Field(default=0, allow_inf_nan=False)
    spread_scale: float = Field(default=1, gt=0, allow_inf_nan=False)

    def prior(self, original: Prior) -> Prior:
        """Transform only the supplied prior; rotate each ellipse about its own mean.

        The environment truncates and renormalises the resulting mixture on the
        unchanged search area. Means are not clipped to the boundary.
        """
        return Prior(
            components=tuple(
                component.model_copy(
                    update={
                        "x_m": component.x_m + self.shift_east_m,
                        "angle_deg": component.angle_deg + self.rotation_deg,
                        "sigma_major_m": component.sigma_major_m * self.spread_scale,
                        "sigma_minor_m": component.sigma_minor_m * self.spread_scale,
                    }
                )
                for component in original.components
            )
        )


class EvaluationProtocol(Config):
    schema_version: Literal[1] = 1
    world_set_id: Literal["22750c80717314c3"] = "22750c80717314c3"
    sensor_model_id: Literal["fcf8cdae58ca3947"] = "fcf8cdae58ca3947"
    environment_seed: int = Field(default=20261027, ge=0)
    action_seed: int = Field(default=20261028, ge=0)
    action_repeats: Literal[3] = 3
    bootstrap_seed: int = Field(default=20261029, ge=0)
    bootstrap_samples: int = Field(default=2000, ge=1000)
    scenarios: tuple[Scenario, ...]

    @model_validator(mode="after")
    def unique_scenarios(self) -> "EvaluationProtocol":
        if not self.scenarios or len({s.name for s in self.scenarios}) != len(self.scenarios):
            raise ValueError("scenario names must be nonempty and unique")
        return self
