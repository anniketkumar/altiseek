"""World specs (CLAUDE.md §11): a world is a lightweight record, not an image. The renderer
composes whatever it needs from the spec and its library.

Positions are in the world frame (origin at the south-west corner, x east, y north, metres;
CLAUDE.md §5). ``world_id`` is the first 16 hex digits of the sha256 of the spec's canonical JSON
without the ID itself, so any change to a world changes its ID.
"""

import hashlib
import json
from pathlib import Path

from altiseek.config import Config


class Gaussian(Config):
    """One anisotropic Gaussian component: mean, spread along and across ``angle_deg``
    (anticlockwise from east, the drift direction), and its weight in the mixture."""

    x_m: float
    y_m: float
    sigma_major_m: float
    sigma_minor_m: float
    angle_deg: float
    weight: float


class Prior(Config):
    """A Gaussian mixture over the area, truncated to it (``altiseek.world.sampler.prior_grid``)."""

    components: tuple[Gaussian, ...]


class Placement(Config):
    """A background patch centred at ``(x_m, y_m)``, turned ``turns`` quarter turns
    anticlockwise, then mirrored left-right if ``flip``."""

    patch_id: str
    x_m: float
    y_m: float
    turns: int
    flip: bool


class WorldObject(Config):
    """An object crop centred at ``(x_m, y_m)``: the crop's object position lands there."""

    category: str
    crop_id: str
    x_m: float
    y_m: float
    #: Anticlockwise rotation of the crop, degrees.
    rotation_deg: float
    scale: float
    #: A swimmer to be found (ground truth); everything else is a distractor.
    target: bool


class WorldSpec(Config):
    """One search world."""

    world_id: str
    seed: int
    split: str
    #: sensor-fit, rl-train, rl-val or test (CLAUDE.md §16).
    world_set: str
    family: str
    library_id: str
    area_m: float
    cell_m: float
    gsd_mpp: float
    #: Appearance condition of the background patches (``conditions.json``).
    condition: int
    #: The prior the agent receives, and the one the targets were drawn from.
    prior_given: Prior
    prior_true: Prior
    background: tuple[Placement, ...]
    objects: tuple[WorldObject, ...]

    @property
    def targets(self) -> tuple[WorldObject, ...]:
        return tuple(o for o in self.objects if o.target)

    def content_id(self) -> str:
        """The ID this spec's content implies."""
        body = self.model_dump(mode="json", exclude={"world_id"})
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def identified(self) -> "WorldSpec":
        """This spec with its ``world_id`` set from its content."""
        return self.model_copy(update={"world_id": self.content_id()})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=1) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "WorldSpec":
        """Read a spec, refusing one whose ID does not match its content."""
        spec = cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
        if spec.world_id != spec.content_id():
            raise ValueError(f"{path}: world_id {spec.world_id} does not match its content")
        return spec
