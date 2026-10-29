"""Paired B distractor scenarios, without changing held-out T worlds."""

from altiseek.world.config import WorldConfig, WorldSet
from altiseek.world.library import Library
from altiseek.world.sampler import sample_world
from altiseek.world.spec import WorldSpec


def stress_pair(
    seed: int, family: tuple[str, str], cfg: WorldConfig, library: Library
) -> tuple[WorldSpec, WorldSpec]:
    """Same random background/targets, different declared distractor density."""
    if library.split != "B":
        raise ValueError("distractor stress requires B imagery")
    grid = cfg.grid.model_copy(update={"distractors": {"low": (0, 3), "high": (30, 40)}})
    config = cfg.model_copy(update={"grid": grid})
    pair = []
    for density in ("low", "high"):
        world_set = WorldSet(name=f"stress-{density}", split="B", per_family=10, seed_base=5000000)
        world = sample_world(seed, (*family, density), world_set, config, library)
        count = len(world.objects) - len(world.targets)
        lo, hi = grid.distractors[density]
        if not lo <= count <= hi:
            raise ValueError("stress layout could not place the prescribed distractor count")
        pair.append(world)
    low, high = pair
    fields = ("background", "targets", "prior_given", "prior_true", "condition", "seed")
    if any(getattr(low, name) != getattr(high, name) for name in fields):
        raise ValueError("stress pair changed background, targets, appearance or prior")
    return low, high
