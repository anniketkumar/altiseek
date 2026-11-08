"""Predeclared D66 resource decision; no parameter search or test data."""

import itertools
import math


def collapsed(evaluations: list[float], step_zero: float) -> bool:
    if len(evaluations) != 4 or not all(math.isfinite(v) for v in [step_zero, *evaluations]):
        raise ValueError("D66 collapse review needs four finite early evaluations")
    below = [value <= step_zero - 0.10 for value in evaluations]
    return any(a and b for a, b in itertools.pairwise(below))
