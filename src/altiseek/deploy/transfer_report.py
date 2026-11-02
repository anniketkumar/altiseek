"""Paired whole-world statistics for the M10 transfer study (D56).

PPO training seeds are averaged within each world before bootstrapping worlds,
so every interval treats the world as the independent unit.
"""

from collections import defaultdict
from typing import Any

import numpy as np

CONDITIONS = ("cached", "live-original", "live-refit")
GROUPS = ("B5", "PPO-17", "PPO-29", "PPO-43", "PPO")


def group_of(method: str, seed: int) -> str:
    return "B5" if method == "B5" else f"PPO-{seed}"


def world_values(rows: list[dict], metric: str, group: str, condition: str) -> dict[str, float]:
    """Per-world metric; group 'PPO' averages the three training seeds of each world."""
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["condition"] != condition:
            continue
        if row["group"] == group or (group == "PPO" and row["group"].startswith("PPO-")):
            values[row["world_id"]].append(float(row[metric]))
    if group == "PPO" and any(len(v) != 3 for v in values.values()):
        raise ValueError("PPO world mean requires all three training seeds")
    return {world: float(np.mean(v)) for world, v in values.items()}


def picks(count: int, seed: int, samples: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(count, size=(samples, count))


def summary(values: dict[str, float], seed: int, samples: int) -> dict[str, Any]:
    worlds = sorted(values)
    a = np.array([values[w] for w in worlds])
    means = a[picks(len(a), seed, samples)].mean(axis=1)
    return {
        "worlds": len(a),
        "mean": float(a.mean()),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
    }


def paired(
    first: dict[str, float], second: dict[str, float], seed: int, samples: int
) -> dict[str, Any]:
    """Mean of first minus second over shared worlds; both must cover the same worlds."""
    if set(first) != set(second):
        raise ValueError("Paired comparison needs identical world sets")
    worlds = sorted(first)
    d = np.array([first[w] - second[w] for w in worlds])
    means = d[picks(len(d), seed, samples)].mean(axis=1)
    return {
        "worlds": len(d),
        "first": float(np.mean([first[w] for w in worlds])),
        "second": float(np.mean([second[w] for w in worlds])),
        "difference": float(d.mean()),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "first_better_worlds": int((d > 0).sum()),
        "second_better_worlds": int((d < 0).sum()),
    }


def recovered_fraction(
    cached: dict[str, float],
    original: dict[str, float],
    refit: dict[str, float],
    seed: int,
    samples: int,
) -> dict[str, Any]:
    """(refit - original) / (cached - original), a ratio of world means.

    Undefined when the deployment gap is not positive or its interval includes
    zero: a ratio over an unresolved gap is not interpretable.
    """
    worlds = sorted(cached)
    if set(original) != set(worlds) or set(refit) != set(worlds):
        raise ValueError("Recovered fraction needs identical world sets")
    c, o, r = (np.array([x[w] for w in worlds]) for x in (cached, original, refit))
    index = picks(len(worlds), seed, samples)
    gap, gain = (c - o)[index].mean(axis=1), (r - o)[index].mean(axis=1)
    point_gap = float((c - o).mean())
    gap_ci = np.quantile(gap, [0.025, 0.975]).tolist()
    if point_gap <= 0 or gap_ci[0] <= 0:
        return {"defined": False, "gap": point_gap, "gap_ci95": gap_ci}
    ratios = gain[gap > 0] / gap[gap > 0]
    return {
        "defined": True,
        "gap": point_gap,
        "gap_ci95": gap_ci,
        "fraction": float((r - o).mean()) / point_gap,
        "ci95": np.quantile(ratios, [0.025, 0.975]).tolist(),
    }


def rate_difference(counts: dict[str, np.ndarray], seed: int, samples: int) -> dict[str, Any]:
    """Cluster-bootstrap rates from per-world [hits, trials, cached hits, cached trials]."""
    worlds = sorted(counts)
    a = np.array([counts[w] for w in worlds], dtype=float)
    total = a.sum(axis=0)

    def rate(hits: np.ndarray, trials: np.ndarray) -> np.ndarray:
        return np.divide(hits, trials, out=np.full_like(hits, np.nan), where=trials > 0)

    resampled = a[picks(len(worlds), seed, samples)].sum(axis=1)
    live, cached = rate(resampled[:, 0], resampled[:, 1]), rate(resampled[:, 2], resampled[:, 3])
    valid = np.isfinite(live) & np.isfinite(cached)
    return {
        "worlds": len(worlds),
        "live": {"hits": int(total[0]), "trials": int(total[1])},
        "cached": {"hits": int(total[2]), "trials": int(total[3])},
        "live_rate": float(total[0] / total[1]) if total[1] else None,
        "cached_rate": float(total[2] / total[3]) if total[3] else None,
        "difference_ci95": np.quantile(live[valid] - cached[valid], [0.025, 0.975]).tolist()
        if valid.any()
        else None,
    }
