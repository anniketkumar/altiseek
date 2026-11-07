"""Audit saved B baseline episodes and select the fixed M7 grid for D65.

This module never runs an environment. Metric reconstruction uses saved records;
ground truth is used only to verify the target count, never to choose actions.
"""

import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from altiseek.calibration.model import digest
from altiseek.env.config import SearchConfig
from altiseek.policies.config import BaselineRun
from altiseek.registry.run import sha256_file
from altiseek.world.spec import WorldSpec

METHODS = ("random", "oracle", "B1", "B2", "B3", "B4", "B5")
ORIGINAL_RUN = "8cd76bdb72f965a1"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def candidates(cfg: BaselineRun) -> list[tuple[str, dict]]:
    """Exactly the declared M7 grid, used to audit every expected episode key."""
    return (
        [("random", {}), ("oracle", {}), ("B4", {}), ("B5", {})]
        + [
            (method, {"level": level, "top_share": cfg.top_share})
            for method in ("B1", "B2")
            for level in cfg.levels
        ]
        + [
            (
                "B3",
                {
                    "coarse": coarse,
                    "confirm": confirm,
                    "candidate": threshold,
                    "top_share": cfg.top_share,
                },
            )
            for coarse, confirm, threshold in itertools.product(
                cfg.coarse_levels, cfg.confirm_levels, cfg.candidate_thresholds
            )
        ]
    )


def choose(rows: list[dict]) -> dict:
    """M7 selection: mean AUC, fewer false declarations, canonical parameter JSON."""
    grouped: defaultdict[str, list] = defaultdict(list)
    for row in rows:
        grouped[json.dumps(row["key"]["parameters"], sort_keys=True)].append(row)
    best = min(
        grouped,
        key=lambda key: (
            -np.mean([r["metrics"]["auc"] for r in grouped[key]]),
            np.mean([r["metrics"]["false_declarations"] for r in grouped[key]]),
            key,
        ),
    )
    return json.loads(best)


def audit_episode(row: dict, target_count: int, config: SearchConfig) -> None:
    """Check finite metrics, endings, budget and AUC against the saved found curve."""
    metrics, records = row["metrics"], row["records"]
    budget = config.energy.battery_s
    numeric = [value for value in metrics.values() if isinstance(value, (int, float))]
    numeric.extend(metrics["time_per_level_s"])
    if not all(math.isfinite(float(v)) for v in numeric):
        raise ValueError("Non-finite episode metrics")
    if (
        not records
        or metrics["steps"] != len(records) - 1
        or not 0 <= metrics["auc"] <= metrics["found_fraction"] <= 1
        or not 0 <= metrics["elapsed_s"] <= metrics["energy_s"] + 1e-8
        or not 0 <= metrics["energy_s"] <= budget + 1e-8
        or not (
            metrics["success"]
            or metrics["energy_s"] >= budget - 1e-8
            or metrics["steps"] >= config.max_steps
        )
    ):
        raise ValueError("Invalid episode budget/ending/metrics")
    previous_time, previous_energy, previous_found, previous_false = 0.0, 0.0, 0, 0
    auc, first = 0.0, budget
    for index, record in enumerate(records):
        time, energy = record["time_s"], record["energy_s"]
        found, false = record["found"], record["false_declarations"]
        if (
            not all(math.isfinite(float(v)) for v in (time, energy, found, false))
            or not previous_time <= time <= budget + 1e-8
            or not previous_energy <= energy <= budget + 1e-8
            or not previous_found <= found <= target_count
            or false < previous_false
            or (index == 0 and record["action"] is not None)
            or (index > 0 and record["action"] not in range(6))
        ):
            raise ValueError("Invalid saved episode trajectory")
        capped = min(time, budget)
        auc += (capped - previous_time) * previous_found / target_count
        if found:
            first = min(first, capped)
        previous_time, previous_energy = capped, energy
        previous_found, previous_false = found, false
    auc += max(0, budget - previous_time) * previous_found / target_count
    expected = {
        "auc": auc / budget,
        "found_fraction": previous_found / target_count,
        "energy_s": previous_energy,
        "elapsed_s": previous_time,
        "first_find_s": first,
        "false_declarations": previous_false,
    }
    if any(not math.isclose(float(metrics[k]), v, abs_tol=1e-8) for k, v in expected.items()):
        raise ValueError("Episode metrics disagree with saved records")
    if bool(metrics["success"]) != (previous_found == target_count):
        raise ValueError("Success disagrees with found count")
    if not math.isclose(sum(metrics["time_per_level_s"]), metrics["elapsed_s"], abs_tol=1e-8):
        raise ValueError("Level-time accounting mismatch")


def verify_grid(
    root: Path,
    worlds: list[WorldSpec],
    config: SearchConfig,
    grid: BaselineRun,
    cache_ids: list[str],
    model_sha256: str,
) -> tuple[list[dict], dict]:
    if any(w.split != "B" or w.world_set != "rl-val" for w in worlds):
        raise ValueError("Baseline review requires rl-val B worlds")
    plan, completion = read_json(root / "plan.json"), read_json(root / "complete.json")
    if (
        digest(plan) != root.name
        or completion["run_id"] != root.name
        or plan["smoke"]
        or plan["implementation"] != "m7-v2-fixed-level-routing"
        or plan["world_set_id"] != "40f07c646a97efba"
        or plan["config"] != grid.model_dump(mode="json")
        or SearchConfig.model_validate(plan["search"]) != config
        or plan["cache_ids"] != cache_ids
        or plan["model_sha256"] != model_sha256
    ):
        raise ValueError("Baseline plan identity/config mismatch")
    expected = {}
    for world, variant, (method, parameters) in itertools.product(
        worlds, grid.variants, candidates(grid)
    ):
        key = {
            "world_id": world.world_id,
            "variant": variant,
            "method": method,
            "parameters": parameters,
            "seed": grid.seed + world.seed,
        }
        expected[f"{digest(key)}.json"] = (key, world)
    if (
        set(completion["files"]) != set(expected)
        or len(completion["files"]) != len(expected)
        or completion["episodes"] != len(expected)
    ):
        raise ValueError("Incomplete or duplicate baseline grid")
    hashes, rows = {}, []
    for name, (key, world) in sorted(expected.items()):
        path = root / name
        row = read_json(path)
        if row["run_id"] != root.name or row["key"] != key or row["family"] != world.family:
            raise ValueError("Baseline episode identity mismatch")
        audit_episode(row, len(world.targets), config)
        rows.append(row)
        hashes[name] = sha256_file(path)
    metadata = read_json(root / "run.meta.json")
    for name, file in (("plan", "plan.json"), ("completion", "complete.json")):
        if metadata["outputs"][name]["sha256"] != sha256_file(root / file):
            raise ValueError("Run completion checksum mismatch")
    return rows, {
        "plan_sha256": sha256_file(root / "plan.json"),
        "completion_sha256": sha256_file(root / "complete.json"),
        "episode_checksums_id": digest(hashes),
        "episode_sha256": hashes,
    }


def original_b5(
    root: Path,
    worlds: list[WorldSpec],
    config: SearchConfig,
    new_plan: dict,
    keys: list[dict],
) -> tuple[dict[str, dict], dict]:
    """Read only saved M7 B5 episodes, verifying their original raw identities."""
    if root.name != ORIGINAL_RUN:
        raise ValueError("Wrong original baseline run")
    if any(w.split != "B" or w.world_set != "rl-val" for w in worlds):
        raise ValueError("Original reference requires rl-val B")
    plan, completion = read_json(root / "plan.json"), read_json(root / "complete.json")
    selection, metadata = read_json(root / "selection.meta.json"), read_json(root / "run.meta.json")
    if (
        digest(plan) != ORIGINAL_RUN
        or completion["run_id"] != ORIGINAL_RUN
        or SearchConfig.model_validate(plan["search"]) != config
        or any(
            plan[key] != new_plan[key]
            for key in (
                "config",
                "cache_ids",
                "model_sha256",
                "world_set_id",
                "implementation",
                "smoke",
            )
        )
        or selection["inputs"]["plan"]["sha256"] != sha256_file(root / "plan.json")
    ):
        raise ValueError("Historical baseline identity/config mismatch")
    for name, file in (("plan", "plan.json"), ("completion", "complete.json")):
        if metadata["outputs"][name]["sha256"] != sha256_file(root / file):
            raise ValueError("Historical completion checksum mismatch")
    world_map = {w.world_id: w for w in worlds}
    rows, hashes = {}, {}
    for key in keys:
        if key["method"] != "B5" or key["parameters"] or key["world_id"] not in world_map:
            raise ValueError("Only paired B5 references are permitted")
        episode_id = digest(key)
        name = f"{episode_id}.json"
        if name not in completion["files"]:
            raise ValueError("Original B5 episode missing")
        path = root / name
        if sha256_file(path) != selection["inputs"][episode_id]["sha256"]:
            raise ValueError("Historical B5 checksum mismatch")
        row = read_json(path)
        if row["key"] != key or row["run_id"] != ORIGINAL_RUN:
            raise ValueError("Historical B5 pair mismatch")
        audit_episode(row, len(world_map[key["world_id"]].targets), config)
        rows[episode_id], hashes[name] = row, sha256_file(path)
    return rows, {
        "run_id": ORIGINAL_RUN,
        "episodes_reused": len(rows),
        "episode_checksums_id": digest(hashes),
        "selection_metadata_sha256": sha256_file(root / "selection.meta.json"),
    }


def interval(values: np.ndarray, draws: np.ndarray) -> list[float]:
    return np.quantile(values[draws].mean(axis=1), [0.025, 0.975]).tolist()


def summarise(rows: list[dict], old: dict[str, dict], variants: tuple[str, ...]) -> dict:
    # M7's reporter reads sorted episode IDs; preserve that reduction/tie order.
    rows = sorted(rows, key=lambda row: digest(row["key"]))
    settings, chosen, families, gaps = {}, [], [], []
    for family in sorted({row["family"] for row in rows}):
        settings[family] = {}
        for variant in variants:
            settings[family][variant] = {}
            means = {}
            for method in METHODS:
                subset = [
                    r
                    for r in rows
                    if r["family"] == family
                    and r["key"]["variant"] == variant
                    and r["key"]["method"] == method
                ]
                params = choose(subset)
                picked = [r for r in subset if r["key"]["parameters"] == params]
                settings[family][variant][method] = params
                chosen.extend(picked)
                means[method] = float(np.mean([r["metrics"]["auc"] for r in picked]))
                families.append(
                    {"family": family, "variant": variant, "method": method, "auc": means[method]}
                )
            gaps.append(
                {
                    "family": family,
                    "variant": variant,
                    "oracle_minus_random_auc": means["oracle"] - means["random"],
                }
            )
    aggregate, paired = [], []
    for variant in variants:
        by_method = {
            method: sorted(
                [
                    r
                    for r in chosen
                    if r["key"]["variant"] == variant and r["key"]["method"] == method
                ],
                key=lambda r: r["key"]["world_id"],
            )
            for method in METHODS
        }
        ids = [r["key"]["world_id"] for r in by_method["B5"]]
        if any([r["key"]["world_id"] for r in group] != ids for group in by_method.values()):
            raise ValueError("Unpaired selected worlds")
        draws = np.random.default_rng(20261106).integers(len(ids), size=(10_000, len(ids)))
        aucs = {
            method: np.array([r["metrics"]["auc"] for r in group])
            for method, group in by_method.items()
        }
        denominator = float(aucs["oracle"].mean() - aucs["random"].mean())
        for method, group in by_method.items():
            avg = float(aucs[method].mean())
            aggregate.append(
                {
                    "variant": variant,
                    "method": method,
                    "auc": avg,
                    "auc_ci95": interval(aucs[method], draws),
                    "normalised_auc": (avg - float(aucs["random"].mean())) / denominator
                    if denominator > 0
                    else None,
                    **{
                        metric: float(np.mean([r["metrics"][metric] for r in group]))
                        for metric in (
                            "found_fraction",
                            "success",
                            "false_declarations",
                            "energy_s",
                        )
                    },
                }
            )
        previous = np.array([old[digest(r["key"])]["metrics"]["auc"] for r in by_method["B5"]])
        differences = aucs["B5"] - previous
        paired.append(
            {
                "variant": variant,
                "worlds": len(ids),
                "old_auc": float(previous.mean()),
                "old_auc_ci95": interval(previous, draws),
                "new_auc": float(aucs["B5"].mean()),
                "new_auc_ci95": interval(aucs["B5"], draws),
                "delta_auc": float(differences.mean()),
                "delta_auc_ci95": interval(differences, draws),
                "per_world": [
                    {
                        "world_id": world_id,
                        "old_auc": float(a),
                        "new_auc": float(b),
                        "delta_auc": float(b - a),
                    }
                    for world_id, a, b in zip(ids, previous, aucs["B5"], strict=True)
                ],
            }
        )
    return {
        "settings": settings,
        "aggregate": aggregate,
        "family_results": families,
        "b5_paired": paired,
        "oracle_gaps": gaps,
        "sanity_passed": all(row["oracle_minus_random_auc"] > 0 for row in gaps),
    }
