"""Verify the complete M7 rl-val grid, freeze selected settings and publish numeric reports."""

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from altiseek.cache.config import CacheConfig
from altiseek.cache.jobs import load_worlds, write_json
from altiseek.calibration.model import SensorModel, digest
from altiseek.config import load_config
from altiseek.policies.config import BaselineRun, SelectedBaselines
from altiseek.registry.run import sha256_file, write_metadata

METHODS = ("random", "oracle", "B1", "B2", "B3", "B4", "B5")


def choose(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean validation AUC; ties prefer fewer false declarations then canonical config."""
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[json.dumps(row["key"]["parameters"], sort_keys=True)].append(row)
    best = min(
        grouped,
        key=lambda k: (
            -np.mean([r["metrics"]["auc"] for r in grouped[k]]),
            np.mean([r["metrics"]["false_declarations"] for r in grouped[k]]),
            k,
        ),
    )
    return json.loads(best)


def interval(values: list[float], rng: np.random.Generator) -> tuple[float, list[float]]:
    a = np.asarray(values, dtype=float)
    draws = rng.integers(len(a), size=(2000, len(a)))
    return float(a.mean()), np.quantile(a[draws].mean(axis=1), [0.025, 0.975]).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--sanity", type=Path, required=True)
    parser.add_argument("--initial-check", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.root / "plan.json").read_text(encoding="utf-8"))
    completion = json.loads((args.root / "complete.json").read_text(encoding="utf-8"))
    if plan["smoke"] or plan["implementation"] != "m7-v2-fixed-level-routing":
        raise ValueError("not the corrected full validation run")
    if digest(plan) != args.root.name or completion["run_id"] != args.root.name:
        raise ValueError("run identity mismatch")
    cfg = BaselineRun.model_validate(plan["config"])
    cache = load_config("configs/cache/default.yaml", CacheConfig)
    worlds = load_worlds(cache, "rl-val")
    if plan["world_set_id"] != cache.world_set_ids["rl-val"]:
        raise ValueError("validation world-set mismatch")
    sensor_path = Path("data/sensor_models/fcf8cdae58ca3947.json")
    sensor = SensorModel.load(sensor_path, detector_id=cache.detector_id)
    if plan["model_sha256"] != sha256_file(sensor_path):
        raise ValueError("model changed since validation")
    candidates = [("random", {}), ("oracle", {}), ("B4", {}), ("B5", {})]
    candidates += [
        (method, {"level": level, "top_share": cfg.top_share})
        for method in ("B1", "B2")
        for level in cfg.levels
    ]
    candidates += [
        ("B3", {"coarse": c, "confirm": f, "candidate": t, "top_share": cfg.top_share})
        for c, f, t in itertools.product(
            cfg.coarse_levels, cfg.confirm_levels, cfg.candidate_thresholds
        )
    ]
    expected = {
        digest(
            {
                "world_id": w.world_id,
                "variant": variant,
                "method": method,
                "parameters": params,
                "seed": cfg.seed + w.seed,
            }
        )
        for w in worlds
        for variant in cfg.variants
        for method, params in candidates
    }
    if set(completion["files"]) != {f"{key}.json" for key in expected} or completion[
        "episodes"
    ] != len(expected):
        raise ValueError("incomplete/duplicate validation grid")
    world_family = {w.world_id: w.family for w in worlds}
    rows, inputs = (
        [],
        {"plan": args.root / "plan.json", "sanity": args.sanity, "model": sensor_path},
    )
    for key in sorted(expected):
        path = args.root / f"{key}.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        if digest(row["key"]) != key or row["run_id"] != args.root.name:
            raise ValueError("episode identity mismatch")
        if row["family"] != world_family[row["key"]["world_id"]]:
            raise ValueError("episode family mismatch")
        rows.append(row)
        inputs[key] = path
    sanity = json.loads(args.sanity.read_text(encoding="utf-8"))
    initial = json.loads(args.initial_check.read_text(encoding="utf-8"))
    if (
        initial["source_summary_sha256"] != sha256_file(args.sanity)
        or initial["source_run_id"] != sanity["run_id"]
        or initial["threshold"] != 0.01
        or initial["sensor_model_id"] != sensor.sensor_model_id
        or initial["world_set_id"] != plan["world_set_id"]
    ):
        raise ValueError("initial-observation diagnostic mismatch")
    inputs["initial_check"] = args.initial_check
    sanity["initial_observation"] = initial
    sanity["total_false_clear_crossings"] = (
        sanity["v0_false_clear_crossings"] + initial["initial_crossings"]
    )
    sanity["passed_including_initial"] = (
        sanity["coverage_complete"]
        and sanity["total_false_clear_crossings"] > 0
        and all(row["fine_exceeds_coarse"] for row in sanity["variants"].values())
    )
    sanity_plan = json.loads(args.sanity.with_name("plan.json").read_text(encoding="utf-8"))
    if (
        digest(sanity_plan) != sanity["run_id"]
        or sanity_plan["sensor_model_id"] != sensor.sensor_model_id
        or sanity_plan["world_set_id"] != plan["world_set_id"]
    ):
        raise ValueError("sanity identity mismatch")
    families = sorted(world_family.values())
    families = sorted(set(families))
    settings, chosen, family_rows = {}, [], []
    for family in families:
        settings[family] = {}
        for variant in cfg.variants:
            settings[family][variant] = {}
            for method in METHODS:
                subset = [
                    r
                    for r in rows
                    if r["family"] == family
                    and r["key"]["variant"] == variant
                    and r["key"]["method"] == method
                ]
                params = choose(subset)
                settings[family][variant][method] = params
                picked = [r for r in subset if r["key"]["parameters"] == params]
                chosen.extend(picked)
                family_rows.append(
                    {
                        "family": family,
                        "variant": variant,
                        "method": method,
                        "auc": float(np.mean([r["metrics"]["auc"] for r in picked])),
                    }
                )
    oracle_gaps = []
    for family in families:
        for variant in cfg.variants:
            indexed = {
                r["method"]: r["auc"]
                for r in family_rows
                if r["family"] == family and r["variant"] == variant
            }
            oracle_gaps.append(
                {"family": family, "variant": variant, "gap": indexed["oracle"] - indexed["random"]}
            )
    aggregate = []
    rng = np.random.default_rng(cfg.seed)
    for variant in cfg.variants:
        means = {
            method: float(
                np.mean(
                    [
                        r["metrics"]["auc"]
                        for r in chosen
                        if r["key"]["variant"] == variant and r["key"]["method"] == method
                    ]
                )
            )
            for method in METHODS
        }
        for method in METHODS:
            picked = [
                r for r in chosen if r["key"]["variant"] == variant and r["key"]["method"] == method
            ]
            avg, ci = interval([r["metrics"]["auc"] for r in picked], rng)
            den = means["oracle"] - means["random"]
            aggregate.append(
                {
                    "variant": variant,
                    "method": method,
                    "auc": avg,
                    "auc_ci95": ci,
                    "normalised_auc": (avg - means["random"]) / den if den > 0 else None,
                    **{
                        metric: float(np.mean([r["metrics"][metric] for r in picked]))
                        for metric in (
                            "found_fraction",
                            "success",
                            "false_declarations",
                            "energy_s",
                        )
                    },
                }
            )
    step_count = sum(r["metrics"]["steps"] for r in rows)
    env_seconds = sum(r["timing"]["environment_s"] for r in rows)
    replay_path = args.root / "replay.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    if (
        replay["run_id"] != args.root.name
        or replay["pilot"]
        or replay["episodes"] != len(worlds) * len(cfg.variants) * 2
        or not replay["outcomes_unchanged"]
    ):
        raise ValueError("incomplete or mismatched optimized-environment replay")
    inputs["replay"] = replay_path
    summary = {
        "run_id": args.root.name,
        "episodes": len(rows),
        "aggregate": aggregate,
        "family_results": family_rows,
        "oracle_gaps": oracle_gaps,
        "sanity": sanity,
        "optimized_environment_replay": replay,
        "environment_plus_logging_ms_per_step": 1000 * env_seconds / max(1, step_count),
        "passed": sanity["passed_including_initial"] and all(r["gap"] > 0 for r in oracle_gaps),
    }
    write_json(Path("docs/baseline-results.json"), summary)
    lines = [
        "# M7 baseline validation",
        "",
        f"Run `{args.root.name}`; {len(rows):,} episodes; "
        "40 rl-val worlds, five per family. Sensor model `fcf8cdae58ca3947`.",
        "",
        "**These are selection-set results, not held-out estimates.** B1-B3 settings were "
        "selected per family and sensor variant by validation AUC. The bootstrap intervals "
        "resample worlds after selection and do not correct selection bias or shared imagery.",
        "",
        "| Sensor | Policy | AUC [95% bootstrap CI] | Found fraction | Success "
        "| False declarations "
        "| Normalised AUC |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lo, hi = row["auc_ci95"]
        normalised = (
            f"{row['normalised_auc']:.3f}" if row["normalised_auc"] is not None else "undefined"
        )
        lines.append(
            f"| {row['variant']} | {row['method']} | {row['auc']:.3f} [{lo:.3f}, {hi:.3f}] "
            f"| {row['found_fraction']:.3f} | {row['success']:.3f} "
            f"| {row['false_declarations']:.2f} | {normalised} |"
        )
    lines += [
        "",
        "Normalisation above uses paired mean oracle/random AUCs over the full validation "
        "set for each sensor variant. The oracle is a finest-level perfect-verification "
        "reference, not a per-episode upper bound. Family-mean gaps are saved in the JSON.",
        "",
        "## Sanity checks",
        "",
        f"Oracle exceeds random in every family/variant: {all(r['gap'] > 0 for r in oracle_gaps)} "
        f"(minimum mean AUC gap {min(r['gap'] for r in oracle_gaps):.3f}).",
        "",
        "| Sensor | Fine full-sweep found fraction | Coarse full-sweep found fraction |",
        "|---|---:|---:|",
    ]
    for variant, row in sanity["variants"].items():
        lines.append(f"| {variant} | {row['fine_found']:.3f} | {row['coarse_found']:.3f} |")
    lines += [
        "",
        "All diagnostic sweeps covered the full lattice or found everyone: "
        f"{sanity['coverage_complete']}. "
        "A 2,000 s budget and 600-step cap make a full fixed-level sweep feasible.",
        "",
        "V0 coarse negative observations drove "
        f"{sanity['total_false_clear_crossings']} target-cell "
        "observations from p >= 0.01 to p < 0.01, including initial observations. "
        f"Of these, {initial['initial_crossings']} occurred at reset and were omitted by the "
        "original step-only diagnostic. Its failed report is retained; a separate check completes "
        "the measurement without changing the threshold or rerunning the sweeps. "
        f"This is weak evidence: {initial['initially_below_threshold']} of {initial['targets']} "
        "true-target priors already start below 0.01. These do not count as threshold crossings.",
        "",
        "Observed environment-plus-logging time: "
        f"{summary['environment_plus_logging_ms_per_step']:.3f} "
        "ms/step, excluding reset and policy planning; measured with concurrent CPU checks.",
        "",
        f"The optimized core environment replayed all {replay['episodes']} random/oracle "
        f"episodes ({replay['steps']:,} steps) without changing saved outcomes. "
        f"Core step time: mean {replay['core_step_mean_ms']:.3f} ms, "
        f"median {replay['core_step_median_ms']:.3f} ms, "
        f"95th percentile {replay['core_step_p95_ms']:.3f} ms. "
        "This excludes reset, logging and policy planning, so it is not directly comparable "
        "to the sweep timing above. The sub-millisecond aim is not met by the full replay; "
        "M8 must measure end-to-end training throughput before choosing worker counts.",
        "",
        "The PPO zero-belief-channel sanity check belongs to M8, once PPO exists. No test "
        "worlds or detector inference were used for M7 validation.",
        "",
        "![Validation AUC](figures/m7_auc.png)",
        "",
    ]
    Path("docs/baseline-results.md").write_text("\n".join(lines), encoding="utf-8")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, variant in zip(axes.flat, cfg.variants, strict=True):
        selected = [r for r in aggregate if r["variant"] == variant]
        means = np.array([r["auc"] for r in selected])
        ci = np.array([r["auc_ci95"] for r in selected]).T
        ax.bar(METHODS, means, yerr=np.maximum(0, [means - ci[0], ci[1] - means]), capsize=3)
        ax.set(title=variant, ylabel="Validation AUC (selected on these worlds)", ylim=(0, 1))
        ax.grid(axis="y", alpha=0.2)
    figure = Path("docs/figures/m7_auc.png")
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    if not summary["passed"]:
        raise ValueError("sanity checks failed; wrote diagnostics but did not freeze settings")
    frozen = SelectedBaselines(
        source_run_id=args.root.name,
        sensor_model_id=sensor.sensor_model_id,
        world_set_id=plan["world_set_id"],
        implementation=plan["implementation"],
        settings=settings,
    )
    selected_path = Path("configs/policy/selected.yaml")
    selected_path.write_text(
        yaml.safe_dump(frozen.model_dump(mode="json"), sort_keys=True), encoding="utf-8"
    )
    write_metadata(
        args.root / "selection.meta.json",
        stage="select-baselines",
        config=frozen,
        inputs=inputs,
        outputs={
            "selected": selected_path,
            "report": Path("docs/baseline-results.json"),
            "table": Path("docs/baseline-results.md"),
            "figure": figure,
        },
    )
    print(f"Verified {len(rows)} episodes; froze {selected_path}; checks passed.")


if __name__ == "__main__":
    main()
