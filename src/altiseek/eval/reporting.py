"""Regenerate M9 numeric tables and figures from a complete, verified run."""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from altiseek.cache.config import CacheConfig
from altiseek.cache.jobs import load_worlds, write_json
from altiseek.calibration.model import digest
from altiseek.eval.config import EvaluationProtocol
from altiseek.eval.runner import episode_keys, read, verified_episode
from altiseek.registry.run import sha256_file

METRICS = (
    "auc",
    "found_fraction",
    "success",
    "first_find_s",
    "false_declarations",
    "energy_s",
    "path_m",
    "total_false_clear_crossings",
)


def interval(values: np.ndarray, protocol: EvaluationProtocol) -> list[float]:
    rng = np.random.default_rng(protocol.bootstrap_seed)
    indices = rng.integers(len(values), size=(protocol.bootstrap_samples, len(values)))
    return np.quantile(values[indices].mean(axis=1), [0.025, 0.975]).tolist()


def curve(records: list[dict[str, Any]], count: int, budget: float) -> np.ndarray:
    times = np.array([r["time_s"] for r in records])
    found = np.array([0] + [r["found"] / count for r in records])
    return found[np.searchsorted(times, np.linspace(0, budget, 101), side="right")]


def report(root: Path, output: Path) -> None:
    plan = read(root / "plan.json")
    if plan["smoke"] or digest(plan) != root.name:
        raise ValueError("report requires a main evaluation with a valid run identity")
    protocol = EvaluationProtocol.model_validate(plan["protocol"])
    metadata = read(root / "run.meta.json")
    manifest_path = Path(metadata["inputs"]["manifest"]["path"])
    if sha256_file(manifest_path) != plan["inputs"]["manifest"]:
        raise ValueError("frozen manifest changed")
    manifest = read(manifest_path)
    cache = CacheConfig.model_validate(plan["cache"])
    worlds = load_worlds(cache, "test")
    if [{"world_id": w.world_id, "seed": w.seed} for w in worlds] != manifest["worlds"]:
        raise ValueError("test world list mismatch")
    jobs = episode_keys(protocol, manifest, worlds)
    if read(root / "status.json") != {"completed": len(jobs), "total": len(jobs), "complete": True}:
        raise ValueError("evaluation incomplete")
    world_by_id = {w.world_id: w for w in worlds}
    budgets = {s.name: s.battery_s for s in protocol.scenarios}
    grouped = defaultdict(lambda: defaultdict(list))
    checksums = {}
    for key in jobs:
        path = root / "episodes" / f"{digest(key)}.json"
        row = verified_episode(path, root.name, key)
        if row is None:
            raise ValueError(f"missing episode {path}")
        checksums[path.name] = sha256_file(path)
        world = world_by_id[key["world_id"]]
        if row["family"] != world.family:
            raise ValueError("episode family mismatch")
        metric = row["metrics"]
        values = np.array([float(metric[m]) for m in METRICS])
        if not np.isfinite(values).all():
            raise ValueError("nonfinite metrics")
        vector = np.concatenate(
            [
                values,
                np.asarray(metric["time_per_level_s"]),
                curve(row["records"], len(world.targets), budgets[key["scenario"]]),
            ]
        )
        policy = (key["scenario"], key["variant"], key["method"], key["training_seed"])
        grouped[policy][world.world_id].append(vector)
    per_seed = {
        policy: {wid: np.mean(vectors, axis=0) for wid, vectors in by_world.items()}
        for policy, by_world in grouped.items()
    }
    combined = defaultdict(lambda: defaultdict(list))
    for (scenario, variant, method, _), by_world in per_seed.items():
        for wid, vector in by_world.items():
            combined[(scenario, variant, method)][wid].append(vector)
    means = {
        key: {wid: np.mean(vectors, axis=0) for wid, vectors in by_world.items()}
        for key, by_world in combined.items()
    }
    families = {w.world_id: "-".join(w.family.split("-")[:2]) for w in worlds}
    groups = {"all": sorted(families)} | {
        family: sorted(wid for wid, value in families.items() if value == family)
        for family in sorted(set(families.values()))
    }
    rows = []
    for (scenario, variant, method), by_world in means.items():
        for family, ids in groups.items():
            values = np.stack([by_world[wid] for wid in ids])
            metric_rows = {
                name: {"mean": float(values[:, i].mean()), "ci95": interval(values[:, i], protocol)}
                for i, name in enumerate(METRICS)
            }
            reference = means[(scenario, variant, "random")]
            oracle = means[(scenario, variant, "oracle")]
            random_auc = np.mean([reference[wid][0] for wid in ids])
            oracle_auc = np.mean([oracle[wid][0] for wid in ids])
            denominator = oracle_auc - random_auc
            seeds = {
                str(seed): {
                    name: float(np.mean([vectors[wid][i] for wid in ids]))
                    for i, name in enumerate(METRICS)
                }
                for (sc, var, meth, seed), vectors in per_seed.items()
                if (sc, var, meth) == (scenario, variant, method) and seed is not None
            }
            rows.append(
                {
                    "scenario": scenario,
                    "variant": variant,
                    "method": method,
                    "family": family,
                    "worlds": len(ids),
                    "metrics": metric_rows,
                    "training_seeds": seeds,
                    "auc_seed_sd": float(np.std([s["auc"] for s in seeds.values()], ddof=1))
                    if seeds
                    else None,
                    "normalised_auc": float((values[:, 0].mean() - random_auc) / denominator)
                    if denominator > 0
                    else None,
                    "time_per_level_s": values[:, len(METRICS) : len(METRICS) + 5]
                    .mean(axis=0)
                    .tolist(),
                    "found_curve": values[:, len(METRICS) + 5 :].mean(axis=0).tolist(),
                }
            )
    paired = []
    for key, vectors in means.items():
        scenario, variant, method = key
        comparisons = []
        if method == "PPO":
            comparisons.extend(
                (scenario, variant, baseline) for baseline in ("B1", "B2", "B3", "B4", "B5")
            )
        if variant != "V0" and method != "PPO":
            comparisons.append((scenario, "V0", method))
        if scenario != "nominal":
            comparisons.append(("nominal", variant, method))
        for reference in comparisons:
            for family, ids in groups.items():
                differences = np.array([vectors[wid][0] - means[reference][wid][0] for wid in ids])
                paired.append(
                    {
                        "method": list(key),
                        "reference": list(reference),
                        "family": family,
                        "auc_difference": float(differences.mean()),
                        "ci95": interval(differences, protocol),
                    }
                )
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "run_id": root.name,
        "episodes": len(jobs),
        "rows": rows,
        "paired": paired,
        "episode_sha256": checksums,
        "protocol": plan["protocol"],
    }
    write_json(output / "results.json", result)
    lines = [
        "# M9 existing-cache results",
        "",
        f"Run `{root.name}`; {len(jobs):,} episodes.",
        "",
        "T has no distractors. This phase does not complete every M9 ablation.",
        "",
        "AUC intervals bootstrap worlds after averaging action draws and training seeds;",
        "training-seed spread is reported separately. All checkpoints were selected on B.",
        "",
        "| Scenario | Variant | Policy | AUC [95% CI] | Found | Normalised AUC | Seed SD |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        if row["family"] != "all":
            continue
        auc = row["metrics"]["auc"]
        norm = "undefined" if row["normalised_auc"] is None else f"{row['normalised_auc']:.3f}"
        sd = "—" if row["auc_seed_sd"] is None else f"{row['auc_seed_sd']:.3f}"
        lines.append(
            f"| {row['scenario']} | {row['variant']} | {row['method']} | "
            f"{auc['mean']:.3f} [{auc['ci95'][0]:.3f}, {auc['ci95'][1]:.3f}] | "
            f"{row['metrics']['found_fraction']['mean']:.3f} | {norm} | {sd} |"
        )
    (output / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    figures(rows, output)
    print(f"Verified {len(jobs)} episodes; reports and figures: {output}", flush=True)


def figures(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for variant in ("V0", "V3"):
        selected = [r for r in rows if r["variant"] == variant and r["family"] == "all"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
        for method in ("random", "B3", "B5", "PPO", "oracle"):
            nominal = next(
                r for r in selected if r["method"] == method and r["scenario"] == "nominal"
            )
            axes[0].plot(np.linspace(0, 500, 101), nominal["found_curve"], label=method)
            sweep = [
                next(r for r in selected if r["method"] == method and r["scenario"] == s)
                for s in ("battery_tight", "nominal", "battery_generous")
            ]
            axes[1].plot(
                [250, 500, 750],
                [r["metrics"]["auc"]["mean"] for r in sweep],
                marker="o",
                label=method,
            )
        for method in ("B3", "PPO"):
            nominal = next(
                r for r in selected if r["method"] == method and r["scenario"] == "nominal"
            )
            axes[2].plot(range(1, 6), nominal["time_per_level_s"], marker="o", label=method)
        for ax, title, xlabel, ylabel in zip(
            axes,
            ("Nominal found curve", "Battery sweep", "Altitude use"),
            ("Elapsed time (s)", "Battery (s)", "Level"),
            ("Found fraction", "AUC", "Mean time (s)"),
            strict=True,
        ):
            ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
            ax.legend()
        fig.suptitle(f"M9 {variant}: frozen policies on T (no distractors)")
        fig.savefig(output / f"summary_{variant}.png", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, default=Path("docs/m9-results"))
    args = parser.parse_args()
    report(args.run, args.out)


if __name__ == "__main__":
    main()
