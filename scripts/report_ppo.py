"""Regenerate M8 learning curves, all-seed comparisons and behaviour from saved B runs."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from altiseek.cache.jobs import write_json
from altiseek.calibration.model import digest
from altiseek.config import load_config
from altiseek.policies.config import SelectedBaselines
from altiseek.policies.ppo import PPOConfig
from altiseek.registry.run import sha256_file, write_metadata


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def paired_interval(values: list[float]) -> list[float]:
    a = np.asarray(values)
    rng = np.random.default_rng(20261028)
    samples = rng.integers(len(a), size=(2000, len(a)))
    return np.quantile(a[samples].mean(axis=1), [0.025, 0.975]).tolist()


def reference(method: str, variant: str) -> dict[str, dict[str, Any]]:
    selected = load_config("configs/policy/selected.yaml", SelectedBaselines)
    root = Path("data/baselines") / selected.source_run_id
    plan = read(root / "plan.json")
    provenance = read(root / "selection.meta.json")
    result = {}
    for filename in read(root / "complete.json")["files"]:
        row = read(root / filename)
        key = row["key"]
        if key["variant"] != variant or key["method"] != method:
            continue
        if key["parameters"] != selected.settings[row["family"]][variant][method]:
            continue
        if row["run_id"] != root.name or digest(key) != Path(filename).stem:
            raise ValueError("baseline episode identity mismatch")
        if sha256_file(root / filename) != provenance["inputs"][Path(filename).stem]["sha256"]:
            raise ValueError("baseline episode checksum mismatch")
        result[key["world_id"]] = row
    if len(result) != 40 or plan["smoke"]:
        raise ValueError("incomplete baseline reference")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path("runs/ppo/m8-initial"))
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    cfg = load_config("configs/policy/ppo.yaml", PPOConfig)
    baseline = read(Path("docs/baseline-results.json"))
    curves, selected, paths, configs = [], [], {}, []
    baseline_comparison = []
    paths["baseline_report"] = Path("docs/baseline-results.json")
    paths["baseline_provenance"] = (
        Path("data/baselines") / baseline["run_id"] / "selection.meta.json"
    )
    for variant in cfg.variants:
        refs = None
        for seed in cfg.seeds:
            plans = list((args.root / f"{variant}-seed{seed}").glob("*/plan.json"))
            if not plans and args.partial:
                continue
            if len(plans) != 1:
                raise ValueError(f"expected one run for {variant}/{seed}; never mix identities")
            root = plans[0].parent
            plan = read(plans[0])
            if (
                digest(plan) != root.name
                or plan["smoke"]
                or plan["benchmark"]
                or plan["variant"] != variant
                or plan["seed"] != seed
            ):
                raise ValueError("production run identity mismatch")
            if plan["config"] != cfg.model_dump(mode="json"):
                raise ValueError("configuration changed across runs")
            configs.append((plan["source_hashes"], plan["input_hashes"], plan["world_sets"]))
            if configs[-1] != configs[0]:
                raise ValueError("source/input identities differ across seeds")
            paths[f"plan_{variant}_{seed}"] = plans[0]
            history = []
            for path in sorted(root.glob("eval-*.json")):
                report = read(path)
                if len(report["episodes"]) != 40:
                    raise ValueError("incomplete validation")
                history.append(
                    {
                        k: report[k]
                        for k in ("steps", "mean_auc", "mean_found", "mean_normalised_auc")
                    }
                )
                paths[f"eval_{variant}_{seed}_{report['steps']}"] = path
            curves.append(
                {"variant": variant, "seed": seed, "run_id": root.name, "history": history}
            )
            if not (root / "complete.json").exists():
                if args.partial:
                    continue
                raise ValueError(f"training incomplete: {root}")
            completion = read(root / "complete.json")
            if completion["steps"] != cfg.total_steps:
                raise ValueError("training step count mismatch")
            best = read(root / completion["validation"])
            expected = max(
                (r for r in history if r["steps"] > 0),
                key=lambda r: (r["mean_normalised_auc"], -r["steps"]),
            )
            if best["steps"] != expected["steps"]:
                raise ValueError("checkpoint selection mismatch")
            checkpoint = root / f"step-{best['steps']:09d}" / "model.zip"
            if sha256_file(checkpoint) != completion["checkpoint_sha256"]:
                raise ValueError("selected checkpoint checksum mismatch")
            zero = read(root / "zero-belief.json")
            if zero["steps"] != best["steps"] or not zero["zero_belief"]:
                raise ValueError("zero-belief checkpoint mismatch")
            paths[f"zero_{variant}_{seed}"] = root / "zero-belief.json"
            paths[f"checkpoint_{variant}_{seed}"] = checkpoint
            if refs is None:
                refs = {method: reference(method, variant) for method in ("B2", "B3", "B5")}
                for method, episodes in refs.items():
                    baseline_comparison.append(
                        {
                            "variant": variant,
                            "method": method,
                            "auc": float(np.mean([r["metrics"]["auc"] for r in episodes.values()])),
                            "first_find_s": float(
                                np.mean([r["metrics"]["first_find_s"] for r in episodes.values()])
                            ),
                            "time_per_level_s": np.mean(
                                [r["metrics"]["time_per_level_s"] for r in episodes.values()],
                                axis=0,
                            ).tolist(),
                        }
                    )
            ids = {r["world_id"] for r in best["episodes"]}
            if ids != set(refs["B2"]) or ids != {r["world_id"] for r in zero["episodes"]}:
                raise ValueError("validation world identities differ from baselines")
            zero_delta = [
                r["metrics"]["auc"] - refs["B2"][r["world_id"]]["metrics"]["auc"]
                for r in zero["episodes"]
            ]
            metrics = {
                key: float(np.mean([r["metrics"][key] for r in best["episodes"]]))
                for key in (
                    "auc",
                    "found_fraction",
                    "success",
                    "first_find_s",
                    "false_declarations",
                )
            }
            altitude = np.mean(
                [r["metrics"]["time_per_level_s"] for r in best["episodes"]], axis=0
            ).tolist()
            opportunities = sum(
                r["coarse_cached_detection_opportunities"] for r in best["episodes"]
            )
            descents = sum(
                r["immediate_descents_after_coarse_cached_detection"] for r in best["episodes"]
            )
            finite_losses = True
            for path in root.glob("logs-*/progress.csv"):
                paths[f"log_{variant}_{seed}_{path.parent.name}"] = path
                with path.open(encoding="utf-8", newline="") as stream:
                    for record in csv.DictReader(stream):
                        for key in ("train/loss", "train/approx_kl", "train/value_loss"):
                            if record.get(key) and not np.isfinite(float(record[key])):
                                finite_losses = False
            selected.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "run_id": root.name,
                    "checkpoint": checkpoint.as_posix(),
                    "sha256": completion["checkpoint_sha256"],
                    "selected_step": best["steps"],
                    "metrics": metrics,
                    "normalised_auc": best["mean_normalised_auc"],
                    "zero_belief_auc": zero["mean_auc"],
                    "zero_minus_B2": float(np.mean(zero_delta)),
                    "zero_minus_B2_ci95": paired_interval(zero_delta),
                    "zero_belief_gate": float(np.mean(zero_delta)) <= 0,
                    "finite_optimizer_losses": finite_losses,
                    "last_three_auc_range": float(np.ptp([r["mean_auc"] for r in history[-3:]])),
                    "time_per_level_s": altitude,
                    "coarse_detection_opportunities": opportunities,
                    "immediate_descents": descents,
                    "episodes": best["episodes"],
                }
            )
    complete = len(selected) == len(cfg.seeds) * len(cfg.variants)
    across_seeds = []
    for variant in cfg.variants:
        rows = [r for r in selected if r["variant"] == variant]
        if rows:
            aucs = [r["metrics"]["auc"] for r in rows]
            across_seeds.append(
                {
                    "variant": variant,
                    "seeds": len(rows),
                    "auc_mean": float(np.mean(aucs)),
                    "auc_seed_sd": float(np.std(aucs, ddof=1)) if len(rows) > 1 else None,
                }
            )
    summary = {
        "complete": complete,
        "curves": curves,
        "selected": selected,
        "baseline_comparison": baseline_comparison,
        "across_seeds": across_seeds,
        "zero_belief_gate": all(r["zero_belief_gate"] for r in selected) if complete else None,
        "finite_optimizer_losses": (
            all(r["finite_optimizer_losses"] for r in selected) if complete else None
        ),
    }
    prefix = args.root / "progress-report" if args.partial else Path("docs/ppo-results")
    write_json(prefix.with_suffix(".json"), summary)
    figure = (
        args.root / "learning-progress.png"
        if args.partial
        else Path("docs/figures/m8_learning.png")
    )
    fig, axes = plt.subplots(
        1, len(cfg.variants), figsize=(12, 4), squeeze=False, constrained_layout=True
    )
    for ax, variant in zip(axes.flat, cfg.variants, strict=True):
        for row in curves:
            if row["variant"] == variant:
                ax.plot(
                    [r["steps"] for r in row["history"]],
                    [r["mean_auc"] for r in row["history"]],
                    marker=".",
                    label=f"seed {row['seed']}",
                )
        for method, style in (("B3", "--"), ("B5", ":")):
            auc = next(
                r["auc"]
                for r in baseline["aggregate"]
                if r["variant"] == variant and r["method"] == method
            )
            ax.axhline(auc, linestyle=style, color="grey", label=method)
        ax.set(
            title=variant,
            xlabel="Training transitions",
            ylabel="Validation AUC",
            ylim=(0, 1),
            xlim=(0, cfg.total_steps),
        )
        ax.grid(alpha=0.2)
        ax.legend()
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    lines = [
        "# M8 PPO validation",
        "",
        f"Completed training runs: {len(selected)}/{len(cfg.seeds) * len(cfg.variants)}.",
        "",
        "These are checkpoint-selection results on B, not held-out test performance. "
        "All training seeds are reported. Paired intervals resample worlds after selection; "
        "they do not correct selection bias or shared imagery.",
        "",
        "| Variant | Seed | Selected step | AUC | Found | All found | False declarations "
        "| Zero-p AUC | Zero-p minus B2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        m = row["metrics"]
        lines.append(
            f"| {row['variant']} | {row['seed']} | {row['selected_step']} | {m['auc']:.3f} "
            f"| {m['found_fraction']:.3f} | {m['success']:.3f} | {m['false_declarations']:.2f} "
            f"| {row['zero_belief_auc']:.3f} | {row['zero_minus_B2']:+.3f} |"
        )
    lines += [
        "",
        "## Across-seed comparison",
        "",
        "| Variant | Completed seeds | Mean AUC | Seed standard deviation | B3 AUC | B5 AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in across_seeds:
        refs_by_method = {
            r["method"]: r for r in baseline_comparison if r["variant"] == row["variant"]
        }
        sd = f"{row['auc_seed_sd']:.3f}" if row["auc_seed_sd"] is not None else "undefined"
        lines.append(
            f"| {row['variant']} | {row['seeds']} | {row['auc_mean']:.3f} | {sd} "
            f"| {refs_by_method['B3']['auc']:.3f} | {refs_by_method['B5']['auc']:.3f} |"
        )
    lines += [
        "",
        f"Zero-belief diagnostic gate: {summary['zero_belief_gate'] if complete else 'pending'}. "
        "A positive zero-p minus B2 triggers investigation; it must not be silently accepted.",
        "",
        "## Behaviour and stability",
        "",
        "| Variant | Seed | Mean first find (s) | Seconds at L1/L2/L3/L4/L5 "
        "| Immediate descents / coarse detections | Final three AUC range |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for row in selected:
        times = "/".join(f"{t:.1f}" for t in row["time_per_level_s"])
        lines.append(
            f"| {row['variant']} | {row['seed']} | {row['metrics']['first_find_s']:.1f} "
            f"| {times} | {row['immediate_descents']}/{row['coarse_detection_opportunities']} "
            f"| {row['last_three_auc_range']:.3f} |"
        )
    for row in baseline_comparison:
        if row["method"] == "B3":
            times = "/".join(f"{t:.1f}" for t in row["time_per_level_s"])
            lines.append(
                f"| {row['variant']} B3 | — | {row['first_find_s']:.1f} "
                f"| {times} | not logged in M7 | — |"
            )
    lines += [
        "",
        "First-find times are censored at 500 s. Coarse-detection counts use raw cached "
        "views, including resolved/suppressed detections. The final-three range describes "
        "curve variation; it is not a convergence proof. Optimizer finiteness and complete "
        "per-world trajectories are in the JSON.",
        "",
        "The zero-p diagnostic retains prior, coverage, resolved cells and vector inputs. "
        "It is an input-removal check, not a separately trained prior-only policy.",
        "",
        "![Learning curves]("
        + ("learning-progress.png" if args.partial else "figures/m8_learning.png")
        + ")",
        "",
    ]
    prefix.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    write_metadata(
        args.root / ("progress.meta.json" if args.partial else "report.meta.json"),
        stage="report-ppo",
        config=cfg,
        inputs=paths,
        outputs={
            "report": prefix.with_suffix(".json"),
            "table": prefix.with_suffix(".md"),
            "figure": figure,
        },
    )
    print(
        f"Reported {len(selected)} completed runs; zero-belief gate={summary['zero_belief_gate']}"
    )


if __name__ == "__main__":
    main()
