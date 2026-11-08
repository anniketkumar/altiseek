"""Publish D66 Phase 2 results from saved B-only demonstrations and pilot logs."""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from altiseek.cache.jobs import write_json
from altiseek.calibration.model import digest
from altiseek.eval.baseline_review import read_json
from altiseek.eval.pilot_review import collapsed
from altiseek.registry.run import sha256_file


def reviewed_run(root: Path, require_complete: bool) -> dict:
    plan = read_json(root / "plan.json")
    if digest(plan) != root.name or plan["variant"] != "V3" or plan["seed"] != 17:
        raise ValueError("Unexpected pilot identity")
    if set(plan["world_sets"]) != {"rl-train", "rl-val"}:
        raise ValueError("Pilot outside B train/validation scope")
    if plan["input_hashes"]["baselines"] != sha256_file(
        "docs/m12-ppo-revisit/baseline-results.json"
    ):
        raise ValueError("Pilot baseline reference changed")
    expected_worlds = set(plan["world_ids"]["rl-val"])
    reports = []
    for path in sorted(root.glob("eval-*.json")):
        saved = read_json(path)
        pairs = {(row["world_id"], row["repeat"]) for row in saved["episodes"]}
        if (
            pairs != {(world, repeat) for world in expected_worlds for repeat in range(3)}
            or len(saved["episodes"]) != 120
            or saved["action_rule"] != "sampled"
            or saved["repeats"] != 3
            or not math.isfinite(saved["mean_auc"])
            or not math.isfinite(saved["mean_normalised_auc"])
        ):
            raise ValueError("Invalid primary pilot evaluation")
        checkpoint = root / f"step-{saved['steps']:09d}"
        marker = read_json(checkpoint / "complete.json")
        if marker["steps"] != saved["steps"] or any(
            sha256_file(checkpoint / name) != checksum
            for name, checksum in marker["hashes"].items()
        ):
            raise ValueError("Pilot checkpoint integrity failure")
        argmax = read_json(root / f"argmax-{saved['steps']:09d}.json")
        reports.append(
            {
                "steps": saved["steps"],
                "mean_auc": saved["mean_auc"],
                "mean_normalised_auc": saved["mean_normalised_auc"],
                "mean_found": saved["mean_found"],
                "argmax_auc": argmax["mean_auc"],
                "primary_sha256": sha256_file(path),
            }
        )
    completion = read_json(root / "complete.json") if require_complete else None
    if require_complete:
        if [r["steps"] for r in reports] != list(range(0, 1048577, 65536)):
            raise ValueError("Incomplete 1M pilot evaluation curve")
        best = max(reports[1:], key=lambda r: (r["mean_normalised_auc"], -r["steps"]))
        if completion["selected_step"] != best["steps"] or completion["steps"] != 1048576:
            raise ValueError("Pilot checkpoint selection mismatch")
    else:
        best = max(reports[1:], key=lambda r: (r["mean_normalised_auc"], -r["steps"]))
    return {
        "run_id": root.name,
        "learning_rate": plan["config"]["learning_rate"],
        "complete": require_complete,
        "curve": reports,
        "selected": best,
        "plan_sha256": sha256_file(root / "plan.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("pretrain", type=Path)
    args = parser.parse_args()
    status = read_json(args.pilot / "status.json")
    if status["stage"] != "awaiting_owner_review":
        raise ValueError("Pilot has not completed its declared budget")
    pretrain_plan = read_json(args.pretrain / "plan.json")
    if digest(pretrain_plan) != args.pretrain.name:
        raise ValueError("Pretraining identity mismatch")
    pretrain = read_json(args.pretrain / "complete.json")
    curve = read_json(args.pretrain / "learning-curve.json")
    if len(curve) != pretrain["epochs"]:
        raise ValueError("Incomplete imitation learning curve")
    best_epoch = min(curve, key=lambda row: (row["holdout"]["loss"], row["epoch"]))
    if best_epoch["epoch"] != pretrain["best"]["epoch"]:
        raise ValueError("Imitation selection mismatch")
    weights = args.pretrain / "best/policy.pt"
    if sha256_file(weights) != status["warm_start_sha256"]:
        raise ValueError("Pilot used different cloned weights")
    collection_root = Path("data/m12/demos") / pretrain["collection_id"]
    collection = read_json(collection_root / "complete.json")
    demo_plan = read_json(collection_root / "plan.json")
    if digest(demo_plan) != collection_root.name:
        raise ValueError("Demonstration identity mismatch")
    baseline = read_json(Path("docs/m12-ppo-revisit/baseline-results.json"))
    b5 = next(
        r["auc"] for r in baseline["aggregate"] if r["variant"] == "V3" and r["method"] == "B5"
    )
    primary = reviewed_run(args.pilot / status["primary_run"], not status["collapse_triggered"])
    early = [r["mean_auc"] for r in primary["curve"] if 0 < r["steps"] <= 262144]
    if collapsed(early, primary["curve"][0]["mean_auc"]) != status["collapse_triggered"]:
        raise ValueError("Fallback decision disagrees with the predeclared rule")
    runs = [primary]
    if status["collapse_triggered"]:
        runs.append(reviewed_run(args.pilot / status["completed_run"], True))
        if runs[1]["curve"][0]["mean_auc"] != primary["curve"][0]["mean_auc"]:
            raise ValueError("Fallback did not reproduce the same step-zero clone")
    completed = runs[-1]
    clone_auc = completed["curve"][0]["mean_auc"]
    report = {
        "decision": "D66",
        "collection_id": collection_root.name,
        "demonstration_episodes": len(collection["episodes"]),
        "demonstration_transitions": collection["transitions"],
        "demonstration_bytes": collection["bytes"],
        "partition": collection["partition"],
        "pretrain_id": args.pretrain.name,
        "pretrain_epochs": len(curve),
        "selected_imitation_epoch": best_epoch,
        "imitation_curve": curve,
        "v3_b5_auc": b5,
        "clone_auc": clone_auc,
        "clone_minus_b5_auc": clone_auc - b5,
        "substantial_clone_deficit": b5 - clone_auc > 0.15,
        "fallback_triggered": status["collapse_triggered"],
        "runs": runs,
        "status_sha256": sha256_file(args.pilot / "status.json"),
        "completed_run": completed["run_id"],
        "selected_minus_b5_auc": completed["selected"]["mean_auc"] - b5,
        "selected_minus_clone_auc": completed["selected"]["mean_auc"] - clone_auc,
    }
    report["review_id"] = digest(report)
    write_json(Path("docs/m12-ppo-revisit/pilot-results.json"), report)
    figure_root = Path("docs/figures")
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for run in runs:
        ax.plot(
            [r["steps"] for r in run["curve"]],
            [r["mean_auc"] for r in run["curve"]],
            marker="o",
            label=f"sampled, lr={run['learning_rate']:g}",
        )
        ax.plot(
            [r["steps"] for r in run["curve"]],
            [r["argmax_auc"] for r in run["curve"]],
            linestyle=":",
            label=f"argmax, lr={run['learning_rate']:g}",
        )
    ax.axhline(b5, color="black", linestyle="--", label="B5, corrected cost")
    ax.set(xlabel="PPO transitions", ylabel="B rl-val mean AUC", ylim=(0, 1))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.savefig(figure_root / "m12_pilot.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for split in ("train", "holdout"):
        ax.plot([r["epoch"] for r in curve], [r[split]["loss"] for r in curve], label=split)
    ax.set(xlabel="Imitation epoch", ylabel="Masked CE + 0.5 critic MSE")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(figure_root / "m12_imitation.png", dpi=160)
    plt.close(fig)
    lines = [
        "# M12 Phase 2: B5 warm start and V3 CPU pilot",
        "",
        f"D66 review `{report['review_id']}`; Phase 2 complete, pending owner review.",
        "",
        f"Collection `{collection_root.name}`: {len(collection['episodes'])} B episodes, "
        f"{collection['transitions']:,} transitions. All 16 holdout worlds came from "
        "rl-train, disjoint from the 144 gradient-training worlds; rl-val was not used "
        "for imitation gradients or early stopping.",
        "",
        f"Actor/critic pretraining selected epoch {best_epoch['epoch']} of {len(curve)} "
        f"by holdout loss {best_epoch['holdout']['loss']:.6f}; holdout masked accuracy "
        f"{best_epoch['holdout']['accuracy']:.3f}, "
        f"critic MSE {best_epoch['holdout']['critic_mse']:.6f}.",
        "",
        "| Policy | Mean B rl-val AUC | Difference from corrected-cost B5 |",
        "|---|---:|---:|",
        f"| B5 | {b5:.6f} | 0 |",
        f"| Clone at step zero | {clone_auc:.6f} | {clone_auc - b5:+.6f} |",
        f"| Selected PPO at {completed['selected']['steps']:,} steps | "
        f"{completed['selected']['mean_auc']:.6f} | {report['selected_minus_b5_auc']:+.6f} |",
        "",
        "Primary values average three sampled-action repeats over each of 40 B rl-val "
        "worlds. Checkpoint selection uses D41 mean family-normalised AUC, ties earliest. "
        "These are selection-set results from one training seed, not held-out evidence.",
        "",
        f"Fallback triggered: **{status['collapse_triggered']}**. "
        "The decision uses exactly the declared two-consecutive-evaluations/0.10 rule "
        "within the first 262,144 steps. Any partial primary run remains in the JSON.",
        "",
        f"Substantial clone deficit (>0.15 AUC): **{report['substantial_clone_deficit']}**. "
        "Covariate shift is a possible contributor; the gap alone does not establish it. "
        "DAgger requires an owner decision and has not been added.",
        "",
        "![Imitation learning curves](../figures/m12_imitation.png)",
        "",
        "![CPU pilot learning curves](../figures/m12_pilot.png)",
        "",
        "The 500 s battery, 8.42 s B-fitted dwell, D65's explicit 30% coarse allowance "
        "and D63's retained original coverage failure remain unchanged. No test world, "
        "T seed or M10 transfer result was used. Every new negative result is retained. "
        "Stop for the owner's full-run go/no-go and 1M/2M budget decision.",
        "",
    ]
    Path("docs/m12-ppo-revisit/pilot-results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Published Phase 2 review {report['review_id']}; full runs await owner decision.")


if __name__ == "__main__":
    main()
