"""Report completed frozen-policy and controlled-learning diagnostics without rerunning them."""

import json
from collections import defaultdict
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


def reward_audit(records: list[dict[str, Any]]) -> dict[str, float]:
    steps = records[1:]
    rewards = np.array([r["reward"] for r in steps])
    false = np.array([r["events"]["false"] for r in steps])
    factors = 0.99 ** np.arange(len(steps))
    return {
        "discounted_return": float(np.sum(rewards * factors)),
        "without_false_penalty": float(np.sum((rewards + 0.5 * false) * factors)),
    }


def interval(values: list[float]) -> list[float]:
    data = np.asarray(values)
    rng = np.random.default_rng(20261028)
    sampled = rng.integers(len(data), size=(2000, len(data)))
    return np.quantile(data[sampled].mean(axis=1), [0.025, 0.975]).tolist()


def main() -> None:
    root = Path("data/ppo-diagnosis/6493069c0f288c5c")
    probe = Path("data/ppo-diagnosis/fe40bd44ec290dd2")
    plan, diagnostics = read(root / "plan.json"), read(root / "summary.json")
    probe_plan, completion = read(probe / "plan.json"), read(probe / "complete.json")
    if (
        digest(plan) != root.name
        or digest(probe_plan) != probe.name
        or completion["steps"] != 32768
    ):
        raise ValueError("diagnostic identity/completion mismatch")
    selected_path = Path("runs/ppo/m8-initial/selected-results.json")
    if sha256_file(selected_path) != plan["selected_report_sha256"]:
        raise ValueError("original selected results changed")
    selected = read(selected_path)
    paths = {
        "selected": selected_path,
        "diagnostics": root / "summary.json",
        "probe_plan": probe / "plan.json",
        "probe_completion": probe / "complete.json",
    }
    episodes = []
    for path in root.glob("*.json"):
        if len(path.stem) != 16:
            continue
        row = read(path)
        if digest(row["key"]) != path.stem:
            raise ValueError("diagnostic episode identity mismatch")
        episodes.append(row)
        paths[path.stem] = path
    if len(episodes) != 960:
        raise ValueError("expected 240 replay and 720 sampled episodes")
    effects = []
    for variant in ("V0", "V3"):
        variant_rows = [r for r in episodes if r["key"]["variant"] == variant]
        paired = defaultdict(lambda: defaultdict(list))
        for row in variant_rows:
            paired[row["key"]["world_id"]][row["key"]["mode"]].append(row["metrics"]["auc"])
        if len(paired) != 40 or any(
            len(r["deterministic"]) != 3 or len(r["stochastic"]) != 9 for r in paired.values()
        ):
            raise ValueError("incomplete paired diagnostic grid")
        differences = [
            float(np.mean(r["stochastic"]) - np.mean(r["deterministic"])) for r in paired.values()
        ]
        effects.append(
            {
                "variant": variant,
                "deterministic_auc": float(
                    np.mean([np.mean(r["deterministic"]) for r in paired.values()])
                ),
                "stochastic_auc": float(
                    np.mean([np.mean(r["stochastic"]) for r in paired.values()])
                ),
                "delta": float(np.mean(differences)),
                "paired_world_ci95": interval(differences),
            }
        )
    rewards = []
    for row in selected["selected"]:
        audited = [reward_audit(e["records"]) for e in row["episodes"]]
        rewards.append(
            {
                "variant": row["variant"],
                "method": f"PPO seed {row['seed']}",
                "auc": row["metrics"]["auc"],
                **{k: float(np.mean([a[k] for a in audited])) for k in audited[0]},
            }
        )
    frozen = load_config("configs/policy/selected.yaml", SelectedBaselines)
    baseline_root = Path("data/baselines") / frozen.source_run_id
    provenance = read(baseline_root / "selection.meta.json")
    paths["baseline_provenance"] = baseline_root / "selection.meta.json"
    baseline_groups = defaultdict(list)
    for filename in read(baseline_root / "complete.json")["files"]:
        path = baseline_root / filename
        row = read(path)
        key = row["key"]
        if key["variant"] not in ("V0", "V3") or key["method"] not in ("B3", "B5", "random"):
            continue
        if key["parameters"] != frozen.settings[row["family"]][key["variant"]][key["method"]]:
            continue
        if sha256_file(path) != provenance["inputs"][path.stem]["sha256"]:
            raise ValueError("baseline trajectory changed")
        baseline_groups[key["variant"], key["method"]].append(
            {"auc": row["metrics"]["auc"], **reward_audit(row["records"])}
        )
    for (variant, method), rows in baseline_groups.items():
        if len(rows) != 40:
            raise ValueError("baseline reference incomplete")
        rewards.append(
            {
                "variant": variant,
                "method": method,
                **{k: float(np.mean([r[k] for r in rows])) for k in rows[0]},
            }
        )
    probe_rows = []
    for path in sorted(probe.glob("eval-*.json")):
        rows = read(path)
        probe_rows.append(
            {
                "steps": rows[0]["steps"],
                "deterministic_auc": rows[0]["auc"],
                "stochastic_auc": float(np.mean([r["auc"] for r in rows[1:]])),
                "deterministic_reward": rows[0]["reward"],
            }
        )
        paths[path.stem] = path
    baseline_probe = read(probe / "baseline.json")
    paths["probe_baseline"] = probe / "baseline.json"
    energy = []
    for level, cost in (("L1", 4.5), ("L5", 9.5)):
        count = int(498 // cost)
        remaining = 498 - count * cost
        discounted = -cost / 500 * sum(0.99 ** np.arange(count)) - remaining / 500 * 0.99**count
        energy.append(
            {
                "level": level,
                "move_cost_s": cost,
                "undiscounted_return": -0.996,
                "discounted_return": float(discounted),
            }
        )
    summary = {
        "diagnostic_run_id": root.name,
        "probe_run_id": probe.name,
        "episodes": len(episodes),
        "sampling_effects": effects,
        "policies": diagnostics["policies"],
        "input_sensitivity": diagnostics["interventions"],
        "reward_audit": rewards,
        "constant_empty_move_energy": energy,
        "controlled_learning": probe_rows,
        "controlled_B5_auc": baseline_probe["auc"],
    }
    output = Path("docs/ppo-diagnosis.json")
    write_json(output, summary)
    lines = [
        "# M8 failure diagnosis",
        "",
        "The initial configuration has multiple demonstrated failure mechanisms. The reward "
        "ranks weak V0 search above strong search; deterministic action selection reduces "
        "coverage; per-decision discounting favors splitting energy expenditure into more actions. "
        "These findings do not fully explain the remaining V3 gap "
        "or establish an architecture defect.",
        "",
        "All six checkpoints and original results are preserved. Evidence: 240 exact "
        "deterministic replays, 720 sampled-action episodes, 1,783 input probes, and a separate "
        "32,768-step four-world learning check. Only B worlds were used.",
        "",
        "## 1. Action selection causes a measurable evaluation loss",
        "",
        "| Sensor | Argmax AUC | Sampled AUC | Paired increase [95% world bootstrap CI] |",
        "|---|---:|---:|---|",
    ]
    for row in effects:
        lo, hi = row["paired_world_ci95"]
        lines.append(
            f"| {row['variant']} | {row['deterministic_auc']:.3f} | {row['stochastic_auc']:.3f} "
            f"| {row['delta']:+.3f} [{lo:+.3f}, {hi:+.3f}] |"
        )
    lines += [
        "",
        "Each sampled estimate averages three action seeds per world and all three training "
        "seeds. The interval resamples the 40 paired worlds after averaging training/action "
        "seeds; it does not capture all training uncertainty or selection bias. "
        "V3 sampled AUC still trails frozen random (0.659) and B5 (0.888). Sampling "
        "improves coverage but does not rescue the learned policy. This does not replace "
        "the frozen deterministic M8 result.",
        "",
        "## 2. The reward conflicts with the selection metric",
        "",
        "| Sensor | Policy | AUC | Discounted action return "
        "| Same trajectory, false penalty removed |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rewards:
        lines.append(
            f"| {row['variant']} | {row['method']} | {row['auc']:.3f} "
            f"| {row['discounted_return']:.3f} | {row['without_false_penalty']:.3f} |"
        )
    lines += [
        "",
        "Returns exclude reset rewards because they precede the first policy action. "
        "The same paired launch makes those initial observations identical across normal "
        "policies. All three V0 PPO seeds earn better discounted returns than B3/B5, "
        "despite finding far fewer targets. Removing only the false-declaration term "
        "from saved trajectories reverses that ranking. This isolates the penalty's "
        "contribution to the objective mismatch; it is not a retraining result or a "
        "recommendation to ignore false alarms.",
        "",
        "V3 also ranks B3 above B5 by training return while AUC prefers B5. However, V3 "
        "PPO remains below both on reward, so objective mismatch alone cannot explain "
        "its optimization failure. The main protocol deliberately measures sensing "
        "quality, but its reward was never aligned with its AUC selection objective.",
        "",
        "## 3. Per-action discounting introduces an altitude bias",
        "",
        "For a constructed no-find/no-false-alarm path, spending the remaining 498 s "
        "of battery on repeated L1 moves costs 4.5 s per action; L5 moves cost 9.5 s. "
        "Both undiscounted returns are -0.996. With gamma=0.99 **per action**, the "
        "discounted energy returns become -0.604 (L1) and -0.778 (L5). Thus the "
        "objective prefers spending the same battery in smaller increments across more decisions. "
        "This analytical bias is real; its share of the learned altitude collapse "
        "has not been isolated by retraining.",
        "",
        "## 4. Input use is weak in V0, present in V3",
        "",
        "| Sensor | Input intervention | Mean action-distribution TV | Argmax changes |",
        "|---|---|---:|---:|",
    ]
    for row in diagnostics["interventions"]:
        lines.append(
            f"| {row['variant']} | {row['intervention']} | {row['mean_tv']:.4f} "
            f"| {100 * row['argmax_change_fraction']:.1f}% |"
        )
    lines += [
        "",
        "V0 barely responds to belief/prior rearrangement. V3 changes its argmax "
        "on about 14% of rotated-belief probes and 25% of prior/coverage probes. "
        "Therefore, saying V3 ignores its map entirely would be incorrect. The "
        "single-pixel drone marker has little effect, but the footprint also encodes "
        "position; that observation alone does not prove a localization defect. "
        "Interventions can be off-distribution and are descriptive sensitivity tests.",
        "",
        "## 5. The training stack can learn a small fixed task",
        "",
        "| Transitions | Argmax AUC | Sampled AUC |",
        "|---:|---:|---:|",
    ]
    for row in probe_rows:
        lines.append(
            f"| {row['steps']} | {row['deterministic_auc']:.3f} | {row['stochastic_auc']:.3f} |"
        )
    lines += [
        "",
        f"B5 AUC on these same four fixed training episodes: {baseline_probe['auc']:.3f}. "
        "The unchanged architecture/reward/optimizer learns from 0.000 to 0.579 "
        "argmax AUC. This rules out a complete inability to learn through the adapter "
        "and supports a difficulty with the broader task. It does not prove convergence, "
        "generalization, or that every integration detail is correct.",
        "",
        "## Decision before another full run",
        "",
        "First settle whether the primary objective is early target recovery (AUC) or "
        "recovery minus false-alarm/energy cost. The current experiment optimizes one "
        "and selects on the other. Then run a controlled revision with an aligned "
        "objective and duration-consistent discounting, while evaluating both action "
        "rules under a newly recorded protocol. Potential-based shaping cannot by "
        "itself repair the underlying objective mismatch. Do not change rewards and "
        "architecture together or simply extend the same failed configuration.",
        "",
        "The residual V3 gap after sampling remains partly unresolved. Credit assignment, "
        "exploration, representation and budget remain hypotheses; no experiment here "
        "uniquely identifies one of them. M8 remains incomplete.",
        "",
        "## Reproduction",
        "",
        "`uv run python scripts/report_ppo_diagnosis.py` regenerates this report and "
        "the numeric figure from completed outputs. It does not restart training. "
        "Original diagnostic commands/protocol are in `docs/ppo-diagnosis-protocol.md`.",
        "",
        "![Diagnostic comparisons](figures/m8_diagnosis.png)",
        "",
    ]
    table = Path("docs/ppo-diagnosis.md")
    table.write_text("\n".join(lines), encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    x = np.arange(2)
    axes[0].bar(x - 0.16, [r["deterministic_auc"] for r in effects], 0.32, label="Argmax")
    axes[0].bar(x + 0.16, [r["stochastic_auc"] for r in effects], 0.32, label="Sampled")
    axes[0].set(
        xticks=x,
        xticklabels=["V0", "V3"],
        ylim=(0, 1),
        ylabel="Validation AUC",
        title="Same frozen checkpoints",
    )
    axes[0].legend()
    chosen = [r for r in rewards if r["variant"] == "V0" and r["method"] in ("PPO seed 17", "B5")]
    axes[1].bar(x - 0.16, [r["discounted_return"] for r in chosen], 0.32, label="Original reward")
    axes[1].bar(
        x + 0.16, [r["without_false_penalty"] for r in chosen], 0.32, label="False penalty removed"
    )
    axes[1].set(
        xticks=x,
        xticklabels=[r["method"] for r in chosen],
        ylabel="Discounted action return",
        title="V0: trajectory-only counterfactual",
    )
    axes[1].axhline(0, color="grey", linewidth=0.7)
    axes[1].legend(fontsize=8)
    for key, label in (("deterministic_auc", "Argmax"), ("stochastic_auc", "Sampled")):
        axes[2].plot(
            [r["steps"] for r in probe_rows], [r[key] for r in probe_rows], marker=".", label=label
        )
    axes[2].axhline(baseline_probe["auc"], linestyle="--", color="grey", label="B5")
    axes[2].set(
        xlabel="Training transitions",
        ylabel="Four-world training AUC",
        ylim=(0, 1),
        title="Controlled learning check",
    )
    axes[2].legend()
    figure = Path("docs/figures/m8_diagnosis.png")
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    write_metadata(
        root / "report.meta.json",
        stage="report-ppo-diagnosis",
        config=load_config("configs/policy/ppo.yaml", PPOConfig),
        inputs=paths,
        outputs={"report": output, "table": table, "figure": figure},
    )
    print("Reported 960 diagnostic episodes and the completed four-world learning probe.")


if __name__ == "__main__":
    main()
