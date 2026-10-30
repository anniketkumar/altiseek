"""Summarise every D46 training seed and validation curve without reselection."""

from pathlib import Path

import numpy as np

from altiseek.cache.jobs import write_json
from altiseek.eval.extension import ExtensionProtocol
from altiseek.eval.reporting import interval
from altiseek.eval.runner import read
from altiseek.registry.run import sha256_file


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = read(Path("configs/eval/artifacts_v2.json"))
    protocol = ExtensionProtocol()
    output = Path("docs/m9-training-results.json")
    rows = []
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), layout="constrained", sharey=True)
    for ax, (case, checkpoints) in zip(axes.flat, manifest["ablations"].items(), strict=True):
        for checkpoint in checkpoints:
            root = Path(checkpoint["checkpoint"]).parent.parent
            selected_path = root / f"eval-{checkpoint['selected_step']:09d}.json"
            if sha256_file(selected_path) != checkpoint["validation_sha256"]:
                raise ValueError("selected B report checksum changed")
            selected = read(selected_path)
            zero_path = root / "zero-belief.json"
            zero = read(zero_path)
            if not zero["zero_belief"] or zero["steps"] != checkpoint["selected_step"]:
                raise ValueError("zero-belief diagnostic differs from selected checkpoint")
            expected = {(e["world_id"], e["repeat"]) for e in selected["episodes"]}
            if {(e["world_id"], e["repeat"]) for e in zero["episodes"]} != expected:
                raise ValueError("unpaired belief-removal diagnostic")
            per_world = {}
            for label, report in (("full", selected), ("zero", zero)):
                grouped = {}
                for episode in report["episodes"]:
                    grouped.setdefault(episode["world_id"], []).append(episode["metrics"]["auc"])
                per_world[label] = {w: float(np.mean(values)) for w, values in grouped.items()}
            difference = np.array(
                [per_world["full"][w] - per_world["zero"][w] for w in sorted(per_world["full"])]
            )
            curve = [read(root / f"eval-{s:09d}.json") for s in range(0, 131073, 16384)]
            timings = [
                read(root / f"step-{s:09d}/timing.json") for s in range(16384, 131073, 16384)
            ]
            seconds = sum(t["seconds"] for t in timings)
            rows.append(
                {
                    "case": case,
                    "seed": checkpoint["seed"],
                    "run_id": root.name,
                    "selected_step": checkpoint["selected_step"],
                    "selected_auc": selected["mean_auc"],
                    "final_auc": curve[-1]["mean_auc"],
                    "zero_auc": zero["mean_auc"],
                    "full_minus_zero": {
                        "mean": float(difference.mean()),
                        "ci95": interval(difference, protocol),
                    },
                    "training_seconds": seconds,
                    "steps_per_second": 131072 / seconds,
                    "training_worlds": checkpoint["training_worlds"],
                    "validation_worlds": checkpoint["validation_worlds"],
                    "curve": [{"steps": r["steps"], "auc": r["mean_auc"]} for r in curve],
                    "zero_report_sha256": sha256_file(zero_path),
                }
            )
            ax.plot(
                [r["steps"] / 1000 for r in curve],
                [r["mean_auc"] for r in curve],
                label=str(checkpoint["seed"]),
            )
            ax.scatter([checkpoint["selected_step"] / 1000], [selected["mean_auc"]], s=22)
        ax.set(title=case, xlabel="Training steps (thousands)", ylim=(0, 1))
        ax.legend(title="Seed", fontsize=8)
    fig.suptitle("M9 B validation: selected points marked; appearance uses a 13-world subset")
    figure = Path("docs/figures/m9_training.png")
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    write_json(
        output, {"rows": rows, "manifest_sha256": sha256_file("configs/eval/artifacts_v2.json")}
    )
    lines = [
        "# M9 training review",
        "",
        "All 24 planned runs completed their 131,072-step budget.",
        "Checkpoint selection uses B only, raw mean AUC and earliest ties. These are selection-set",
        "scores, not held-out performance. Appearance cases use only 13 validation worlds;",
        "their scores are not directly comparable with the 40-world scores of other cases.",
        "",
        "| Case | Seed | Selected step | Selected AUC | Final AUC | Zero-p AUC | Steps/s |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['case']} | {r['seed']} | {r['selected_step']} | "
            f"{r['selected_auc']:.3f} | {r['final_auc']:.3f} | "
            f"{r['zero_auc']:.3f} | {r['steps_per_second']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "Several curves fluctuate and final performance can fall below "
            "the selected checkpoint.",
            "Completion of the fixed budget does not establish stable convergence. "
            "Report every seed,",
            "retain B-selected checkpoints, and do not extend or reselect after viewing T results.",
            "",
            "Belief removal is a diagnostic: a small or negative paired effect does not establish",
            "leakage or prove the policy ignores all spatial input. Prior, coverage and verified",
            "feedback remain available (D42). The input-access and adapter checks passed before T.",
            "",
            "The re-look configuration assumes independent evidence across "
            "distinct variants; higher",
            "B AUC is not evidence that this assumption holds in real sensing. "
            "Learned DECLARE also",
            "changes action cost and verification behavior as predeclared, "
            "so it is a task ablation.",
            "",
            "Reproduce: `uv run python scripts/report_m9_training.py`.",
        ]
    )
    Path("docs/m9-training-results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote all {len(rows)} training seeds and curves to {output}", flush=True)


if __name__ == "__main__":
    main()
