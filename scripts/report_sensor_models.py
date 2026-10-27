"""Regenerate M6 numeric tables and figures from saved fit/evaluation reports."""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from altiseek.cache.config import CacheConfig
from altiseek.cache.jobs import write_json
from altiseek.calibration.model import SensorModel
from altiseek.config import load_config
from altiseek.env.energy import EnergyModel
from altiseek.geometry.level_design import coverage_energy_s


def rate_text(rate: dict[str, Any], scale: float = 1) -> str:
    if rate["raw"] is None:
        return "not estimable"
    lo, hi = rate["ci95"]
    return f"{rate['raw'] * scale:.3f} [{lo * scale:.3f}, {hi * scale:.3f}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--real", type=Path)
    args = parser.parse_args()
    cache = load_config("configs/cache/default.yaml", CacheConfig)
    model = SensorModel.load(args.model, detector_id=cache.detector_id)
    fit = json.loads(args.model.with_suffix(".fit.json").read_text(encoding="utf-8"))
    test = json.loads(args.model.with_suffix(".test.json").read_text(encoding="utf-8"))
    reports = {"Sensor-fit B (in-sample)": fit, "Test worlds (held-out)": test}
    if args.real:
        reports["Real T (held-out)"] = json.loads(args.real.read_text(encoding="utf-8"))
    for report in list(reports.values())[1:]:
        if report["sensor_model_id"] != model.sensor_model_id:
            raise ValueError("report/model mismatch")
    energy = load_config("configs/env/energy.yaml", EnergyModel)
    costs = [
        coverage_energy_s(cache.render.levels, lvl, cache.render.camera.hfov_deg, energy)
        for lvl in cache.render.levels.levels
    ]
    combined = {
        "sensor_model_id": model.sensor_model_id,
        "thresholds": {"tau": model.fit_config.tau, "tau_min": model.fit_config.tau_min},
        "temperatures": [level.temperature for level in model.levels],
        "coverage_energy_s": costs,
        "battery_s": energy.battery_s,
        "reports": reports,
    }
    write_json(Path("docs/sensor-results.json"), combined)
    lines = [
        "# M6 sensor results",
        "",
        f"Model `{model.sensor_model_id}`; detector `{model.detector_id}`; "
        f"fit world set `{model.fit_world_set_id}`.",
        "",
        "Reproduce with `scripts/report_sensor_models.py --model <model.json> "
        "--real <real-T/report.json>`. Protocol: [sensor-protocol.md](sensor-protocol.md).",
        "",
        "Rates below are raw estimates with 95% cluster-bootstrap intervals. "
        "Runtime likelihoods use the fixed smoothing documented in the protocol. "
        "False alarms are per 10,000 observed empty cells, including distractor cells.",
        "",
    ]
    for title, report in reports.items():
        lines += [
            f"## {title}",
            "",
            "| Level | Recall at tau [95% CI] | False alarms / 10k [95% CI] "
            "| Detected cells at tau_min | ECE raw → calibrated | BCE raw → calibrated |",
            "|---|---|---|---:|---|---|",
        ]
        for level, stats, calibration in zip(
            model.levels, report["levels"], report["calibration"], strict=True
        ):
            before, after = calibration["before"], calibration["after"]
            lines.append(
                f"| {level.name} | {rate_text(stats['binary']['recall'])} "
                f"| {rate_text(stats['binary']['false_alarm'], 10000)} "
                f"| {before['count']} | {before['ece']:.3f} → {after['ece']:.3f} "
                f"| {before['bce']:.3f} → {after['bce']:.3f} |"
            )
        lines.append("")
    lines += [
        "## Re-look and level-design check (B only)",
        "",
        "| Level | Temperature | P(hit second \u007c missed first), tau [95% CI] "
        "| Same, tau_min [95% CI] | Full lattice sweep (s) |",
        "|---|---:|---|---|---:|",
    ]
    for level, stats, cost in zip(model.levels, fit["levels"], costs, strict=True):
        lines.append(
            f"| {level.name} | {level.temperature:.3f} "
            f"| {rate_text(stats['binary']['conditional_recall'])} "
            f"| {rate_text(stats['soft']['conditional_recall'])} | {cost:.1f} |"
        )
    lines += [
        "",
        "Each second-look measurement pools all ordered distinct variant pairs at the "
        "same viewpoint; intervals resample worlds. It does not prove independence across "
        "overlapping views. The default remains one informative observation per cell/level "
        "(D35).",
        "",
        "The owner retained the five levels and 500 s battery (D35). L1's "
        "cell recall is slightly below the original 0.80 target; its interval includes 0.80. "
        "L5 is below the revised 0.65 target. Fine coverage exceeds the battery; coarse "
        "coverage costs less than one quarter. Rates are cell events, not D24's 1 m "
        "object-level recall.",
        "",
        "## Calibration limitations",
        "",
        "Temperature scaling is fit by BCE, so ECE need not decrease. A positive "
        "temperature cannot move a raw score across 0.5 or change score ordering. "
        "The remaining calibration error is reported without refitting on T. B worlds "
        "include distractors; test worlds contain none, which changes the prevalence of "
        "true cells among detections. Real T consists of exercise scenes and has different "
        "context and density again. Detection-score calibration is not a transferable "
        "occupancy posterior; V3 uses conditional likelihoods and a separate prior.",
        "",
        "There are 40 fit worlds and 80 test worlds, ten per family. World bootstrap "
        "intervals do not capture uncertainty from having only one source day per split "
        "or repeated imagery across worlds; frame intervals likewise do not capture "
        "between-video/day variation. Simulation only.",
        "",
        "![Rates and re-look](figures/m6_rates.png)",
        "",
        "![Reliability by domain and level](figures/m6_reliability.png)",
        "",
    ]
    Path("docs/sensor-results.md").write_text(
        "\n".join(lines).replace("second | missed", "second / missed"), encoding="utf-8"
    )
    figures = Path("docs/figures")
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for title, report in reports.items():
        for ax, metric, scale in ((axes[0], "recall", 1), (axes[1], "false_alarm", 10000)):
            values = [r["binary"][metric] for r in report["levels"]]
            centre = np.array([r["raw"] for r in values]) * scale
            ci = np.array([r["ci95"] for r in values]).T * scale
            ax.errorbar(
                range(1, 6),
                centre,
                yerr=np.maximum(0, [centre - ci[0], ci[1] - centre]),
                marker="o",
                capsize=3,
                label=title,
            )
    conditional = [level["binary"]["conditional_recall"] for level in fit["levels"]]
    centre = np.array([r["raw"] for r in conditional])
    ci = np.array([r["ci95"] for r in conditional]).T
    axes[2].errorbar(
        range(1, 6), centre, yerr=[centre - ci[0], ci[1] - centre], marker="o", capsize=3
    )
    for ax, title in zip(
        axes,
        (
            "Cell recall at tau",
            "False alarms / 10,000 empty cells",
            "Second-look recall after a miss (B)",
        ),
        strict=True,
    ):
        ax.set(title=title, xlabel="Level", xticks=range(1, 6))
        ax.grid(alpha=0.2)
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=7)
    fig.savefig(figures / "m6_rates.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(
        len(reports), 5, figsize=(15, 3 * len(reports)), squeeze=False, constrained_layout=True
    )
    for row, (title, report) in enumerate(reports.items()):
        for li, calibration in enumerate(report["calibration"]):
            ax = axes[row, li]
            ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
            for key, label in (("before", "Raw"), ("after", "Calibrated")):
                bins = [b for b in calibration[key]["bins"] if b["count"]]
                ax.plot(
                    [b["confidence"] for b in bins],
                    [b["accuracy"] for b in bins],
                    marker="o",
                    markersize=3,
                    label=label,
                )
            ax.set(xlim=(0, 1), ylim=(0, 1), title=f"{title}\nL{li + 1}", xlabel="Mean score")
            if li == 0:
                ax.set_ylabel("Occupied fraction")
                ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
    fig.savefig(figures / "m6_reliability.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
