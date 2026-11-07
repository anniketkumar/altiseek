"""Publish the saved D63 timing result and candidate dwell; never run a baseline."""

import argparse
import json
from pathlib import Path

from altiseek.calibration.model import digest
from altiseek.registry.run import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    saved = json.loads(args.summary.read_text(encoding="utf-8"))
    if digest({k: v for k, v in saved.items() if k != "analysis_id"}) != saved["analysis_id"]:
        raise ValueError("Analysis identity mismatch")
    if saved["decision"] != "D63" or saved["baseline_retune"] != "not started":
        raise ValueError("Expected the original D63 energy analysis")
    for name, checksum in saved["source_sha256"].items():
        if sha256_file(name) != checksum:
            raise ValueError(f"Analysis input changed: {name}")
    primary, calibration = saved["primary"], saved["calibration"]
    coverage = {row["setting"]: row for row in saved["coverage"]}
    passed = saved["coverage_gate_passed"]
    lo, hi = primary["ci95_s"]
    point = coverage["estimate"]
    minimum_battery = point["costs_s"][-1] / 0.25
    lines = [
        "# M12 Phase 1: B-fitted energy correction",
        "",
        f"D63 analysis `{saved['analysis_id']}`, computed 2026-11-07 with code commit "
        f"`{saved['git_commit']}` under the [predeclared protocol](energy-protocol.md).",
        "",
        f"**Extra elapsed time: {primary['overhead_s']:.2f} s per view "
        f"(95% whole-flight bootstrap interval {lo:.2f} to {hi:.2f} s).** "
        f"The candidate dwell is **{primary['candidate_dwell_s']:.2f} s**, "
        "replacing 2 s only in the new candidate config. Speeds, climb multiplier "
        "and the 500 s battery are unchanged.",
        "",
        "**Coverage gate: " + ("passed." if passed else "failed; owner decision required.") + "** "
        f"Full L5 coverage costs {point['costs_s'][-1]:.2f} s "
        f"({100 * point['l5_budget_fraction']:.3f}% of the battery), "
        "against the existing 25% / 125 s limit. "
        "Baseline retuning has not started. No B5 AUC change has been measured; "
        "`selected_live.yaml` has not been created. Phase 1 is incomplete.",
        "",
        "## Primary estimate",
        "",
        f"{primary['transitions']} complete transitions from five measured-search flights "
        "on two B rl-val worlds. The point estimate is the median of flight medians, "
        "so the 15-transition PPO flight does not outweigh the shorter flights. "
        "10,000 bootstrap resamples use seed 20261106. The interval is conditional "
        "on these few records, not a deployment-wide uncertainty guarantee.",
        "",
        "| Flight | Policy | Transitions | Median overhead (s) | Initial measured time (s) "
        "| Median energy residual (hover-s) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in primary["flights"]:
        policy = row["method"] + (f" {row['training_seed']}" if row["training_seed"] else "")
        lines.append(
            f"| {row['flight']} | {policy} | {row['transitions']} | "
            f"{row['median_overhead_s']:.3f} | {row['initial_elapsed_s']:.3f} | "
            f"{row['median_energy_residual_s']:.3f} |"
        )
    lines += [
        "",
        "Intervals use post-inference `measured.elapsed_s`, subtracting the unchanged "
        "nominal move plus 2 s dwell for the same commanded ENU displacement. "
        "Initial observations, takeoff, landing and incomplete moves are excluded. "
        "The initial measured times are all below the candidate shared dwell, so "
        "applying it at reset is a conservative timing approximation on these five cases. "
        "Energy residuals are descriptive only; the estimator fits elapsed time, "
        "not measured climb-energy excess.",
        "",
        "## Coverage at the unchanged battery",
        "",
        "Uses the existing full-lattice lawnmower route, starting at each level's own "
        "altitude. The existing `coarse_share: 0.25` in `configs/levels/design.yaml` "
        "resolves the handoff's qualitative word 'cheap'. No new threshold was selected.",
        "",
        "| Setting | Dwell (s) | L1 coverage (s) | L5 coverage (s) | L5 share | L5 ≤125 s |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for label, row in coverage.items():
        lines.append(
            f"| {label} | {row['dwell_s']:.3f} | {row['costs_s'][0]:.3f} | "
            f"{row['costs_s'][-1]:.3f} | {100 * row['l5_budget_fraction']:.3f}% | "
            f"{'yes' if row['l5_within_coarse_share'] else 'no'} |"
        )
    lines += [
        "",
        "L1 remains infeasible at every reported setting. The L5 interval crosses "
        "the constraint, but the predeclared point-estimate gate fails. The fit was "
        "not lowered to meet the constraint. Keeping the 25% rule would require "
        f"at least {minimum_battery:.2f} s battery at the point estimate; alternatively "
        "the owner could retain 500 s and explicitly amend the coarse-coverage rule. "
        "Neither change has been made or evaluated.",
        "",
        "## Separate calibration diagnostics",
        "",
        f"Verified {calibration['views']:,} accepted views on {calibration['worlds']} "
        f"sensor-fit B worlds in acquisition `{calibration['acquisition_id']}`. "
        f"There are {calibration['transitions']:,} within-flight capture intervals; "
        f"{calibration['excluded_first_captures']} first captures are excluded. "
        "Accepted views from subsequently failed flights remain included. No interval "
        "bridges flights. Every view's NPZ checksum, key, contract, B identity and "
        "capture limits were checked; no image was decoded or inference rerun.",
        "",
        "Capture intervals contain previous-frame processing and saving, but omit "
        "current-frame inference and search-policy decisions. They are not pooled "
        "into the primary estimate. These summaries use all transitions arriving "
        "at a level, including climbs:",
        "",
        "| Destination | Transitions | Median capture-interval overhead (s) |",
        "|---|---:|---:|",
    ]
    for row in calibration["per_level"]:
        lines.append(
            f"| L{row['level'] + 1} | {row['transitions']:,} | {row['median_overhead_s']:.3f} |"
        )
    lines += [
        "",
        "| Destination | Motion | Horizontal distance (m) | Intervals | Median overhead (s) |",
        "|---|---|---:|---:|---:|",
    ]
    for row in calibration["strata"]:
        lines.append(
            f"| L{row['level'] + 1} | {row['motion']} | {row['horizontal_m']:.3f} | "
            f"{row['transitions']:,} | {row['median_overhead_s']:.3f} |"
        )
    lines += [
        "",
        "Regular same-level strides of 25 to 75 m have median residuals of 6 to 6.5 s, "
        "without a monotonic increase with distance. Level and stride length are "
        "confounded, and the rare longer moves have little replication. These data "
        "do not isolate a distance-dependent settling term. Cross-level residuals "
        "are much smaller, sometimes negative; simultaneous horizontal/vertical "
        "motion versus the nominal model's summed travel times is a possible "
        "contributor, not an isolated causal finding. No distance term or schema "
        "field was added. A constant dwell remains an approximation, particularly "
        "for level changes; richer modeling needs a new declared protocol.",
        "",
        "## Reproduction and preservation",
        "",
        "From the repository root in WSL (the first command only reads saved B sources):",
        "",
        "```sh",
        "/home/aniket/.venvs/altiseek/bin/python scripts/search/fit_live_energy.py "
        "--native-root /home/aniket/altiseek-m10",
        "uv run python scripts/search/report_live_energy.py data/m12/energy/summary.json",
        "```",
        "",
        "The JSON report carries source checksums, the analysis ID, per-world numeric "
        "diagnostics and flight/controller/failure provenance. Raw timing rows stay "
        "in ignored `data/m12/energy/`. No schema or historical implementation changed; "
        "no frozen config, world-set entry, registry entry or historical result changed. "
        "The candidate config is not referenced by any existing run config. "
        "Historical artifacts retain their verification context at `results-final`.",
        "",
        "No T source, test world, test seed or M10 transfer-results file selected a "
        "parameter. No completed experiment was rerun. The coverage stop is required "
        "by handoff Phase 1 step 4 and the approved D63 protocol; owner authorization "
        "of a revised constraint or battery is needed before the baseline retune.",
        "",
    ]
    destination = Path("docs/m12-ppo-revisit")
    (destination / "energy-correction.json").write_text(
        json.dumps(saved, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (destination / "energy-correction.md").write_text("\n".join(lines), encoding="utf-8")
    Path("configs/env/energy_live.yaml").write_text(
        "# D63 B-only fit; candidate only: the L5 coverage gate failed.\n"
        "# Owner review is required before using this config for the baseline retune.\n"
        "# Source: docs/m12-ppo-revisit/energy-correction.json\n"
        "_base_: energy.yaml\n"
        f"dwell_s: {primary['candidate_dwell_s']!r}\n",
        encoding="utf-8",
    )
    print(f"Published {saved['analysis_id']}; candidate only; coverage gate passed: {passed}")


if __name__ == "__main__":
    main()
