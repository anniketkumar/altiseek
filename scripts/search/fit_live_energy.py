"""D63: verify B live records, estimate additive dwell and check the coverage gate.

Run from the repository root, with --native-root pointing at the saved WSL M10
directory. Reads only named B sources; never launches a flight or detector.
"""

import argparse
import itertools
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from altiseek.calibration.model import digest
from altiseek.config import load_config
from altiseek.deploy.energy_fit import (
    SOURCES,
    b_worlds,
    calibration_intervals,
    calibration_records,
    describe,
    finite,
    median_fit,
    residual,
    strata,
    verified_search,
)
from altiseek.env.energy import EnergyModel
from altiseek.geometry.design_config import LevelDesign
from altiseek.geometry.level_design import coverage_energy_s
from altiseek.geometry.levels import LevelTable
from altiseek.registry.run import sha256_file


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/m12/energy"))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    energy = load_config(repo / "configs/env/energy.yaml", EnergyModel)
    if energy.model_dump() != {
        "horizontal_speed_mps": 10.0,
        "vertical_speed_mps": 3.0,
        "dwell_s": 2.0,
        "climb_multiplier": 1.5,
        "battery_s": 500.0,
    }:
        raise ValueError("Historical energy values changed")
    worlds = {w.world_id for w in b_worlds(repo, "rl-val")}
    flights, primary_rows, values = [], [], []
    for source in SOURCES:
        if source.world not in worlds:
            raise ValueError("Primary source is not an original B validation world")
        report, provenance = verified_search(args.native_root, source)
        views = report["views"]
        if len(views) < 2:
            raise ValueError("Primary flight has no complete transition")
        for view in views:
            if (
                not 0 <= finite(view["pose_delta_s"]) <= 0.04
                or not 0 <= finite(view["goal_error_m"]) <= 1
            ):
                raise ValueError("Search view violates capture limits")
        rows = [
            {
                "flight": source.flight,
                "transition": i,
                **residual(previous, current, energy, measured=True),
            }
            for i, (previous, current) in enumerate(itertools.pairwise(views), start=1)
        ]
        values.append([r["overhead_s"] for r in rows])
        primary_rows.extend(rows)
        flights.append(
            {
                **provenance,
                "views": len(views),
                **describe(rows),
                "initial_elapsed_s": finite(views[0]["measured"]["elapsed_s"]),
                "initial_overhead_s": finite(views[0]["measured"]["elapsed_s"]) - energy.dwell_s,
                "median_energy_residual_s": float(
                    np.median([r["energy_residual_s"] for r in rows])
                ),
            }
        )
    estimate = median_fit(values)
    dwell = energy.dwell_s + estimate["overhead_s"]
    if estimate["overhead_s"] <= 0 or dwell <= 0:
        raise ValueError("Nonpositive correction/dwell; owner review required")
    table = load_config(repo / "configs/levels/default.yaml", LevelTable)
    design = load_config(repo / "configs/levels/design.yaml", LevelDesign)
    if design.coarse_share != 0.25:
        raise ValueError("Historical coarse coverage criterion changed")
    coverage = []
    for label, overhead in (
        ("original", 0.0),
        ("estimate", estimate["overhead_s"]),
        ("ci_low", estimate["ci95_s"][0]),
        ("ci_high", estimate["ci95_s"][1]),
    ):
        candidate = energy.model_copy(update={"dwell_s": energy.dwell_s + overhead})
        costs = [
            coverage_energy_s(table, level, design.camera.hfov_deg, candidate)
            for level in table.levels
        ]
        coverage.append(
            {
                "setting": label,
                "dwell_s": candidate.dwell_s,
                "costs_s": costs,
                "l1_exceeds_battery": costs[0] > energy.battery_s,
                "l5_budget_fraction": costs[-1] / energy.battery_s,
                "l5_within_coarse_share": costs[-1] <= design.coarse_share * energy.battery_s,
            }
        )
    keys = [
        (li, ix, iy) for li, level in enumerate(table.levels) for ix, iy in table.lattice(level)
    ]
    records, calibration = calibration_records(args.native_root, repo, keys)
    diagnostic_rows = calibration_intervals(records, energy)
    by_world: defaultdict[str, list] = defaultdict(list)
    by_level: defaultdict[int, list] = defaultdict(list)
    for row in diagnostic_rows:
        by_world[row["world_id"]].append(row)
        by_level[row["level"]].append(row)
    calibration.update(
        {
            **describe(diagnostic_rows),
            "strata": strata(diagnostic_rows),
            "per_world": [
                {"world_id": w, **describe(rows)} for w, rows in sorted(by_world.items())
            ],
            "per_level": [{"level": li, **describe(rows)} for li, rows in sorted(by_level.items())],
            "excluded_first_captures": len(records) - len(diagnostic_rows),
            "clock": "capture-to-capture; not pooled into the primary estimate",
        }
    )
    sources = [
        "docs/m12-ppo-revisit/energy-protocol.md",
        "src/altiseek/deploy/energy_fit.py",
        "scripts/search/fit_live_energy.py",
        "src/altiseek/env/energy.py",
        "src/altiseek/geometry/level_design.py",
        "configs/env/energy.yaml",
        "configs/levels/default.yaml",
        "configs/levels/design.yaml",
    ]
    gate = coverage[1]["l1_exceeds_battery"] and coverage[1]["l5_within_coarse_share"]
    summary = {
        "decision": "D63",
        "protocol": "docs/m12-ppo-revisit/energy-protocol.md",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "source_sha256": {name: sha256_file(repo / name) for name in sources},
        "primary": {
            **estimate,
            "candidate_dwell_s": dwell,
            "flights": flights,
            "strata": strata(primary_rows),
            "transitions": len(primary_rows),
        },
        "coverage": coverage,
        "coverage_gate_passed": gate,
        "calibration": calibration,
        "limitations": [
            "Five search flights, two B worlds; bootstrap is conditional on these records.",
            "Calibration intervals omit current inference and policy decisions.",
            "Calibration level and stride length are confounded; no distance term is fitted.",
            "A shared dwell approximates both initial observation and additional hover energy.",
        ],
        "baseline_retune": "not started",
        "owner_stop": "coverage gate failed" if not gate else "after baseline retune",
    }
    summary["analysis_id"] = digest(summary)
    write_json(
        args.out / "transitions.json", {"primary": primary_rows, "calibration": diagnostic_rows}
    )
    write_json(args.out / "summary.json", summary)
    print(
        json.dumps(
            {
                "analysis_id": summary["analysis_id"],
                **estimate,
                "candidate_dwell_s": dwell,
                "coverage": coverage,
                "coverage_gate_passed": gate,
                "out": str(args.out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
