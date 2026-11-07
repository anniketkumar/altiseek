"""D65: audit saved corrected-cost B episodes and publish Phase 1 results."""

import argparse
import json
from pathlib import Path

import yaml

from altiseek.cache.build import header_for
from altiseek.cache.config import CacheConfig
from altiseek.cache.jobs import resume_metadata, write_json
from altiseek.calibration.model import SensorModel, digest
from altiseek.config import load_config
from altiseek.deploy.energy_fit import b_worlds
from altiseek.env.config import SearchConfig
from altiseek.eval.baseline_review import (
    ORIGINAL_RUN,
    original_b5,
    read_json,
    summarise,
    verify_grid,
)
from altiseek.geometry.design_config import LevelDesign
from altiseek.geometry.level_design import coverage_energy_s
from altiseek.geometry.levels import LevelTable
from altiseek.policies.config import BaselineRun, SelectedBaselines
from altiseek.registry.run import sha256_file, write_metadata


def markdown(report: dict) -> str:
    paired = report["b5_paired"]
    lines = [
        "# M12 Phase 1: corrected-cost baseline results",
        "",
        f"D65 run `{report['run_id']}`; {report['episodes']:,} CPU episodes on 40 rl-val B "
        "worlds, V0-V3, the unchanged M7 candidate grid and seeds. "
        f"Audit `{report['review_id']}`. No T data or new flight was used.",
        "",
        "**Phase 1 is complete, pending owner review.** "
        "B1-B3 were retuned and B4/B5 rerun under the fixed B-fitted costs. "
        "The sensor, 500 s battery, speeds and climb multiplier remain unchanged. "
        "PPO training and Phase 2 have not started.",
        "",
        "The [D63 fit](energy-correction.md) remains **6.42 s extra per view "
        "[3.31, 7.16]**, giving **8.42 s dwell**. Its original 25% L5 gate failed: "
        "135.78 s exceeds 125 s. The owner-authorized [D65 amendment](baseline-protocol.md) "
        "allows 30% for M12 only. The same 135.78 s costs 27.156% of 500 s, "
        "within the new 150 s allowance; L1 costs 1,318.82 s and remains infeasible. "
        "This explicit post-B amendment does not erase the original failure.",
        "",
        "## B5 paired comparison",
        "",
        "Old-cost B5 episodes are reused from saved run `8cd76bdb72f965a1`, "
        "not rerun. Each pair has the same world, sensor variant and reset seed. "
        "Differences include changes to B5's energy-aware action choices as well "
        "as timing; this is not just a rescaling of a fixed path.",
        "",
        "| Sensor | Original AUC | Corrected AUC | Paired change [95% CI] |",
        "|---|---:|---:|---:|",
    ]
    for row in paired:
        lo, hi = row["delta_auc_ci95"]
        lines.append(
            f"| {row['variant']} | {row['old_auc']:.6f} | {row['new_auc']:.6f} | "
            f"{row['delta_auc']:+.6f} [{lo:+.6f}, {hi:+.6f}] |"
        )
    main = next(row for row in paired if row["variant"] == "V3")
    lo, hi = main["delta_auc_ci95"]
    interpretation = (
        "The paired interval lies below zero."
        if hi < 0
        else "The paired interval lies above zero."
        if lo > 0
        else "The paired interval includes zero."
    )
    lines += [
        "",
        f"The V3 reference changes by {main['delta_auc']:+.6f} AUC. {interpretation} "
        "This measures B5 under a more costly simulator; it does not measure "
        "PPO improvement or live transfer. No setting was changed in response to the sign.",
        "",
        "## All selected baselines",
        "",
        "| Sensor | Method | AUC [95% CI] | Found fraction | False declarations | Normalised AUC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in report["aggregate"]:
        lo, hi = row["auc_ci95"]
        norm = "undefined" if row["normalised_auc"] is None else f"{row['normalised_auc']:.4f}"
        lines.append(
            f"| {row['variant']} | {row['method']} | {row['auc']:.4f} [{lo:.4f}, {hi:.4f}] | "
            f"{row['found_fraction']:.4f} | {row['false_declarations']:.3f} | {norm} |"
        )
    minimum = min(row["oracle_minus_random_auc"] for row in report["oracle_gaps"])
    lines += [
        "",
        "These are **B selection-set results**, not held-out performance. "
        "B1-B3 settings maximize mean family/variant AUC, breaking ties by fewer false "
        "declarations then canonical parameter JSON. B4/B5 parameters remain empty. "
        "The original `configs/policy/selected.yaml` remains unchanged; new choices "
        "are in `configs/policy/selected_live.yaml`. Per-family numbers, all B5 pairs "
        "and chosen settings are in the JSON report.",
        "",
        "Intervals use 10,000 whole-world resamples per variant, seed 20261106, "
        "after sorting world IDs. The same draws serve all methods and paired changes. "
        "Intervals do not correct baseline-selection bias or shared imagery. "
        "Normalisation uses pooled new oracle/random means; the privileged L1 oracle "
        "is not a per-episode mathematical upper bound.",
        "",
        "## Verification and reproduction",
        "",
        f"All {report['episodes']:,} expected new episode keys are present once. "
        "World/cache/sensor/config identities, complete-grid metadata, finite metrics, "
        "budget limits, valid endings and AUC reconstructed from saved found curves "
        "passed. All 160 original B5 report checksums matched their saved selection "
        "metadata, and the original raw plan retained its ID. No historical episode "
        "was re-executed.",
        "",
        f"Oracle exceeds random in every family/variant: **{report['sanity_passed']}**, "
        f"minimum family/variant gap {minimum:.6f} AUC. "
        "Existing full-coverage detector sweeps were not repeated; sensing is unchanged.",
        "",
        "Regenerate this report from the saved episodes:",
        "",
        "```sh",
        "uv run python scripts/search/report_live_baselines.py "
        f"data/m12/baselines/{report['run_id']}",
        "```",
        "",
        "The full new grid was launched detached from commit `65d724f`, using "
        "`uv run python scripts/search/run_baselines.py --search-config "
        "configs/env/search_live.yaml --out data/m12/baselines`. Its existing "
        "resume logic skips completed episodes. Raw records remain in ignored data/. "
        "Only new configs and reports are published. No config schema, frozen "
        "evaluation file, historical result, world-set entry or registry entry changed.",
        "",
        "All earlier qualifications remain: M8's qualified acceptance, historical "
        "PPO below B5, and unsupported H3b. Phase 1 completion establishes a documented "
        "B-only timing model and retuned references, not an affirmative PPO finding. "
        "The next step, only after owner review, is the B5 warm start and V3 CPU pilot.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    grid = load_config("configs/policy/baselines.yaml", BaselineRun)
    search = load_config("configs/env/search_live.yaml", SearchConfig)
    original = load_config("configs/env/search.yaml", SearchConfig)
    if search.model_dump(exclude={"energy"}) != original.model_dump(exclude={"energy"}):
        raise ValueError("Only energy may change")
    if search.energy.model_dump(exclude={"dwell_s"}) != original.energy.model_dump(
        exclude={"dwell_s"}
    ):
        raise ValueError("Only dwell may change")
    energy_report = read_json(Path("docs/m12-ppo-revisit/energy-correction.json"))
    if (
        digest({k: v for k, v in energy_report.items() if k != "analysis_id"})
        != energy_report["analysis_id"]
        or search.energy.dwell_s != energy_report["primary"]["candidate_dwell_s"]
    ):
        raise ValueError("D63 fit mismatch")
    table = load_config("configs/levels/default.yaml", LevelTable)
    design = load_config("configs/levels/design_live.yaml", LevelDesign)
    costs = [
        coverage_energy_s(table, level, design.camera.hfov_deg, search.energy)
        for level in table.levels
    ]
    if not (
        design.coarse_share == 0.30
        and search.energy.battery_s == 500
        and costs[0] > 500
        and costs[-1] <= 150
    ):
        raise ValueError("D65 coverage gate failed")
    cache = load_config("configs/cache/default.yaml", CacheConfig)
    worlds = b_worlds(repo, "rl-val")
    sensor_path = Path("data/sensor_models/fcf8cdae58ca3947.json")
    sensor = SensorModel.load(
        sensor_path,
        detector_id=cache.detector_id,
        fit_world_set_id=cache.world_set_ids["sensor-fit"],
    )
    for world in worlds:
        if (
            resume_metadata(cache.out_dir / "rl-val" / f"{world.world_id}.npz", world, cache)
            is None
        ):
            raise ValueError("Missing verified B cache")
    rows, verification = verify_grid(
        args.root,
        worlds,
        search,
        grid,
        [header_for(w, cache).cache_id for w in worlds],
        sha256_file(sensor_path),
    )
    if len(rows) != 3520:
        raise ValueError("Expected 3,520 declared episodes")
    old, historical = original_b5(
        Path("data/baselines") / ORIGINAL_RUN,
        worlds,
        original,
        read_json(args.root / "plan.json"),
        [r["key"] for r in rows if r["key"]["method"] == "B5"],
    )
    results = summarise(rows, old, grid.variants)
    if not results["sanity_passed"]:
        write_json(args.root / "review-failed.json", results)
        raise ValueError("Oracle/random sanity failed; stop without freezing settings")
    selected = SelectedBaselines(
        source_run_id=args.root.name,
        sensor_model_id=sensor.sensor_model_id,
        world_set_id=cache.world_set_ids["rl-val"],
        implementation="m7-v2-fixed-level-routing",
        settings=results["settings"],
    )
    source_names = (
        "docs/m12-ppo-revisit/baseline-protocol.md",
        "configs/env/search_live.yaml",
        "configs/env/energy_live.yaml",
        "configs/levels/design_live.yaml",
        "configs/policy/baselines.yaml",
        "src/altiseek/eval/baseline_review.py",
        "scripts/search/report_live_baselines.py",
        "scripts/search/run_baselines.py",
    )
    report = {
        "decision": "D65",
        "run_id": args.root.name,
        "episodes": len(rows),
        "world_set_id": cache.world_set_ids["rl-val"],
        "sensor_model_id": sensor.sensor_model_id,
        "energy_analysis_id": energy_report["analysis_id"],
        "coverage_costs_s": costs,
        "coarse_share": design.coarse_share,
        "bootstrap_draws": 10_000,
        "bootstrap_seed": 20261106,
        "source_sha256": {name: sha256_file(name) for name in source_names},
        "verification": {k: v for k, v in verification.items() if k != "episode_sha256"},
        "historical_reference": historical,
        **results,
    }
    report["review_id"] = digest(report)
    selected_path = Path("configs/policy/selected_live.yaml")
    selected_path.write_text(
        yaml.safe_dump(selected.model_dump(mode="json"), sort_keys=True), encoding="utf-8"
    )
    output = Path("docs/m12-ppo-revisit/baseline-results.json")
    table_path = output.with_suffix(".md")
    write_json(output, report)
    table_path.write_text(markdown(report), encoding="utf-8")
    write_json(args.root / "episode-checksums.json", verification["episode_sha256"])
    write_metadata(
        args.root / "selection.meta.json",
        stage="select-live-baselines",
        config=selected,
        inputs={
            "plan": args.root / "plan.json",
            "completion": args.root / "complete.json",
            "protocol": Path("docs/m12-ppo-revisit/baseline-protocol.md"),
            "checksums": args.root / "episode-checksums.json",
        },
        outputs={"selected": selected_path, "report": output, "table": table_path},
        extra={"review_id": report["review_id"]},
    )
    print(
        json.dumps(
            {
                "review_id": report["review_id"],
                "episodes": len(rows),
                "sanity_passed": results["sanity_passed"],
                "b5_paired": [
                    {k: v for k, v in r.items() if k != "per_world"} for r in results["b5_paired"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
