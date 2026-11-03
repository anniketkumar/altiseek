"""Numeric M10 transfer report (D56) from verified saved results only.

Refuses to report unless every frozen live and cached case is complete and its
checksum verifies. Writes docs/m10-deployment/m10-transfer-results/{results.json,results.md,
episodes.csv}. No figure or table is edited by hand.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from altiseek.cache.build import header_for
from altiseek.cache.config import CacheConfig
from altiseek.cache.storage import WorldCache
from altiseek.calibration.config import SensorConfig
from altiseek.calibration.model import digest
from altiseek.calibration.outcomes import cell_outcomes
from altiseek.config import load_config
from altiseek.deploy.transfer import METHODS, live_cases, load_protocol, paired_seed
from altiseek.deploy.transfer_report import (
    CONDITIONS,
    GROUPS,
    group_of,
    paired,
    rate_difference,
    recovered_fraction,
    summary,
    world_values,
)
from altiseek.registry.run import sha256_file
from altiseek.world.spec import WorldSpec

METRICS = (
    "auc",
    "found_fraction",
    "success",
    "first_find_s",
    "false_declarations",
    "energy_s",
    "elapsed_s",
    "path_m",
)


def verified(path: Path, pid: str) -> dict:
    saved = json.loads(path.read_text())
    if saved["protocol_id"] != pid or sha256_file(Path(saved["report_path"])) != saved["sha256"]:
        raise ValueError(f"Unverified transfer result {path}")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--out", type=Path, default=Path("docs/m10-deployment/m10-transfer-results")
    )
    parser.add_argument(
        "--partial", action="store_true", help="Code check on complete worlds only; not a result"
    )
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    pid, root = protocol["protocol_id"], args.root
    seed, samples = protocol["bootstrap"]["seed"], protocol["bootstrap"]["samples"]
    cache = load_config("configs/cache/default.yaml", CacheConfig)
    tau = load_config("configs/sensor/default.yaml", SensorConfig).tau
    family = {w["world_id"]: w["family"] for w in protocol["worlds"]}
    specs = {
        w: WorldSpec.load(cache.worlds_dir / "test" / f"{w}.json") for w in protocol["world_ids"]
    }

    rows, live = [], []
    for case in live_cases(protocol["world_ids"]):
        key = {**case, "smoke": False}
        if args.partial and not (root / "live" / f"{digest(key)}.json").exists():
            continue
        saved = verified(root / "live" / f"{digest(key)}.json", pid)
        if saved["key"] != key:
            raise ValueError("Live completion key mismatch")
        report_path = Path(saved["report_path"])
        result = json.loads(report_path.read_text())
        retries = report_path.parent / "capture-retries.jsonl"
        recaptures = len(retries.read_text().splitlines()) if retries.exists() else 0
        rows.append(
            {
                "world_id": case["world_id"],
                "family": family[case["world_id"]],
                "group": group_of(case["method"], case["training_seed"]),
                "condition": f"live-{case['sensor']}",
                **{m: result["metrics"][m] for m in METRICS},
                "views": len(result["views"]),
                "recaptures": recaptures,
                "wall_s": saved["wall_s"],
                "flight": str(report_path.parent),
            }
        )
        live.append((case, result, report_path.parent))
    for world_id in protocol["world_ids"]:
        for method, training_seed in METHODS:
            key = {
                "world_id": world_id,
                "method": method,
                "training_seed": training_seed,
                "variant": "V3",
                "environment_seed": paired_seed(world_id, "launch"),
                "action_seed": paired_seed(world_id, "actions", training_seed),
            }
            if args.partial and not (root / "cached" / f"{digest(key)}.json").exists():
                continue
            saved = verified(root / "cached" / f"{digest(key)}.json", pid)
            if saved["key"] != key:
                raise ValueError("Cached reference key mismatch")
            metrics = json.loads(Path(saved["report_path"]).read_text())["metrics"]
            rows.append(
                {
                    "world_id": world_id,
                    "family": family[world_id],
                    "group": group_of(method, training_seed),
                    "condition": "cached",
                    **{m: metrics[m] for m in METRICS},
                }
            )
    if args.partial:
        per_world = defaultdict(int)
        for row in rows:
            per_world[row["world_id"]] += 1
        complete = {w for w, n in per_world.items() if n == 3 * len(METHODS)}
        if not complete:
            raise SystemExit("No world has every live and cached case yet")
        rows = [r for r in rows if r["world_id"] in complete]
        live = [item for item in live if item[0]["world_id"] in complete]
        family = {w: f for w, f in family.items() if w in complete}
    elif len(rows) != protocol["live_episodes"] + protocol["cached_episodes"]:
        raise ValueError("Incomplete transfer result set")

    tables: dict = {"summary": {}, "paired": {}, "recovered_fraction": {}, "per_family_auc": {}}
    for metric in METRICS:
        tables["summary"][metric] = {
            group: {
                condition: summary(world_values(rows, metric, group, condition), seed, samples)
                for condition in CONDITIONS
            }
            for group in GROUPS
        }
        comparisons = {}
        for group in GROUPS:
            values = {c: world_values(rows, metric, group, c) for c in CONDITIONS}
            comparisons[f"{group}: live-original - cached"] = paired(
                values["live-original"], values["cached"], seed, samples
            )
            comparisons[f"{group}: live-refit - cached"] = paired(
                values["live-refit"], values["cached"], seed, samples
            )
            comparisons[f"{group}: live-refit - live-original"] = paired(
                values["live-refit"], values["live-original"], seed, samples
            )
        for condition in CONDITIONS:
            comparisons[f"PPO - B5: {condition}"] = paired(
                world_values(rows, metric, "PPO", condition),
                world_values(rows, metric, "B5", condition),
                seed,
                samples,
            )
        tables["paired"][metric] = comparisons
    for group in GROUPS:
        values = {c: world_values(rows, "auc", group, c) for c in CONDITIONS}
        tables["recovered_fraction"][group] = recovered_fraction(
            values["cached"], values["live-original"], values["live-refit"], seed, samples
        )
        tables["per_family_auc"][group] = {
            f: {
                c: float(np.mean([v for w, v in values[c].items() if family[w] == f]))
                for c in CONDITIONS
            }
            for f in sorted(set(family.values()))
        }

    # Level-wise detector diagnostic on executed live views against cached variant 0
    # at the same lattice node: cells fully visible live and inside the nominal footprint.
    table = cache.render.levels
    counts = {
        kind: [defaultdict(lambda: np.zeros(4, dtype=np.int64)) for _ in table.levels]
        for kind in ("recall", "false_alarm")
    }
    caches = {}
    for case, result, flight in live:
        world = specs[case["world_id"]]
        if world.world_id not in caches:
            path = cache.out_dir / "test" / f"{world.world_id}.npz"
            caches[world.world_id] = WorldCache.load(path, expected=header_for(world, cache))
        occupied = {(int(t.x_m // world.cell_m), int(t.y_m // world.cell_m)) for t in world.targets}
        for view in result["views"]:
            li = view["level"]
            level = table.levels[li]
            goal = np.asarray(view["goal_enu_m"][:2])
            cell = next(
                c for c in table.lattice(level) if np.allclose(table.centre_m(level, c), goal)
            )
            with np.load(flight / f"detections-{view['stamp_s']:.6f}.npz") as arrays:
                detections, visible = arrays["detections"], arrays["visible"]
            bounds = (
                cell[0],
                cell[1],
                cell[0] + level.footprint_cells,
                cell[1] + level.footprint_cells,
            )
            inside = np.zeros_like(visible)
            inside[bounds[0] : bounds[2], bounds[1] : bounds[3]] = True
            cells = {tuple(c) for c in np.argwhere(visible & inside)}
            live_hits = {c for c, s in cell_outcomes(detections, bounds).items() if s >= tau}
            cached_detections, _ = caches[world.world_id].lookup(li, *cell, 0)
            cached_hits = {
                c for c, s in cell_outcomes(cached_detections, bounds).items() if s >= tau
            }
            truth = cells & occupied
            empty = cells - occupied
            counts["recall"][li][world.world_id] += (
                len(live_hits & truth),
                len(truth),
                len(cached_hits & truth),
                len(truth),
            )
            counts["false_alarm"][li][world.world_id] += (
                len(live_hits & empty),
                len(empty),
                len(cached_hits & empty),
                len(empty),
            )
    tables["detector_levels"] = [
        {
            "level": level.name,
            **{
                kind: rate_difference(dict(counts[kind][li]), seed, samples)
                for kind in ("recall", "false_alarm")
            },
        }
        for li, level in enumerate(table.levels)
    ]

    preflight = []
    for path in sorted((root / "preflight").glob("*.json")):
        saved = verified(path, pid)
        result = json.loads(Path(saved["report_path"]).read_text())
        retries = Path(saved["report_path"]).parent / "capture-retries.jsonl"
        preflight.append(
            {
                "group": group_of(saved["key"]["method"], saved["key"]["training_seed"]),
                "world_id": saved["key"]["world_id"],
                "sensor": saved["key"]["sensor"],
                **{m: result["metrics"][m] for m in METRICS},
                "views": len(result["views"]),
                "recaptures": len(retries.read_text().splitlines()) if retries.exists() else 0,
                "wall_s": saved["wall_s"],
            }
        )
    failures_log = root / "failures.jsonl"
    failures = (
        [json.loads(line) for line in failures_log.read_text().splitlines()]
        if failures_log.exists()
        else []
    )
    live_rows = [r for r in rows if r["condition"] != "cached"]
    operations = {
        "live_episodes": len(live_rows),
        "failed_attempts_deferred_and_reflown": len(failures),
        "episodes_with_recaptures": sum(r["recaptures"] > 0 for r in live_rows),
        "recaptures": sum(r["recaptures"] for r in live_rows),
        "live_wall_hours": sum(r["wall_s"] for r in live_rows) / 3600,
        "mean_wall_s": float(np.mean([r["wall_s"] for r in live_rows])),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    output = {
        "protocol_id": pid,
        "partial": args.partial,
        "worlds": len(family),
        "source_sha256": sha256_file(Path(__file__)),
        "bootstrap": protocol["bootstrap"],
        "notes": [
            "Worlds are the bootstrap unit; PPO seeds are averaged within world for 'PPO'.",
            "T worlds contain no distractors; original family labels are not density contrasts.",
            "Pose is Gazebo ground truth; this is simulator-to-simulator transfer only.",
            "Detector-level rows repeat lattice nodes across episodes and are clustered by world.",
            "Preflight rows are B integration checks, not held-out results.",
        ],
        "tables": tables,
        "operations": operations,
        "failures": failures,
        "preflight": preflight,
    }
    (args.out / "results.json").write_text(json.dumps(output, indent=2) + "\n")
    with (args.out / "episodes.csv").open("w", newline="") as handle:
        fields = ["world_id", "family", "group", "condition", *METRICS]
        fields += ["views", "recaptures", "wall_s", "flight"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "results.md").write_text(markdown(output))


def markdown(output: dict) -> str:
    def ci(values: list[float] | None) -> str:
        return "n/a" if values is None else f"[{values[0]:.3f}, {values[1]:.3f}]"

    t = output["tables"]
    lines = [
        "# M10 transfer results (generated)",
        "",
        f"Protocol `{output['protocol_id']}`. Generated by "
        "`scripts/deploy/report_m10_transfer.py`;",
        "do not edit. Intervals are 95% whole-world bootstrap intervals.",
        "",
        *[f"- {note}" for note in output["notes"]],
        "",
        "## Found-fraction AUC by condition",
        "",
        "| Policy | Cached | Live, original sensor | Live, refit sensor |",
        "|---|---|---|---|",
    ]
    for group, row in t["summary"]["auc"].items():
        cells = [f"{row[c]['mean']:.3f} {ci(row[c]['ci95'])}" for c in CONDITIONS]
        lines.append(f"| {group} | " + " | ".join(cells) + " |")
    lines += ["", "## Paired differences", ""]
    for metric in ("auc", "found_fraction", "success", "false_declarations", "first_find_s"):
        lines += [
            f"### {metric}",
            "",
            "| Comparison | First | Second | Difference [95% CI] | Worlds first/second better |",
            "|---|---|---|---|---|",
        ]
        for name, p in t["paired"][metric].items():
            lines.append(
                f"| {name} | {p['first']:.3f} | {p['second']:.3f} | "
                f"{p['difference']:+.3f} {ci(p['ci95'])} | "
                f"{p['first_better_worlds']}/{p['second_better_worlds']} |"
            )
        lines.append("")
    lines += [
        "## Gap recovered by the refit sensor (AUC)",
        "",
        "| Policy | Gap cached - live original [95% CI] | Fraction recovered [95% CI] |",
        "|---|---|---|",
    ]
    for group, r in t["recovered_fraction"].items():
        fraction = (
            f"{r['fraction']:.2f} {ci(r['ci95'])}"
            if r["defined"]
            else "undefined (gap not resolved)"
        )
        lines.append(f"| {group} | {r['gap']:+.3f} {ci(r['gap_ci95'])} | {fraction} |")
    lines += [
        "",
        "## Detector outcomes on executed views (live vs cached variant 0)",
        "",
        "| Level | Live recall | Cached recall | Recall diff CI"
        " | Live FA | Cached FA | FA diff CI |",
        "|---|---|---|---|---|---|---|",
    ]

    def rate(x: float | None, scale: float = 1.0) -> str:
        return "n/a" if x is None else f"{x * scale:.3f}"

    for row in t["detector_levels"]:
        r, f = row["recall"], row["false_alarm"]
        fa_ci = None if f["difference_ci95"] is None else [v * 1e4 for v in f["difference_ci95"]]
        trials = f"({r['live']['hits']}/{r['live']['trials']})"
        lines.append(
            f"| {row['level']} | {rate(r['live_rate'])} {trials}"
            f" | {rate(r['cached_rate'])} | {ci(r['difference_ci95'])}"
            f" | {rate(f['live_rate'], 1e4)}e-4 | {rate(f['cached_rate'], 1e4)}e-4"
            f" | {ci(fa_ci)}e-4 |"
        )
    lines += ["", "## Per-family AUC (3 worlds each; descriptive only)", ""]
    lines += ["| Policy | Family | Cached | Live original | Live refit |", "|---|---|---|---|---|"]
    for group in ("B5", "PPO"):
        for fam, row in t["per_family_auc"][group].items():
            lines.append(
                f"| {group} | {fam} | " + " | ".join(f"{row[c]:.3f}" for c in CONDITIONS) + " |"
            )
    o = output["operations"]
    lines += [
        "",
        "## Operations",
        "",
        f"- Live episodes: {o['live_episodes']}; failed attempts deferred and re-flown: "
        f"{o['failed_attempts_deferred_and_reflown']}.",
        f"- Same-waypoint recaptures: {o['recaptures']} in "
        f"{o['episodes_with_recaptures']} episodes.",
        f"- Live wall time: {o['live_wall_hours']:.1f} h; "
        f"mean {o['mean_wall_s']:.0f} s per episode.",
        "",
        "## B preflight (refit sensor; integration, not held-out)",
        "",
        "| Policy | AUC | Found | Sim time (s) | Views | Recaptures | Wall (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in output["preflight"]:
        lines.append(
            f"| {p['group']} | {p['auc']:.3f} | {p['found_fraction']:.2f} | "
            f"{p['elapsed_s']:.0f} | {p['views']} | {p['recaptures']} | {p['wall_s']:.0f} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
