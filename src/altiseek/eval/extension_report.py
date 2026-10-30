"""Verified M9 extension tables, paired world intervals and numeric figures."""

from collections import defaultdict
from pathlib import Path

import numpy as np

from altiseek.cache.jobs import write_json
from altiseek.calibration.model import digest
from altiseek.eval.extension import ABLATIONS, ExtensionProtocol, keys
from altiseek.eval.reporting import METRICS, curve, interval
from altiseek.eval.runner import read, verified_episode
from altiseek.registry.run import sha256_file


def aggregate(rows: list[dict]) -> tuple[dict, dict]:
    """Average action repeats, then training seeds; worlds remain the sampling unit."""
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        k, m = row["key"], row["metrics"]
        policy = (k["scenario"], k["variant"], k["method"], k["training_seed"])
        vector = np.r_[
            [float(m[n]) for n in METRICS],
            m["time_per_level_s"],
            curve(row["records"], row["target_count"], 500),
            row["coarse_detection_opportunities"],
            row["immediate_descents"],
        ]
        if not np.isfinite(vector).all() or not 0 <= m["auc"] <= 1 + 1e-9:
            raise ValueError("invalid episode metrics")
        if m["energy_s"] > 500 + 1e-6 or not 0 <= m["found_fraction"] <= 1:
            raise ValueError("invalid episode budget/found fraction")
        grouped[policy][k["world_id"]].append(vector)
    per_seed = {
        p: {w: np.mean(v, axis=0) for w, v in worlds.items()} for p, worlds in grouped.items()
    }
    combined = defaultdict(lambda: defaultdict(list))
    for (case, variant, method, _), worlds in per_seed.items():
        for wid, vector in worlds.items():
            combined[(case, variant, method)][wid].append(vector)
    means = {
        p: {w: np.mean(v, axis=0) for w, v in worlds.items()} for p, worlds in combined.items()
    }
    return per_seed, means


def report(root: Path, output: Path) -> None:
    plan = read(root / "plan.json")
    if digest(plan) != root.name or plan["smoke"]:
        raise ValueError("requires a complete main extension run")
    protocol = ExtensionProtocol.model_validate(plan["protocol"])
    manifest = plan["manifest"]
    jobs = keys(protocol, manifest)
    if read(root / "status.json") != {"completed": len(jobs), "total": len(jobs), "complete": True}:
        raise ValueError("extension run incomplete")
    rows, checksums = [], {}
    descriptors = {}
    for case, dataset in manifest["datasets"].items():
        for w in dataset["worlds"]:
            descriptors[(case, w["world_id"])] = w
    for key in jobs:
        path = root / "episodes" / f"{digest(key)}.json"
        row = verified_episode(path, root.name, key)
        if row is None:
            raise ValueError(f"missing episode: {path}")
        w = descriptors[(key["scenario"], key["world_id"])]
        if row["family"] != w["family"] or row["target_count"] != w["targets"]:
            raise ValueError("episode world descriptor mismatch")
        rows.append(row)
        checksums[path.as_posix()] = sha256_file(path)
    # Reuse original nominal episodes, retaining both their checksums and provenance.
    v1 = Path(manifest["v1_run"])
    published_path = Path("docs/m9-results/results.json")
    if sha256_file(published_path) != manifest["files"][published_path.as_posix()]:
        raise ValueError("original published reference report changed")
    published = read(published_path)
    old_plan = read(v1 / "plan.json")
    if digest(old_plan) != v1.name or not read(v1 / "status.json")["complete"]:
        raise ValueError("invalid original reference run")
    original_worlds = manifest["datasets"]["control"]["worlds"]
    for w in original_worlds:
        descriptors[("original", w["world_id"])] = w
    reference_manifest = {
        "datasets": {"original": {"worlds": original_worlds}},
        "historical": manifest["historical"],
        "ablations": {},
    }
    for new_key in keys(protocol, reference_manifest):
        old_key = new_key | {"scenario": "nominal"}
        path = v1 / "episodes" / f"{digest(old_key)}.json"
        if sha256_file(path) != published["episode_sha256"][path.name]:
            raise ValueError("original reference differs from published checksum")
        row = verified_episode(path, v1.name, old_key)
        if row is None:
            raise ValueError("missing original nominal reference")
        w = descriptors[("original", new_key["world_id"])]
        row.update(key=new_key, target_count=w["targets"], condition=w["condition"])
        rows.append(row)
        checksums[path.as_posix()] = sha256_file(path)
    per_seed, means = aggregate(rows)
    tables = []
    group_ids = {}
    for case, variant, method in means:
        vectors = means[(case, variant, method)]
        world_ids = sorted(vectors)
        families = {"-".join(descriptors[(case, w)]["family"].split("-")[:2]) for w in world_ids}
        groups = {"all": world_ids}
        groups.update(
            {
                f: [w for w in world_ids if descriptors[(case, w)]["family"].startswith(f + "-")]
                for f in sorted(families)
            }
        )
        groups.update(
            {
                f"condition-{c}": [w for w in world_ids if descriptors[(case, w)]["condition"] == c]
                for c in (0, 1)
            }
        )
        for group, ids in groups.items():
            if not ids:
                continue
            group_ids[(case, group)] = ids
            values = np.stack([vectors[w] for w in ids])
            seeds = {
                str(seed): {
                    name: float(np.mean([v[w][i] for w in ids])) for i, name in enumerate(METRICS)
                }
                for (c, var, meth, seed), v in per_seed.items()
                if (c, var, meth) == (case, variant, method) and seed is not None
            }
            reference_case = "original" if case in ABLATIONS else case
            random_auc = np.mean([means[(reference_case, variant, "random")][w][0] for w in ids])
            oracle_auc = np.mean([means[(reference_case, variant, "oracle")][w][0] for w in ids])
            denominator = oracle_auc - random_auc
            tables.append(
                {
                    "case": case,
                    "variant": variant,
                    "method": method,
                    "group": group,
                    "worlds": len(ids),
                    "metrics": {
                        name: {
                            "mean": float(values[:, i].mean()),
                            "ci95": interval(values[:, i], protocol),
                        }
                        for i, name in enumerate(METRICS)
                    },
                    "training_seeds": seeds,
                    "auc_seed_sd": float(np.std([s["auc"] for s in seeds.values()], ddof=1))
                    if seeds
                    else None,
                    "normalised_auc": float((values[:, 0].mean() - random_auc) / denominator)
                    if denominator > 0
                    else None,
                    "normalisation_reference_case": reference_case,
                    "time_per_level_s": values[:, len(METRICS) : len(METRICS) + 5]
                    .mean(axis=0)
                    .tolist(),
                    "found_curve": values[:, len(METRICS) + 5 : len(METRICS) + 106]
                    .mean(axis=0)
                    .tolist(),
                    "descent_after_detection_rate": float(values[:, -1].sum() / values[:, -2].sum())
                    if values[:, -2].sum()
                    else None,
                }
            )
    paired = []
    for (case, variant, method), vectors in means.items():
        refs = []
        if case in ABLATIONS and case != "control":
            refs.append(
                ("appearance_control" if case == "appearance" else "control", variant, "PPO")
            )
        if case in ("detector", "camera"):
            refs.append(("original", variant, method))
        if case == "stress-high":
            refs.append(("stress-low", variant, method))
        if method == "PPO":
            refs.append(("original" if case in ABLATIONS else case, variant, "B5"))
        for reference in refs:
            for (group_case, group), ids in group_ids.items():
                if group_case != case:
                    continue
                if case == "stress-high" and reference[0] == "stress-low":
                    by_seed = {descriptors[("stress-low", w)]["seed"]: w for w in means[reference]}
                    ref_ids = [by_seed[descriptors[(case, w)]["seed"]] for w in ids]
                else:
                    ref_ids = ids
                diffs = np.stack(
                    [vectors[w] - means[reference][r] for w, r in zip(ids, ref_ids, strict=True)]
                )
                paired.append(
                    {
                        "method": [case, variant, method],
                        "reference": list(reference),
                        "group": group,
                        "worlds": len(ids),
                        "metrics": {
                            name: {
                                "difference": float(diffs[:, i].mean()),
                                "ci95": interval(diffs[:, i], protocol),
                            }
                            for i, name in enumerate(METRICS)
                        },
                    }
                )
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "run_id": root.name,
        "new_episodes": len(jobs),
        "reference_episodes": len(rows) - len(jobs),
        "protocol": plan["protocol"],
        "rows": tables,
        "paired": paired,
        "episode_sha256": checksums,
        "per_world": [
            {
                "case": c,
                "variant": v,
                "method": m,
                "world_id": w,
                "metrics": dict(zip(METRICS, map(float, vector[: len(METRICS)]), strict=True)),
            }
            for (c, v, m), worlds in means.items()
            for w, vector in worlds.items()
        ],
    }
    write_json(output / "results.json", result)
    lines = [
        "# M9 extension results",
        "",
        f"Run `{root.name}`: {len(jobs):,} new episodes; "
        f"{len(rows) - len(jobs):,} original nominal episodes reused.",
        "",
        "Intervals bootstrap worlds after averaging action repeats and training seeds.",
        "Stress comparisons pair low/high worlds by seed; these use B imagery, not held-out T.",
        "Appearance condition 1 has only 11 T worlds and was held out of policy training only.",
        "Re-look assumes independent evidence across distinct variants; "
        "calibration remains imperfect.",
        "",
        "| Case | Variant | Policy | AUC [95% CI] | Found | False declarations | Seed SD |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in tables:
        if row["group"] != "all":
            continue
        auc = row["metrics"]["auc"]
        sd = "-" if row["auc_seed_sd"] is None else f"{row['auc_seed_sd']:.3f}"
        lines.append(
            f"| {row['case']} | {row['variant']} | {row['method']} | "
            f"{auc['mean']:.3f} [{auc['ci95'][0]:.3f}, {auc['ci95'][1]:.3f}] | "
            f"{row['metrics']['found_fraction']['mean']:.3f} | "
            f"{row['metrics']['false_declarations']['mean']:.2f} | {sd} |"
        )
    (output / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    figures(tables, output)
    print(f"Verified {len(rows)} episodes; report at {output}", flush=True)


def figures(rows: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    chosen = [next(r for r in rows if r["case"] == c and r["group"] == "all") for c in ABLATIONS]
    axes[0].barh(range(8), [r["metrics"]["auc"]["mean"] for r in chosen])
    axes[0].set(yticks=range(8), yticklabels=ABLATIONS, xlabel="Mean AUC on T", xlim=(0, 1))
    for case in ("original", "detector", "camera", "stress-low", "stress-high"):
        r = next(
            r
            for r in rows
            if r["case"] == case
            and r["variant"] == "V3"
            and r["method"] == "PPO"
            and r["group"] == "all"
        )
        axes[1].plot(np.linspace(0, 500, 101), r["found_curve"], label=case)
    axes[1].set(xlabel="Elapsed time (s)", ylabel="Found fraction", ylim=(0, 1))
    axes[1].legend()
    for method in ("PPO", "B3", "B5"):
        r = next(
            r
            for r in rows
            if r["case"] == "stress-high"
            and r["variant"] == "V3"
            and r["method"] == method
            and r["group"] == "all"
        )
        axes[2].plot(range(1, 6), r["time_per_level_s"], marker="o", label=method)
    axes[2].set(xlabel="Level", ylabel="Mean time (s)", title="High-density B stress")
    axes[2].legend()
    fig.savefig(output / "extension_summary.png", dpi=160)
    plt.close(fig)
