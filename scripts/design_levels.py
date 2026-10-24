"""design-levels: propose altitude levels, cell size and battery from the recall curve (§8, M3).

Reads eval-detector's swimmer recall per GSD bin on split B, searches level tables under the §8
rules (``altiseek.geometry.level_design``), and writes ``<out>/candidates.md`` (the best designs),
``<out>/proposal.yaml`` (the best one, in the ``configs/levels/`` format) and
``<out>/level_design.png`` (the curve and the best design's levels). Choosing the final table
stays a reviewed decision (D7): copy the proposal to ``configs/levels/default.yaml`` and log it.
If no design is feasible, ``candidates.md`` names the rules the curve cannot meet, the figure
shows the curve alone, and the exit status is non-zero.

    uv run python scripts/design_levels.py [--config configs/levels/design.yaml] [--out data/levels]
"""

import argparse
import csv
import math
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from altiseek.config import load_config
from altiseek.geometry.design_config import LevelDesign
from altiseek.geometry.level_design import Candidate, candidates, unmet_recall_rules
from altiseek.registry.run import write_metadata


def read_curve(path: Path, split: str) -> list[tuple[float, float]]:
    """``(gsd_mpp, recall)`` rows of one split, skipping bins without a finite recall."""
    with path.open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["split"] == split]
    curve = [(float(r["gsd_mpp"]), float(r["recall"])) for r in rows]
    return [(g, r) for g, r in curve if math.isfinite(r)]


def describe(c: Candidate) -> str:
    lines = [
        f"cell {c.table.cell_m:g} m, grid {c.table.grid_cells}, battery {c.battery_s:.0f} s, "
        f"largest recall step {c.largest_recall_step:.2f}",
        "",
        "| Level | Footprint | Altitude | Sim GSD | Stride | Views | Recall | Full coverage |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for lvl, h, g, r, e in zip(
        c.table.levels, c.altitude_m, c.gsd_mpp, c.recall, c.coverage_energy_s, strict=True
    ):
        lines.append(
            f"| {lvl.name} | {c.table.footprint_m(lvl):g} m ({lvl.footprint_cells} cells) "
            f"| {h:.1f} m | {100 * g:.1f} cm/px | {lvl.stride_cells} cells "
            f"| {c.table.views_per_world(lvl)} | {r:.2f} | {e:.0f} s |"
        )
    return "\n".join(lines)


def no_design(reasons: Sequence[str]) -> str:
    if not reasons:
        return (
            "# No feasible level design\n\nRecall at both ends is reachable, but no set of "
            "whole-cell footprints has recall falling at every level and a battery that fits "
            "the coverage rule (§8).\n"
        )
    return (
        "# No feasible level design\n\nThe measured curve cannot meet these rules (§8):\n\n"
        + "".join(f"- {r}\n" for r in reasons)
    )


def plot(
    curve: Sequence[tuple[float, float]], best: Candidate | None, cfg: LevelDesign, path: Path
) -> None:
    """The measured curve, the §8 recall limits and the best design's levels, if any."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(
        [100 * g for g, _ in curve],
        [r for _, r in curve],
        marker="o",
        label=f"measured ({cfg.recall_split})",
    )
    title = f"R = {cfg.camera.resolution_px}, HFOV {cfg.camera.hfov_deg:g} deg"
    if best is None:
        title += ": no feasible design"
    else:
        for lvl, g, r in zip(best.table.levels, best.gsd_mpp, best.recall, strict=True):
            ax.scatter([100 * g], [r], color="red", zorder=3)
            ax.annotate(lvl.name, (100 * g, r), textcoords="offset points", xytext=(5, 5))
        title += f", cell {best.table.cell_m:g} m, battery {best.battery_s:.0f} s"
    ax.axhline(cfg.finest_min_recall, color="grey", lw=0.8, ls="--", label="finest-level minimum")
    ax.axhline(
        cfg.coarsest_max_recall, color="grey", lw=0.8, ls=":", label="coarsest-level maximum"
    )
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks([100 * g for g, _ in curve], [f"{100 * g:g}" for g, _ in curve])
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("simulated GSD (cm/px)")
    ax.set_ylabel("swimmer recall")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Propose altitude levels from the recall curve.")
    parser.add_argument("--config", type=Path, default=Path("configs/levels/design.yaml"))
    parser.add_argument("--out", type=Path, default=Path("data/levels"))
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("overrides", nargs="*", help="config overrides, dotted.key=value")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, LevelDesign, args.overrides)

    curve = read_curve(cfg.recall_csv, cfg.recall_split)
    if len(curve) < 2:
        raise SystemExit(f"no recall curve for split {cfg.recall_split} in {cfg.recall_csv}")
    found = candidates(
        area_m=cfg.area_m,
        cell_options_m=cfg.cell_options_m,
        level_counts=cfg.level_counts,
        curve=curve,
        resolution_px=cfg.camera.resolution_px,
        hfov_deg=cfg.camera.hfov_deg,
        energy=cfg.energy,
        finest_min=cfg.finest_min_recall,
        coarsest_max=cfg.coarsest_max_recall,
        coarse_share=cfg.coarse_share,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    best = found[0] if found else None
    report = args.out / "candidates.md"
    proposal = args.out / "proposal.yaml"
    figure = args.out / "level_design.png"
    outputs = {"candidates": report, "figure": figure}
    plot(curve, best, cfg, figure)
    if best is None:
        reasons = unmet_recall_rules(curve, cfg.finest_min_recall, cfg.coarsest_max_recall)
        report.write_text(no_design(reasons), encoding="utf-8")
        proposal.unlink(missing_ok=True)  # never leave an earlier run's proposal behind
    else:
        report.write_text(
            f"# Level design candidates ({len(found)} feasible)\n\n"
            + "\n\n".join(f"## {i + 1}\n\n{describe(c)}" for i, c in enumerate(found[: args.top]))
            + "\n",
            encoding="utf-8",
        )
        proposal.write_text(
            yaml.safe_dump(
                {**best.table.model_dump(mode="json"), "battery_s": best.battery_s},
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        outputs["proposal"] = proposal
    write_metadata(
        args.out / "design_levels.meta.json",
        stage="design-levels",
        config=cfg,
        inputs={"recall": cfg.recall_csv},
        outputs=outputs,
        extra={"feasible": len(found)},
    )
    if best is None:
        raise SystemExit(f"no feasible design; see {report}")
    print(describe(best))


if __name__ == "__main__":
    main()
