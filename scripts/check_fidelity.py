"""check-fidelity: detection on rendered worlds vs real frames, per level (CLAUDE.md §11, M4).

Rendered side: the worlds of ``--rendered`` world sets (sensor-fit worlds use split B's library,
test worlds split T's) are composed and viewed at every level. Each world contributes every
lattice view that holds a target, plus ``--empty`` seeded views without one. The detector runs
at the fixed operating threshold ``--tau``, and swimmer detections are matched to targets by
centre within 1 m (D24). This gives target recall, false detections per view (swimmer
detections more than 1 m from every target, so hits on distractors count) and the share of
distractors hit.

Real side: eval-detector's per-view counts (``views.csv``) on real B and T frames at exactly the
level GSDs and the same tau (Kaggle run ``altiseek-eval-levels``). Centre recall is sum tp_centre
/ sum n_truth, and false detections are per full view of valid pixels.

Confidence intervals are 95% bootstrap intervals over worlds (rendered) and frames (real).
Per-view rows are appended to ``views.csv``, so a rerun skips finished worlds. Writes
``<out>/views.csv``, ``fidelity.csv``, ``fidelity.json`` and ``fidelity.png``:

    uv run python scripts/check_fidelity.py --real data/kaggle/altiseek-eval-levels/eval-levels/eval
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from altiseek.config import load_config
from altiseek.detector.evaluate import match
from altiseek.detector.infer import Detection, Detector
from altiseek.detector.views import CLASSES, Label
from altiseek.registry.run import write_metadata
from altiseek.world.config import RenderConfig, WorldConfig
from altiseek.world.library import Library
from altiseek.world.render import ViewObject, compose_world, render_view
from altiseek.world.spec import WorldSpec

SWIMMER = CLASSES.index("swimmer")
#: The real split each rendered world set is compared with.
REAL_SPLIT = {"sensor-fit": "B", "rl-train": "B", "rl-val": "B", "test": "T"}


@dataclass(frozen=True)
class ViewRow:
    """One rendered view's counts at tau, a row of ``views.csv``."""

    world_set: str
    world_id: str
    family: str
    level: str
    gsd_mpp: float
    ix: int
    iy: int
    targets: int
    found: int
    false: int
    distractors: int
    distractor_hits: int


def bootstrap(
    groups: dict[str, tuple[float, float]], rng: np.random.Generator, n: int = 1000
) -> tuple[float, float]:
    """95% interval of sum(numerator) / sum(denominator), resampling whole groups."""
    keys = list(groups)
    if not keys:
        return math.nan, math.nan
    num = np.array([groups[k][0] for k in keys])
    den = np.array([groups[k][1] for k in keys])
    picks = rng.integers(len(keys), size=(n, len(keys)))
    ratios = num[picks].sum(axis=1) / np.maximum(den[picks].sum(axis=1), 1e-12)
    return float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


def view_counts(
    truth: Sequence[ViewObject], detections: Sequence[Detection], gsd_mpp: float, tau: float
) -> tuple[int, int, int, int, int]:
    """targets, found, false, distractors and distractors hit, matching centres within 1 m."""
    swimmers = [d for d in detections if d.class_id == SWIMMER and d.score >= tau]
    targets = [
        Label(SWIMMER, t.col_px, t.row_px, t.col_px + 1, t.row_px + 1) for t in truth if t.target
    ]
    radius = 1.0 / gsd_mpp
    _, hits, used = match(targets, swimmers, SWIMMER, iou_min=None, radius_px=radius)
    distractors = [t for t in truth if not t.target]
    hit = sum(
        any(
            math.hypot(
                (d.x_min + d.x_max) / 2 - (t.col_px + 0.5),
                (d.y_min + d.y_max) / 2 - (t.row_px + 0.5),
            )
            <= radius
            for d in swimmers
        )
        for t in distractors
    )
    return len(targets), sum(used), len(hits) - sum(hits), len(distractors), hit


def render_rows(
    spec: WorldSpec,
    library: Library,
    render: RenderConfig,
    detector: Detector,
    tau: float,
    empty: int,
) -> list[ViewRow]:
    """Every target-bearing lattice view of each level, plus ``empty`` seeded others."""
    composed = compose_world(spec, library)
    table = render.levels
    rng = np.random.default_rng([spec.seed, 7])
    rows = []
    for level in table.levels:
        side = table.footprint_m(level)
        holding, others = [], []
        for cell in table.lattice(level):
            x0, y0 = cell[0] * table.cell_m, cell[1] * table.cell_m
            inside = any(x0 <= t.x_m < x0 + side and y0 <= t.y_m < y0 + side for t in spec.targets)
            (holding if inside else others).append(cell)
        picked = holding + [
            others[k]
            for k in sorted(rng.choice(len(others), min(empty, len(others)), replace=False))
        ]
        views, truths = [], []
        for cell in picked:
            image, truth = render_view(
                spec,
                composed,
                table=table,
                camera=render.camera,
                level=level,
                cell=cell,
                variant=0,
                settings=render.variants,
            )
            views.append(image)
            truths.append(truth)
        gsd = side / render.camera.resolution_px
        for start in range(0, len(views), 32):
            batch = detector.detect(np.stack(views[start : start + 32]))
            for cell, truth, dets in zip(
                picked[start : start + 32], truths[start : start + 32], batch, strict=True
            ):
                counts = view_counts(truth, dets, gsd, tau)
                rows.append(
                    ViewRow(
                        spec.world_set,
                        spec.world_id,
                        spec.family,
                        level.name,
                        gsd,
                        cell[0],
                        cell[1],
                        *counts,
                    )
                )
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare detection on rendered and real views.")
    parser.add_argument("--render", type=Path, default=Path("configs/world/render.yaml"))
    parser.add_argument("--worlds", type=Path, default=Path("configs/world/default.yaml"))
    parser.add_argument("--rendered", nargs="+", default=["sensor-fit", "test"])
    parser.add_argument("--per-family", type=int, default=5, help="worlds per family and set")
    parser.add_argument("--empty", type=int, default=4, help="views without targets per level")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/kaggle/altiseek-detector-v3/detector/train/weights/best.pt"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--tau", type=float, default=0.37650254368782043, help="D25 operating threshold"
    )
    parser.add_argument(
        "--real", type=Path, required=True, help="eval-detector output (eval/) at the level GSDs"
    )
    parser.add_argument("--out", type=Path, default=Path("data/fidelity"))
    args = parser.parse_args(argv)
    render = load_config(args.render, RenderConfig)
    worlds = load_config(args.worlds, WorldConfig)
    detector = Detector(args.checkpoint, device=args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    views_csv = args.out / "views.csv"
    names = [f.name for f in fields(ViewRow)]
    done = set()
    if views_csv.exists():
        with views_csv.open(newline="", encoding="utf-8") as fh:
            done = {r["world_id"] for r in csv.DictReader(fh)}
    else:
        with views_csv.open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(names)

    libraries: dict[str, Library] = {}
    for world_set in args.rendered:
        folder = worlds.out_dir / world_set
        with (folder / "worlds.csv").open(newline="", encoding="utf-8") as fh:
            listed = list(csv.DictReader(fh))
        per_family: defaultdict[str, int] = defaultdict(int)
        chosen = []
        for r in listed:
            if per_family[r["family"]] < args.per_family:
                per_family[r["family"]] += 1
                chosen.append(r["world_id"])
        for i, world_id in enumerate(chosen, 1):
            if world_id in done:
                continue
            spec = WorldSpec.load(folder / f"{world_id}.json")
            if spec.split not in libraries:
                libraries[spec.split] = Library.load(worlds.libraries[spec.split])
            rows = render_rows(spec, libraries[spec.split], render, detector, args.tau, args.empty)
            with views_csv.open("a", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerows([getattr(r, n) for n in names] for r in rows)
            print(f"{world_set}: {i}/{len(chosen)} worlds", flush=True)

    with views_csv.open(newline="", encoding="utf-8") as fh:
        rendered = list(csv.DictReader(fh))
    with (args.real / "views.csv").open(newline="", encoding="utf-8") as fh:
        real = list(csv.DictReader(fh))
    rng = np.random.default_rng(0)
    table, out_rows = render.levels, []
    for world_set in args.rendered:
        split = REAL_SPLIT[world_set]
        for level in table.levels:
            gsd = table.footprint_m(level) / render.camera.resolution_px
            mine = [r for r in rendered if r["world_set"] == world_set and r["level"] == level.name]
            theirs = [
                r
                for r in real
                if r["split"] == split and math.isclose(float(r["gsd_mpp"]), gsd, rel_tol=1e-6)
            ]
            recall_w: defaultdict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
            false_w: defaultdict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
            for r in mine:
                recall_w[r["world_id"]][0] += int(r["found"])
                recall_w[r["world_id"]][1] += int(r["targets"])
                false_w[r["world_id"]][0] += int(r["false"])
                false_w[r["world_id"]][1] += 1
            recall_f: defaultdict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
            false_f: defaultdict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
            for r in theirs:
                recall_f[r["image_id"]][0] += int(r["tp_centre"])
                recall_f[r["image_id"]][1] += int(r["n_truth"])
                false_f[r["image_id"]][0] += int(r["fp_centre"])
                false_f[r["image_id"]][1] += float(r["valid_share"])

            def ratio(groups: dict[str, list[float]]) -> float:
                den = sum(v[1] for v in groups.values())
                return sum(v[0] for v in groups.values()) / den if den else math.nan

            hits = sum(int(r["distractor_hits"]) for r in mine)
            distractors = sum(int(r["distractors"]) for r in mine)
            out_rows.append(
                {
                    "world_set": world_set,
                    "split": split,
                    "level": level.name,
                    "gsd_mpp": gsd,
                    "rendered_views": len(mine),
                    "rendered_targets": int(sum(v[1] for v in recall_w.values())),
                    "rendered_recall": ratio(recall_w),
                    "rendered_recall_ci": bootstrap(
                        {k: tuple(v) for k, v in recall_w.items()}, rng
                    ),
                    "rendered_false_per_view": ratio(false_w),
                    "rendered_false_ci": bootstrap({k: tuple(v) for k, v in false_w.items()}, rng),
                    "distractor_hit_rate": hits / distractors if distractors else math.nan,
                    "real_views": len(theirs),
                    "real_swimmers": int(sum(v[1] for v in recall_f.values())),
                    "real_recall": ratio(recall_f),
                    "real_recall_ci": bootstrap({k: tuple(v) for k, v in recall_f.items()}, rng),
                    "real_false_per_view": ratio(false_f),
                    "real_false_ci": bootstrap({k: tuple(v) for k, v in false_f.items()}, rng),
                }
            )
    for row in out_rows:
        row["recall_gap"] = row["rendered_recall"] - row["real_recall"]
        row["false_ratio"] = (
            row["rendered_false_per_view"] / row["real_false_per_view"]
            if row["real_false_per_view"]
            else math.nan
        )
    with (args.out / "fidelity.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)
    (args.out / "fidelity.json").write_text(json.dumps(out_rows, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for world_set in args.rendered:
        sel = [r for r in out_rows if r["world_set"] == world_set]
        gsd = [100 * r["gsd_mpp"] for r in sel]
        for ax, key in zip(axes, ("recall", "false_per_view"), strict=True):
            ax.plot(
                gsd, [r[f"rendered_{key}"] for r in sel], marker="o", label=f"rendered {world_set}"
            )
            ax.plot(
                gsd,
                [r[f"real_{key}"] for r in sel],
                marker="s",
                ls="--",
                label=f"real {REAL_SPLIT[world_set]}",
            )
    axes[0].set_ylabel("swimmer centre recall")
    axes[1].set_ylabel("false swimmer detections per view")
    for ax in axes:
        ax.set_xlabel("level GSD (cm/px)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"fidelity at tau = {args.tau:.3f}, detector {detector.detector_id}", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out / "fidelity.png", dpi=130)
    plt.close(fig)
    write_metadata(
        args.out / "check_fidelity.meta.json",
        stage="check-fidelity",
        config=render,
        inputs={"checkpoint": args.checkpoint, "real_views": args.real / "views.csv"},
        outputs={"views": views_csv, "fidelity": args.out / "fidelity.csv"},
        extra={
            "detector_id": detector.detector_id,
            "tau": args.tau,
            "rendered": args.rendered,
            "per_family": args.per_family,
            "empty": args.empty,
        },
    )
    for row in out_rows:
        print(
            f"{row['world_set']:10} {row['level']} {100 * row['gsd_mpp']:5.1f} cm/px"
            f"  recall rendered {row['rendered_recall']:.2f} real {row['real_recall']:.2f}"
            f" gap {row['recall_gap']:+.2f}  false/view rendered"
            f" {row['rendered_false_per_view']:.2f} real {row['real_false_per_view']:.2f}"
        )


if __name__ == "__main__":
    main()
