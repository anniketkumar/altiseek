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

Paste test (D30): real swimmers sit in exercise scenes, among other swimmers and boats, while
world targets sit alone on open sea, and the detector finds a swimmer more readily near
others. The world-level comparison therefore mixes the renderer with the scene. The paste test
holds the scene fixed: library frames of each split are rectified at g0 (within theta_lib) into
level footprints holding real swimmers, and swimmer crops of the split's library (from other
frames) are pasted in with the renderer's own ``paste``, rotated and scaled as in worlds. In one
copy they lie 3-8 m from a real swimmer (``near``, the scene's context); in another at least
10 m from every object (``isolated``). Each copy goes through the shared resampling path. Real
swimmers are counted on both copies. The renderer's fidelity is near-pasted recall against
real recall on the near copy: the same views, with the pasted and real swimmers among each
other.

Confidence intervals are 95% bootstrap intervals over worlds (rendered) and frames (real,
paste). Per-view rows are appended to ``views.csv`` and ``paste.csv``, so a rerun skips
finished worlds and frames. Writes ``<out>/views.csv``, ``paste.csv``, ``fidelity.csv``,
``paste_summary.csv``, ``fidelity.json`` and ``fidelity.png``:

    uv run python scripts/check_fidelity.py --real data/kaggle/altiseek-eval-levels/eval-levels/eval
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, fields
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from altiseek.config import load_config
from altiseek.data.frames import Box, Frame
from altiseek.data.images import fetch_images, image_path, load_rgb
from altiseek.data.splits import SplitRow
from altiseek.data.tables import read_csv
from altiseek.detector.evaluate import match
from altiseek.detector.infer import Detection, Detector
from altiseek.detector.views import (
    CLASSES,
    CameraModel,
    Label,
    frame_camera,
    labels_for,
    tile_windows,
)
from altiseek.geometry.camera import TiltedCamera
from altiseek.geometry.rectify import ViewWindow, rectify, valid_pixels
from altiseek.imaging import pipeline
from altiseek.registry.run import write_metadata
from altiseek.world.config import LibraryConfig, RenderConfig, WorldConfig
from altiseek.world.library import NO_PIPELINE, Library, select_frames
from altiseek.world.render import ViewObject, compose_world, paste, render_view
from altiseek.world.spec import WorldSpec

SWIMMER = CLASSES.index("swimmer")
#: The real split each rendered world set is compared with.
REAL_SPLIT = {"sensor-fit": "B", "rl-train": "B", "rl-val": "B", "test": "T"}
#: Paste test: crops per placement and view, the near band and isolation distance (metres), and
#: the valid margin a real swimmer needs around it (metres).
PASTES, NEAR_M, ISOLATED_M, REAL_MARGIN_M = 4, (3.0, 8.0), 10.0, 1.5


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


@dataclass(frozen=True)
class PasteRow:
    """One real view of the paste test, a row of ``paste.csv``."""

    split: str
    level: str
    gsd_mpp: float
    image_id: int
    x0_m: float
    y0_m: float
    real: int
    #: Real swimmers found on the isolated copy (their scene as filmed) and on the near copy
    #: (with the pasted swimmers among them).
    real_found: int
    real_near_found: int
    near: int
    near_found: int
    isolated: int
    isolated_found: int


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


def paired_bootstrap(
    a: dict[str, list[float]], b: dict[str, list[float]], rng: np.random.Generator, n: int = 1000
) -> tuple[float, float]:
    """95% interval of ratio(a) - ratio(b), resampling the groups (keys of ``b``) jointly;
    each ratio is sum(numerator) / sum(denominator)."""
    keys = list(b)
    if not keys:
        return math.nan, math.nan
    picks = rng.integers(len(keys), size=(n, len(keys)))

    def resampled(groups: dict[str, list[float]]) -> NDArray[np.float64]:
        num = np.array([groups.get(k, [0.0, 0.0])[0] for k in keys])[picks].sum(axis=1)
        den = np.array([groups.get(k, [0.0, 0.0])[1] for k in keys])[picks].sum(axis=1)
        return num / np.maximum(den, 1e-12)

    diff = resampled(a) - resampled(b)
    return float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))


def found_count(
    points_px: Sequence[tuple[float, float]],
    detections: Sequence[Detection],
    gsd_mpp: float,
    tau: float,
) -> tuple[int, int]:
    """Targets at ``points_px`` (column, row; pixel centres at integers) found by a swimmer
    detection at ``tau`` within 1 m, one-to-one; and the swimmer detections left unmatched."""
    swimmers = [d for d in detections if d.class_id == SWIMMER and d.score >= tau]
    targets = [Label(SWIMMER, c, r, c + 1, r + 1) for c, r in points_px]
    _, hits, used = match(targets, swimmers, SWIMMER, iou_min=None, radius_px=1.0 / gsd_mpp)
    return sum(used), len(hits) - sum(hits)


def view_counts(
    truth: Sequence[ViewObject], detections: Sequence[Detection], gsd_mpp: float, tau: float
) -> tuple[int, int, int, int, int]:
    """targets, found, false, distractors and distractors hit, matching centres within 1 m."""
    targets = [(t.col_px, t.row_px) for t in truth if t.target]
    found, false = found_count(targets, detections, gsd_mpp, tau)
    swimmers = [d for d in detections if d.class_id == SWIMMER and d.score >= tau]
    radius = 1.0 / gsd_mpp
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
    return len(targets), found, false, len(distractors), hit


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


@dataclass(frozen=True)
class ViewScale:
    """Maps g0 pixel positions to view pixels (pixel centres at integers)."""

    factor: float

    def __call__(self, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        f = self.factor
        return [((c + 0.5) * f - 0.5, (r + 0.5) * f - 0.5) for c, r in points]


#: split, frame, its boxes, camera model, library, library config, render config, scale jitter,
#: views per frame and level, seed
PasteTask = tuple[
    str, Frame, list[Box], CameraModel, Library, LibraryConfig, RenderConfig, float, int, int
]
#: The view's paste row (counts still zero), the near and isolated views, and the view-pixel
#: positions of the real, near-pasted and isolated-pasted swimmers.
PasteViews = tuple[
    PasteRow,
    NDArray[np.uint8],
    NDArray[np.uint8],
    list[tuple[float, float]],
    list[tuple[float, float]],
    list[tuple[float, float]],
]


def _centre(label: Label) -> tuple[float, float]:
    """A label's centre, pixel centres at integers."""
    return (label.x_min + label.x_max) / 2 - 0.5, (label.y_min + label.y_max) / 2 - 0.5


def _clear(valid: NDArray[np.bool_], c: float, r: float, half: int) -> bool:
    """Whether the square of half-side ``half`` px around (c, r) is all valid."""
    r0, c0 = round(r) - half, round(c) - half
    if r0 < 0 or c0 < 0 or r0 + 2 * half >= valid.shape[0] or c0 + 2 * half >= valid.shape[1]:
        return False
    return bool(valid[r0 : r0 + 2 * half + 1, c0 : c0 + 2 * half + 1].all())


def rectified_g0(
    rgb: NDArray[np.uint8],
    camera: TiltedCamera,
    window: ViewWindow,
    lib_cfg: LibraryConfig,
    tile_px: int = 1000,
) -> tuple[NDArray[np.uint8], NDArray[np.bool_]]:
    """The window rectified as the library rectifies frames (no camera pipeline), in square
    tiles of ``tile_px`` so the supersampled grid stays small; the last tiles overlap."""
    n = window.size_px
    image = np.empty((n, n, 3), dtype=np.uint8)
    valid = np.zeros((n, n), dtype=bool)
    t = min(tile_px, n)
    starts = sorted({min(s, n - t) for s in range(0, n, t)})
    top = window.y0_m + window.side_m
    for r0 in starts:
        for c0 in starts:
            sub = ViewWindow(
                window.x0_m + c0 * window.gsd_mpp,
                top - (r0 + t) * window.gsd_mpp,
                window.gsd_mpp,
                t,
            )
            view = rectify(
                rgb,
                camera,
                sub,
                max_off_nadir_deg=lib_cfg.max_off_nadir_deg,
                pipeline=NO_PIPELINE,
                supersample=lib_cfg.supersample,
            )
            image[r0 : r0 + t, c0 : c0 + t] = view.image
            valid[r0 : r0 + t, c0 : c0 + t] = view.valid
    return image, valid


def paste_views(task: PasteTask) -> list[PasteViews]:
    """The paste-test views of one frame at every level: up to ``per_frame`` footprints with the
    most real swimmers."""
    split, frame, boxes, model, library, lib_cfg, render, jitter, per_frame, seed = task
    camera = frame_camera(frame, model)
    g0, size = lib_cfg.gsd_mpp, render.camera.resolution_px
    swimmer_boxes = [b for b in boxes if b.category == "swimmer"]
    crops = [o for o in library.objects if o.category == "swimmer" and o.image_id != frame.image_id]
    rgb = load_rgb(image_path(lib_cfg.data.images, frame))
    out = []
    for li, level in enumerate(render.levels.levels):
        side = render.levels.footprint_m(level)
        n0 = round(side / g0)
        gsd = side / size
        ranked = []
        for w in tile_windows(camera, gsd, size, lib_cfg.max_off_nadir_deg, 0.05):
            fine = ViewWindow(w.x0_m, w.y0_m, g0, n0)
            valid, _ = valid_pixels(camera, fine, lib_cfg.max_off_nadir_deg)
            count = len(labels_for(camera, fine, swimmer_boxes, valid))
            if count:
                ranked.append((-count, w.y0_m, w.x0_m, fine))
        for wi, (_, _, _, fine) in enumerate(sorted(ranked)[:per_frame]):
            image, valid = rectified_g0(rgb, camera, fine, lib_cfg)
            margin = math.ceil(REAL_MARGIN_M / g0)
            occupied = [_centre(lab) for lab in labels_for(camera, fine, boxes, valid)]
            real = [
                pt
                for pt in (_centre(lab) for lab in labels_for(camera, fine, swimmer_boxes, valid))
                if _clear(valid, *pt, margin)
            ]
            rng = np.random.default_rng([seed, frame.image_id, li, wi])
            rows, cols = np.nonzero(valid)
            copies = {"near": image.copy(), "isolated": image.copy()}
            placed: dict[str, list[tuple[float, float]]] = {"near": [], "isolated": []}
            for name, canvas in copies.items():
                for _ in range(400 * PASTES):
                    if len(placed[name]) >= PASTES or not len(rows) or not crops:
                        break
                    k = int(rng.integers(len(rows)))
                    c, r = float(cols[k]), float(rows[k])
                    crop = crops[int(rng.integers(len(crops)))]
                    if not _clear(valid, c, r, math.ceil(crop.size_px * 0.75)):
                        continue
                    others = occupied + placed[name]
                    gaps = [math.hypot(c - x, r - y) * g0 for x, y in others]
                    if name == "near":
                        near = [math.hypot(c - x, r - y) * g0 for x, y in real]
                        if min(gaps, default=math.inf) < NEAR_M[0] or not any(
                            d <= NEAR_M[1] for d in near
                        ):
                            continue
                    elif min(gaps, default=math.inf) < ISOLATED_M:
                        continue
                    rotation = float(rng.uniform(0.0, 360.0))
                    scale = 1.0 + float(rng.uniform(-1, 1)) * jitter
                    paste(canvas, library.crop_image(crop.crop_id), crop, (c, r), rotation, scale)
                    placed[name].append((c, r))

            to_view = ViewScale(size / n0)
            row = PasteRow(
                split, level.name, gsd, frame.image_id, fine.x0_m, fine.y0_m, 0, 0, 0, 0, 0, 0, 0
            )
            out.append(
                (
                    row,
                    pipeline.render_view(copies["near"], size, size, render.camera.pipeline),
                    pipeline.render_view(copies["isolated"], size, size, render.camera.pipeline),
                    to_view(real),
                    to_view(placed["near"]),
                    to_view(placed["isolated"]),
                )
            )
    return out


def paste_tasks(
    split: str,
    library: Library,
    lib_cfg: LibraryConfig,
    render: RenderConfig,
    jitter: float,
    n_frames: int,
    per_frame: int,
    seed: int,
    done: set[int],
) -> list[PasteTask]:
    """Up to ``n_frames`` of the split's library frames that hold swimmers, evenly spaced."""
    data = lib_cfg.data.out_dir
    frames = read_csv(data / "frames.csv", Frame)
    split_rows = {r.image_id: r for r in read_csv(data / "splits.csv", SplitRow)}
    boxes: defaultdict[int, list[Box]] = defaultdict(list)
    for box in read_csv(data / "boxes.csv", Box):
        boxes[box.image_id].append(box)
    models = json.loads((data / "camera_models.json").read_text(encoding="utf-8"))
    able = [
        f
        for f in select_frames(
            frames, split_rows, split, lib_cfg.camera_groups, lib_cfg.frame_stride
        )
        if any(b.category == "swimmer" for b in boxes[f.image_id])
    ]
    idx = np.linspace(0, len(able) - 1, min(n_frames, len(able))).round().astype(int)
    chosen = [able[k] for k in sorted(set(idx.tolist())) if able[k].image_id not in done]
    fetch_images(lib_cfg.data.images, chosen, workers=16)
    tasks = []
    for f in chosen:
        m = models[f.camera_group]
        model = CameraModel(m["focal_px"], m["pitch_bias_deg"])
        tasks.append(
            (split, f, boxes[f.image_id], model, library, lib_cfg, render, jitter, per_frame, seed)
        )
    return tasks


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare detection on rendered and real views.")
    parser.add_argument("--render", type=Path, default=Path("configs/world/render.yaml"))
    parser.add_argument("--worlds", type=Path, default=Path("configs/world/default.yaml"))
    parser.add_argument("--library", type=Path, default=Path("configs/world/library.yaml"))
    parser.add_argument("--rendered", nargs="+", default=["sensor-fit", "test"])
    parser.add_argument("--per-family", type=int, default=5, help="worlds per family and set")
    parser.add_argument("--empty", type=int, default=4, help="views without targets per level")
    parser.add_argument("--paste-frames", type=int, default=60, help="paste-test frames per split")
    parser.add_argument("--paste-views", type=int, default=2, help="per frame and level")
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--workers", type=int, default=min(3, os.cpu_count() or 1))
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
    lib_cfg = load_config(args.library, LibraryConfig)
    detector = Detector(args.checkpoint, device=args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    libraries: dict[str, Library] = {}

    def library_of(split: str) -> Library:
        if split not in libraries:
            libraries[split] = Library.load(worlds.libraries[split])
        return libraries[split]

    views_csv = args.out / "views.csv"
    names = [f.name for f in fields(ViewRow)]
    done = set()
    if views_csv.exists():
        with views_csv.open(newline="", encoding="utf-8") as fh:
            done = {r["world_id"] for r in csv.DictReader(fh)}
    else:
        with views_csv.open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(names)
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
            rows = render_rows(spec, library_of(spec.split), render, detector, args.tau, args.empty)
            with views_csv.open("a", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerows([getattr(r, n) for n in names] for r in rows)
            print(f"{world_set}: {i}/{len(chosen)} worlds", flush=True)

    paste_csv = args.out / "paste.csv"
    paste_names = [f.name for f in fields(PasteRow)]
    pasted_done: defaultdict[str, set[int]] = defaultdict(set)
    if paste_csv.exists():
        with paste_csv.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                pasted_done[r["split"]].add(int(r["image_id"]))
    else:
        with paste_csv.open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(paste_names)
    splits = sorted({REAL_SPLIT[w] for w in args.rendered})
    for split in splits:
        tasks = paste_tasks(
            split,
            library_of(split),
            lib_cfg,
            render,
            worlds.scale_jitter,
            args.paste_frames,
            args.paste_views,
            args.seed,
            pasted_done[split],
        )
        with Pool(args.workers) as pool:
            for i, views in enumerate(pool.imap_unordered(paste_views, tasks, chunksize=1), 1):
                rows = []
                for start in range(0, len(views), 8):
                    chunk = views[start : start + 8]
                    batch = detector.detect(np.stack([v[k] for v in chunk for k in (1, 2)]))
                    for j, (row, _, _, real, near, isolated) in enumerate(chunk):
                        near_dets, iso_dets = batch[2 * j], batch[2 * j + 1]
                        g = row.gsd_mpp
                        rows.append(
                            PasteRow(
                                *(getattr(row, n) for n in paste_names[:6]),
                                len(real),
                                found_count(real, iso_dets, g, args.tau)[0],
                                found_count(real, near_dets, g, args.tau)[0],
                                len(near),
                                found_count(near, near_dets, g, args.tau)[0],
                                len(isolated),
                                found_count(isolated, iso_dets, g, args.tau)[0],
                            )
                        )
                with paste_csv.open("a", newline="", encoding="utf-8") as fh:
                    csv.writer(fh).writerows([getattr(r, n) for n in paste_names] for r in rows)
                print(f"paste {split}: {i}/{len(tasks)} frames", flush=True)

    with views_csv.open(newline="", encoding="utf-8") as fh:
        rendered = list(csv.DictReader(fh))
    with (args.real / "views.csv").open(newline="", encoding="utf-8") as fh:
        real = list(csv.DictReader(fh))
    with paste_csv.open(newline="", encoding="utf-8") as fh:
        pasted = list(csv.DictReader(fh))
    rng = np.random.default_rng(0)
    table, out_rows = render.levels, []

    def ratio(groups: dict[str, list[float]]) -> float:
        den = sum(v[1] for v in groups.values())
        return sum(v[0] for v in groups.values()) / den if den else math.nan

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

    paste_rows = []
    for split in splits:
        for level in table.levels:
            sel = [r for r in pasted if r["split"] == split and r["level"] == level.name]
            groups = {
                kind: defaultdict(lambda: [0.0, 0.0])
                for kind in ("real", "real_near", "near", "isolated")
            }
            for r in sel:
                for kind, g in groups.items():
                    g[r["image_id"]][0] += int(r[f"{kind}_found"])
                    g[r["image_id"]][1] += int(r["real" if kind == "real_near" else kind])
            summary = {
                "split": split,
                "level": level.name,
                "gsd_mpp": table.footprint_m(level) / render.camera.resolution_px,
                "frames": len({r["image_id"] for r in sel}),
                "views": len(sel),
            }
            for kind, g in groups.items():
                summary[f"{kind}_swimmers"] = int(sum(v[1] for v in g.values()))
                summary[f"{kind}_recall"] = ratio(g)
                summary[f"{kind}_ci"] = bootstrap({k: tuple(v) for k, v in g.items()}, rng)
            summary["near_gap"] = summary["near_recall"] - summary["real_near_recall"]
            summary["near_gap_ci"] = paired_bootstrap(groups["near"], groups["real_near"], rng)
            paste_rows.append(summary)
    if paste_rows:
        with (args.out / "paste_summary.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(paste_rows[0]))
            writer.writeheader()
            writer.writerows(paste_rows)
    (args.out / "fidelity.json").write_text(
        json.dumps({"worlds": out_rows, "paste": paste_rows}, indent=2) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for world_set in args.rendered:
        sel = [r for r in out_rows if r["world_set"] == world_set]
        gsd = [100 * r["gsd_mpp"] for r in sel]
        for ax, key in zip(axes[:2], ("recall", "false_per_view"), strict=True):
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
    for split in splits:
        sel = [r for r in paste_rows if r["split"] == split]
        gsd = [100 * r["gsd_mpp"] for r in sel]
        for kind, style in (("real_near", "--"), ("near", "-"), ("isolated", ":")):
            axes[2].plot(
                gsd,
                [r[f"{kind}_recall"] for r in sel],
                marker="o",
                ls=style,
                label=f"{split} {kind}",
            )
    axes[0].set_ylabel("swimmer centre recall")
    axes[1].set_ylabel("false swimmer detections per view")
    axes[2].set_ylabel("paste test: centre recall")
    for ax in axes:
        ax.set_xlabel("level GSD (cm/px)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"fidelity at tau = {args.tau:.3f}, detector {detector.detector_id}", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out / "fidelity.png", dpi=130)
    plt.close(fig)
    write_metadata(
        args.out / "check_fidelity.meta.json",
        stage="check-fidelity",
        config=render,
        inputs={"checkpoint": args.checkpoint, "real_views": args.real / "views.csv"},
        outputs={
            "views": views_csv,
            "paste": paste_csv,
            "fidelity": args.out / "fidelity.csv",
            "paste_summary": args.out / "paste_summary.csv",
        },
        extra={
            "detector_id": detector.detector_id,
            "tau": args.tau,
            "rendered": args.rendered,
            "per_family": args.per_family,
            "empty": args.empty,
            "paste_frames": args.paste_frames,
            "paste_views": args.paste_views,
            "seed": args.seed,
            "libraries": {s: library_of(s).library_id for s in splits},
        },
    )
    for row in out_rows:
        print(
            f"{row['world_set']:10} {row['level']} {100 * row['gsd_mpp']:5.1f} cm/px"
            f"  recall rendered {row['rendered_recall']:.2f} real {row['real_recall']:.2f}"
            f" gap {row['recall_gap']:+.2f}  false/view rendered"
            f" {row['rendered_false_per_view']:.2f} real {row['real_false_per_view']:.2f}"
        )
    for row in paste_rows:
        print(
            f"paste {row['split']} {row['level']}  real {row['real_recall']:.2f}"
            f" ({row['real_swimmers']})  real among pasted {row['real_near_recall']:.2f}"
            f"  near {row['near_recall']:.2f} ({row['near_swimmers']})"
            f"  isolated {row['isolated_recall']:.2f} ({row['isolated_swimmers']})"
            f"  near gap {row.get('near_gap', math.nan):+.2f}"
        )


if __name__ == "__main__":
    main()
