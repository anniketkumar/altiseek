"""eval-detector: swimmer AP50, recall, precision and false positives per view vs GSD (§9).

Evaluation views tile each frame's usable footprint at every GSD bin and go through the
simulated camera pipeline (``camera.pipeline``), the same path the simulator will use. Per split
and bin, at most ``frames_per_bin`` frames (spread evenly) and ``views_per_frame`` tiles per frame
(seeded) are used. The operating threshold tau is the F1-optimal swimmer threshold on the
internal-validation sessions of A, and only there; B and T are then reported at that tau.

Writes ``<out_dir>/eval/metrics.csv`` (split x GSD bin), ``size_recall.csv`` (split x object size),
``eval.json`` and ``eval_vs_gsd.png``. Run after train-detector:

    uv run python scripts/eval_detector.py [--checkpoint PATH] [--device 0] [key=value ...]
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from altiseek.config import load_config
from altiseek.data.frames import Box, Frame
from altiseek.data.images import fetch_images, image_path, load_rgb
from altiseek.data.splits import SplitRow
from altiseek.data.tables import read_csv
from altiseek.detector.config import DetectorConfig
from altiseek.detector.evaluate import ViewResult, best_f1_threshold, size_recall, summarise
from altiseek.detector.infer import Detector
from altiseek.detector.views import (
    CLASSES,
    CameraModel,
    Label,
    frame_camera,
    labels_for,
    tile_windows,
)
from altiseek.geometry.rectify import rectify
from altiseek.registry.run import write_metadata

SWIMMER = CLASSES.index("swimmer")
SPLITS = ("val", "B", "T")

Task = tuple[str, Frame, list[Box], list[float], DetectorConfig, CameraModel]
Rendered = tuple[str, float, int, np.ndarray, tuple[Label, ...]]


def render(task: Task) -> list[Rendered]:
    """Evaluation views of one frame for every GSD bin it was chosen for."""
    split, frame, boxes, gsds, cfg, model = task
    camera = frame_camera(frame, model)
    rgb = load_rgb(image_path(cfg.data.images, frame))
    vs, size = cfg.views, cfg.camera.resolution_px
    out = []
    for gsd in gsds:
        windows = tile_windows(camera, gsd, size, vs.max_off_nadir_deg, vs.min_coverage)
        rng = np.random.default_rng([cfg.eval_views.seed, frame.image_id, round(gsd * 1e4)])
        if len(windows) > cfg.eval_views.views_per_frame:
            keep = rng.choice(len(windows), cfg.eval_views.views_per_frame, replace=False)
            windows = [windows[k] for k in sorted(keep)]
        for window in windows:
            view = rectify(
                rgb,
                camera,
                window,
                max_off_nadir_deg=vs.max_off_nadir_deg,
                pipeline=cfg.camera.pipeline,
                supersample=vs.supersample,
            )
            labels = tuple(labels_for(camera, window, boxes, view.valid))
            out.append((split, gsd, frame.image_id, view.image, labels))
    return out


def plan(frames: Sequence[Frame], bins: Sequence[float], per_bin: int) -> dict[int, list[float]]:
    """Which GSD bins each frame serves: up to ``per_bin`` frames per bin, evenly spaced."""
    ordered = sorted(frames, key=lambda f: f.image_id)
    chosen: defaultdict[int, list[float]] = defaultdict(list)
    for gsd in bins:
        idx = np.linspace(0, len(ordered) - 1, min(per_bin, len(ordered))).round().astype(int)
        for k in sorted(set(idx.tolist())):
            chosen[ordered[k].image_id].append(gsd)
    return chosen


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the detector against GSD.")
    parser.add_argument("--config", type=Path, default=Path("configs/detector/default.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("overrides", nargs="*", help="config overrides, dotted.key=value")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, DetectorConfig, args.overrides)
    data = cfg.data.out_dir
    checkpoint = args.checkpoint or cfg.out_dir / "train" / "weights" / "best.pt"
    detector = Detector(checkpoint, device=args.device)

    frames = read_csv(data / "frames.csv", Frame)
    split_row = {r.image_id: r for r in read_csv(data / "splits.csv", SplitRow)}
    boxes: defaultdict[int, list[Box]] = defaultdict(list)
    for box in read_csv(data / "boxes.csv", Box):
        boxes[box.image_id].append(box)
    model = json.loads((data / "camera_models.json").read_text(encoding="utf-8"))

    def split_of(frame: Frame) -> str | None:
        row = split_row[frame.image_id]
        if frame.camera_group not in cfg.views.camera_groups:
            return None
        if row.split == "A":
            return "val" if row.session_id in cfg.internal_val_sessions else None
        return row.split

    tasks: list[Task] = []
    for split in SPLITS:
        members = [f for f in frames if split_of(f) == split]
        for image_id, gsds in plan(
            members, cfg.eval_views.gsd_bins_mpp, cfg.eval_views.frames_per_bin
        ).items():
            frame = next(f for f in members if f.image_id == image_id)
            m = model[frame.camera_group]
            tasks.append(
                (
                    split,
                    frame,
                    boxes[image_id],
                    gsds,
                    cfg,
                    CameraModel(m["focal_px"], m["pitch_bias_deg"]),
                )
            )
    print(f"{len(tasks)} frames to render; fetching images ...", flush=True)
    fetch_images(cfg.data.images, [t[1] for t in tasks], workers=16)

    results: list[ViewResult] = []
    pending: list[Rendered] = []

    def flush() -> None:
        detections = detector.detect(np.stack([p[3] for p in pending]))
        for (split, gsd, image_id, _, labels), dets in zip(pending, detections, strict=True):
            results.append(ViewResult(split, gsd, image_id, labels, tuple(dets)))
        pending.clear()

    with Pool(args.workers) as pool:
        for i, views in enumerate(pool.imap_unordered(render, tasks, chunksize=1), 1):
            pending.extend(views)
            if len(pending) >= args.batch:
                flush()
            if i % 200 == 0:
                print(f"{i}/{len(tasks)} frames, {len(results)} views", flush=True)
    if pending:
        flush()

    tau = best_f1_threshold([r for r in results if r.split == "val"], SWIMMER)
    out = cfg.out_dir / "eval"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in SPLITS:
        for gsd in cfg.eval_views.gsd_bins_mpp:
            views = [r for r in results if r.split == split and r.gsd_mpp == gsd]
            if views:
                rows.append(
                    {"split": split, "gsd_mpp": gsd, **asdict(summarise(views, SWIMMER, tau))}
                )
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out / "size_recall.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["split", "size_bin", "n_truth", "recall"])
        for split in SPLITS:
            for label, n, recall in size_recall(
                [r for r in results if r.split == split], SWIMMER, tau
            ):
                writer.writerow([split, label, n, recall])
    overall = {
        s: asdict(summarise([r for r in results if r.split == s], SWIMMER, tau)) for s in SPLITS
    }
    record = {"detector_id": detector.detector_id, "tau": tau, "overall": overall}
    (out / "eval.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for split in SPLITS:
        sel = [r for r in rows if r["split"] == split]
        gsd_cm = [100 * r["gsd_mpp"] for r in sel]
        for ax, key in zip(axes, ("recall", "precision", "fp_per_view"), strict=True):
            ax.plot(gsd_cm, [r[key] for r in sel], marker="o", label=split)
    for ax, title in zip(
        axes, ("swimmer recall", "swimmer precision", "false positives per view"), strict=True
    ):
        ax.set_xlabel("GSD (cm/px)")
        ax.set_title(f"{title} at tau = {tau:.2f}", fontsize=9)
        ax.set_xscale("log")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"detector {detector.detector_id}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "eval_vs_gsd.png", dpi=130)
    plt.close(fig)
    write_metadata(
        out / "eval_detector.meta.json",
        stage="eval-detector",
        config=cfg,
        inputs={"checkpoint": checkpoint, "splits": data / "splits.csv"},
        outputs={"metrics": out / "metrics.csv", "size_recall": out / "size_recall.csv"},
        extra={**record, "views": len(results)},
    )
    print(json.dumps(record, indent=2))
    if not math.isfinite(tau):
        raise SystemExit("no threshold could be chosen")


if __name__ == "__main__":
    main()
