"""make-views: render the detector's training and internal-validation views (CLAUDE.md §9).

Train views come from split A minus the internal-validation sessions; val views from those
sessions. Each frame gets ``gsds_per_frame`` target GSDs, drawn log-uniformly from the part of
``gsd_range_mpp`` the frame can serve, and ``views_per_gsd`` random windows per GSD. Each view gets
a random quarter turn, a random blur, and is written as JPEG at a random quality: the write is the
camera pipeline's JPEG step, so no image is compressed twice. Output:

    <views_dir>/yolo/images/{train,val}/<view>.jpg, <views_dir>/yolo/labels/{train,val}/<view>.txt,
    <views_dir>/yolo/data.yaml, <views_dir>/views.csv

Resumable: frames listed in done_frames.txt are skipped. Run from the repository root:

    uv run python scripts/make_views.py [--config configs/detector/default.yaml] [key=value ...]
"""

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, fields
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from altiseek.config import load_config
from altiseek.data.frames import Box, Frame
from altiseek.data.images import fetch_images, image_path, load_rgb
from altiseek.data.splits import SplitRow
from altiseek.data.tables import read_csv
from altiseek.detector.config import DetectorConfig
from altiseek.detector.views import (
    CLASSES,
    CameraModel,
    finest_usable_gsd_mpp,
    frame_camera,
    labels_for,
    object_centres_m,
    random_windows,
    rotate_quarter,
    yolo_lines,
)
from altiseek.geometry.rectify import rectify
from altiseek.imaging.pipeline import CameraPipeline
from altiseek.registry.run import write_metadata


@dataclass(frozen=True)
class ViewRecord:
    """One rendered view, a row of ``views.csv``."""

    view_id: str
    image_id: int
    subset: str
    gsd_mpp: float
    x0_m: float
    y0_m: float
    turns: int
    blur_sigma_px: float
    jpeg_quality: int
    valid_share: float
    n_labels: int
    n_swimmer: int


Task = tuple[Frame, list[Box], str, DetectorConfig, CameraModel, Path]


def render_frame(task: Task) -> tuple[int, list[ViewRecord]]:
    """The frame's id and all its views. Seeded per frame, so results do not depend on worker
    order."""
    frame, boxes, subset, cfg, model, yolo_dir = task
    tv, vs = cfg.train_views, cfg.views
    size = cfg.camera.resolution_px
    rng = np.random.default_rng([tv.seed, frame.image_id])
    camera = frame_camera(frame, model)
    lo = max(tv.gsd_range_mpp[0], finest_usable_gsd_mpp(camera, vs.max_off_nadir_deg))
    hi = tv.gsd_range_mpp[1]
    if lo >= hi:
        return frame.image_id, []
    rgb = load_rgb(image_path(cfg.data.images, frame))
    centres = object_centres_m(camera, boxes)
    records = []
    for g in range(tv.gsds_per_frame):
        gsd = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
        windows = random_windows(
            camera,
            gsd,
            size,
            vs.max_off_nadir_deg,
            vs.min_coverage,
            centres,
            tv.views_per_gsd,
            tv.object_share,
            rng,
        )
        for w, window in enumerate(windows):
            blur = float(rng.uniform(*tv.blur_sigma_px))
            quality = int(rng.integers(tv.jpeg_quality[0], tv.jpeg_quality[1] + 1))
            view = rectify(
                rgb,
                camera,
                window,
                max_off_nadir_deg=vs.max_off_nadir_deg,
                pipeline=CameraPipeline(blur_sigma_px=blur),
                supersample=vs.supersample,
            )
            labels = labels_for(camera, window, boxes, view.valid)
            turns = int(rng.integers(4))
            image, labels = rotate_quarter(view.image, labels, turns)
            view_id = f"{frame.image_id}_{g}_{w}"
            Image.fromarray(image).save(
                yolo_dir / "images" / subset / f"{view_id}.jpg", quality=quality
            )
            (yolo_dir / "labels" / subset / f"{view_id}.txt").write_text(
                "".join(line + "\n" for line in yolo_lines(labels, size)), encoding="utf-8"
            )
            records.append(
                ViewRecord(
                    view_id,
                    frame.image_id,
                    subset,
                    gsd,
                    window.x0_m,
                    window.y0_m,
                    turns,
                    blur,
                    quality,
                    float(view.valid.mean()),
                    len(labels),
                    sum(lab.class_id == CLASSES.index("swimmer") for lab in labels),
                )
            )
    return frame.image_id, records


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render training and validation views.")
    parser.add_argument("--config", type=Path, default=Path("configs/detector/default.yaml"))
    parser.add_argument("--limit", type=int, default=None, help="first N frames only (smoke test)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("overrides", nargs="*", help="config overrides, dotted.key=value")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, DetectorConfig, args.overrides)
    data_dir = cfg.data.out_dir

    frames = read_csv(data_dir / "frames.csv", Frame)
    splits = {r.image_id: r for r in read_csv(data_dir / "splits.csv", SplitRow)}
    boxes: defaultdict[int, list[Box]] = defaultdict(list)
    for box in read_csv(data_dir / "boxes.csv", Box):
        boxes[box.image_id].append(box)
    models = json.loads((data_dir / "camera_models.json").read_text(encoding="utf-8"))

    def subset(frame: Frame) -> str | None:
        row = splits[frame.image_id]
        if row.split != "A" or frame.camera_group not in cfg.views.camera_groups:
            return None
        return "val" if row.session_id in cfg.internal_val_sessions else "train"

    chosen = [(f, s) for f in sorted(frames, key=lambda f: f.image_id) if (s := subset(f))]
    if args.limit is not None:
        chosen = chosen[: args.limit]

    yolo_dir = cfg.views_dir / "yolo"
    for kind in ("images", "labels"):
        for name in ("train", "val"):
            (yolo_dir / kind / name).mkdir(parents=True, exist_ok=True)
    views_csv = cfg.views_dir / "views.csv"
    done_txt = cfg.views_dir / "done_frames.txt"
    names = [f.name for f in fields(ViewRecord)]
    done_frames = set()
    if done_txt.exists():
        done_frames = {int(x) for x in done_txt.read_text(encoding="utf-8").split()}
    todo = [(f, s) for f, s in chosen if f.image_id not in done_frames]
    print(f"{len(chosen)} frames ({len(todo)} to render); fetching images ...", flush=True)
    fetch_images(cfg.data.images, [f for f, _ in todo], workers=16)

    tasks: list[Task] = []
    for frame, name in todo:
        model = models[frame.camera_group]
        camera_model = CameraModel(model["focal_px"], model["pitch_bias_deg"])
        tasks.append((frame, boxes[frame.image_id], name, cfg, camera_model, yolo_dir))
    new_file = not views_csv.exists()
    with (
        views_csv.open("a", newline="", encoding="utf-8") as fh,
        done_txt.open("a", encoding="utf-8") as done,
        Pool(args.workers) as pool,
    ):
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(names)
        results = pool.imap_unordered(render_frame, tasks, chunksize=2)
        for i, (image_id, records) in enumerate(results, 1):
            writer.writerows([getattr(r, n) for n in names] for r in records)
            fh.flush()
            done.write(f"{image_id}\n")  # after its views: the frame is complete
            done.flush()
            if i % 100 == 0:
                print(f"{i}/{len(tasks)} frames", flush=True)

    (yolo_dir / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(yolo_dir.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": dict(enumerate(CLASSES)),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    with views_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    summary = {
        "frames": len(chosen),
        "views": dict(Counter(r["subset"] for r in rows)),
        "swimmer_labels": sum(int(r["n_swimmer"]) for r in rows),
        "views_per_gsd": dict(sorted(Counter(f"{float(r['gsd_mpp']):.2f}" for r in rows).items())),
    }
    write_metadata(
        cfg.views_dir / "make_views.meta.json",
        stage="make-views",
        config=cfg,
        inputs={
            "frames": data_dir / "frames.csv",
            "splits": data_dir / "splits.csv",
            "boxes": data_dir / "boxes.csv",
            "camera_models": data_dir / "camera_models.json",
        },
        outputs={"views": views_csv, "data_yaml": yolo_dir / "data.yaml"},
        extra={"summary": summary},
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
