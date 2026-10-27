"""Resumable CPU-only real-T calibration check under docs/sensor-protocol.md."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from altiseek.cache.config import CacheConfig
from altiseek.cache.jobs import write_json
from altiseek.calibration.fit import calibration_report, measurements
from altiseek.calibration.model import SensorModel, digest
from altiseek.calibration.outcomes import WorldOutcomes
from altiseek.calibration.real import real_outcomes
from altiseek.config import load_config
from altiseek.data.frames import Box, Frame
from altiseek.data.images import fetch_images, image_path, load_rgb
from altiseek.data.splits import SplitRow
from altiseek.data.tables import read_csv
from altiseek.detector.config import DetectorConfig
from altiseek.detector.infer import Detector, detector_id_of
from altiseek.detector.views import CameraModel, frame_camera, labels_for, tile_windows
from altiseek.geometry.rectify import rectify
from altiseek.registry.run import sha256_file, write_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("data/kaggle/altiseek-eval-levels/eval-levels/eval/views.csv"),
    )
    parser.add_argument("--out", type=Path, default=Path("data/sensor_models/real-T"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    cache = load_config("configs/cache/default.yaml", CacheConfig)
    cfg = load_config("configs/detector/default.yaml", DetectorConfig)
    model = SensorModel.load(
        args.model,
        detector_id=cache.detector_id,
        fit_world_set_id=cache.world_set_ids["sensor-fit"],
    )
    if detector_id_of(cache.checkpoint) != model.detector_id:
        raise ValueError("checkpoint mismatch")
    expected_contract = model.cache_contract
    if (
        expected_contract["render"] != cache.render.model_dump(mode="json")
        or expected_contract["inference"] != cache.inference.model_dump(mode="json")
        or cfg.camera != cache.render.camera
    ):
        raise ValueError("real inference/camera differs from fitted model")
    table = cache.render.levels
    gsds = [table.footprint_m(lvl) / cfg.camera.resolution_px for lvl in table.levels]
    # Deliberately discard every historical outcome column before constructing the plan.
    with args.selection.open(encoding="utf-8", newline="") as fh:
        selection = [
            {k: r[k] for k in ("image_id", "gsd_mpp", "x0_m", "y0_m")}
            for r in csv.DictReader(fh)
            if r["split"] == "T"
        ]
    selection.sort(
        key=lambda r: (int(r["image_id"]), float(r["gsd_mpp"]), float(r["y0_m"]), float(r["x0_m"]))
    )
    if not selection or len({digest(r) for r in selection}) != len(selection):
        raise ValueError("empty or duplicate real evaluation plan")
    data = cfg.data.out_dir
    frames = {f.image_id: f for f in read_csv(data / "frames.csv", Frame)}
    splits = {s.image_id: s.split for s in read_csv(data / "splits.csv", SplitRow)}
    if any(splits[int(r["image_id"])] != "T" for r in selection):
        raise ValueError("real selection contains a non-T frame")
    boxes: defaultdict[int, list[Box]] = defaultdict(list)
    for box in read_csv(data / "boxes.csv", Box):
        boxes[box.image_id].append(box)
    cameras = json.loads((data / "camera_models.json").read_text(encoding="utf-8"))
    identity = {
        "model_sha256": sha256_file(args.model),
        "selection": selection,
        "detector_config": cfg.model_dump(mode="json"),
        "inputs": {
            name: sha256_file(data / name)
            for name in ("frames.csv", "boxes.csv", "splits.csv", "camera_models.json")
        },
    }
    run_id = digest(identity)
    root = args.out / run_id
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "plan.json", identity)
    selected_frames = sorted({int(r["image_id"]) for r in selection})
    print(
        f"Real T: {len(selection)} views, {len(selected_frames)} frames; run {run_id}", flush=True
    )
    fetch_images(cfg.data.images, [frames[i] for i in selected_frames], workers=8)
    detector = Detector(
        cache.checkpoint,
        device="cpu",
        score_min=model.fit_config.tau_min,
        iou=cache.inference.iou,
        max_det=cache.inference.max_det,
    )
    rows = []
    current_id, rgb, windows = None, None, {}
    for index, item in enumerate(selection):
        image_id, gsd = int(item["image_id"]), float(item["gsd_mpp"])
        li = next((i for i, g in enumerate(gsds) if abs(g - gsd) < 1e-10), None)
        if li is None:
            raise ValueError("selection has a GSD outside the frozen levels")
        frame = frames[image_id]
        source = image_path(cfg.data.images, frame)
        source_sha = sha256_file(source)
        path = root / f"{digest(item)}.json"
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
            if (
                row["run_id"] != run_id
                or row["selection"] != item
                or row["source_sha256"] != source_sha
            ):
                raise ValueError("real evaluation resume mismatch")
        else:
            camera_values = cameras[frame.camera_group]
            camera = frame_camera(
                frame, CameraModel(camera_values["focal_px"], camera_values["pitch_bias_deg"])
            )
            if image_id != current_id:
                current_id, rgb, windows = image_id, load_rgb(source), {}
            if gsd not in windows:
                windows[gsd] = tile_windows(
                    camera,
                    gsd,
                    cfg.camera.resolution_px,
                    cfg.views.max_off_nadir_deg,
                    cfg.eval_views.min_coverage,
                )
            chosen = [
                w
                for w in windows[gsd]
                if f"{w.x0_m:.3f}" == item["x0_m"] and f"{w.y0_m:.3f}" == item["y0_m"]
            ]
            if len(chosen) != 1:
                raise ValueError("cannot reconstruct unique exact evaluation window")
            window = chosen[0]
            view = rectify(
                rgb,
                camera,
                window,
                max_off_nadir_deg=cfg.views.max_off_nadir_deg,
                pipeline=cfg.camera.pipeline,
                supersample=cfg.views.supersample,
            )
            truth = labels_for(camera, window, boxes[image_id], view.valid)
            detections = detector.detect(view.image[None])[0]
            row = real_outcomes(
                window, view.valid, truth, detections, table.cell_m, model.fit_config
            )
            row.update(run_id=run_id, selection=item, source_sha256=source_sha, level=li)
            write_json(path, row)
        rows.append(row)
        if (index + 1) % 10 == 0:
            print(f"{index + 1}/{len(selection)} views", flush=True)
    groups = []
    for image_id in selected_frames:
        members = [r for r in rows if int(r["selection"]["image_id"]) == image_id]
        counts = np.zeros((len(gsds), 2, 4), dtype=np.int64)
        levels, scores, labels = [], [], []
        for row in members:
            counts[row["level"]] += np.asarray(row["counts"])
            scores.extend(row["scores"])
            labels.extend(row["labels"])
            levels.extend([row["level"]] * len(row["scores"]))
        groups.append(
            WorldOutcomes(
                str(image_id),
                counts,
                np.zeros((len(gsds), 2, 2), dtype=int),
                np.asarray(levels),
                np.asarray(scores),
                np.asarray(labels),
            )
        )
    report = measurements(groups, model.fit_config)
    report["frames"] = report.pop("worlds")
    for level in report["levels"]:
        for threshold in ("binary", "soft"):
            del level[threshold]["conditional_recall"]
    report.update(
        sensor_model_id=model.sensor_model_id,
        run_id=run_id,
        views=len(rows),
        calibration=calibration_report(groups, model),
    )
    output = root / "report.json"
    write_json(output, report)
    write_metadata(
        root / "evaluate.meta.json",
        stage="evaluate-sensor-real",
        config=cfg,
        inputs={"model": args.model, "selection": args.selection, "plan": root / "plan.json"},
        outputs={"report": output},
        extra={"run_id": run_id, "device": "cpu"},
    )
    print(f"Completed: {output}", flush=True)


if __name__ == "__main__":
    main()
