"""train-detector: fine-tune Ultralytics YOLO on the rendered views (CLAUDE.md §9, D19).

Reads ``<views_dir>/yolo/data.yaml`` (make-views) and trains into ``<out_dir>/train``. Resumable:
if ``<out_dir>/train/weights/last.pt`` exists, or ``--resume-from`` holds a previous run (e.g. an
attached Kaggle kernel output), training continues from it. On success it writes
``<out_dir>/detector.json`` with the best checkpoint's ``detector_id`` and a metadata record.
Scale and mosaic augmentation stay off: views already fix the GSD, and rescaling would change it.

    uv run python scripts/train_detector.py [--config configs/detector/default.yaml] [key=value ...]
"""

import argparse
import json
import shutil
from collections.abc import Sequence
from pathlib import Path

from altiseek.config import load_config
from altiseek.detector.config import DetectorConfig
from altiseek.detector.infer import detector_id_of
from altiseek.detector.views import CLASSES
from altiseek.registry.run import write_metadata


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the detector on rendered views.")
    parser.add_argument("--config", type=Path, default=Path("configs/detector/default.yaml"))
    parser.add_argument("--resume-from", type=Path, default=None, help="a previous out_dir")
    parser.add_argument("--device", default="0", help="CUDA ids, e.g. 0 or 0,1; or cpu")
    parser.add_argument("overrides", nargs="*", help="config overrides, dotted.key=value")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, DetectorConfig, args.overrides)

    from ultralytics import YOLO, settings

    settings.update({"sync": False})
    run_dir = cfg.out_dir / "train"
    if args.resume_from is not None and not run_dir.exists():
        shutil.copytree(args.resume_from / "train", run_dir)
    last = run_dir / "weights" / "last.pt"
    m = cfg.model

    def fit(device: str) -> None:
        if last.exists():
            print(f"resuming from {last} on device {device}", flush=True)
            YOLO(str(last)).train(resume=True, device=device)
            return
        YOLO(m.weights).train(
            data=str(cfg.views_dir / "yolo" / "data.yaml"),
            epochs=m.epochs,
            imgsz=cfg.camera.resolution_px,
            batch=m.batch,  # the total over all devices
            patience=m.patience,
            workers=m.workers,
            device=device,
            project=str(cfg.out_dir.resolve()),
            name="train",
            exist_ok=True,
            seed=m.seed,
            deterministic=True,
            mosaic=m.mosaic,
            close_mosaic=0,
            scale=0.0,
            degrees=m.degrees,
            fliplr=m.fliplr,
            flipud=m.flipud,
            hsv_h=m.hsv_h,
            hsv_s=m.hsv_s,
            hsv_v=m.hsv_v,
            copy_paste=0.0,
            save_period=5,
            plots=True,
        )

    try:
        fit(args.device)
    except Exception as err:
        # Multi-GPU (DDP) training runs in a subprocess; if it fails, continue on one GPU,
        # resuming from its last checkpoint when there is one.
        if "," not in args.device:
            raise
        print(f"multi-GPU training failed ({err!r}); continuing on GPU 0", flush=True)
        fit("0")

    best = run_dir / "weights" / "best.pt"
    record = {
        "detector_id": detector_id_of(best),
        "checkpoint": "train/weights/best.pt",
        "classes": list(CLASSES),
        "imgsz": cfg.camera.resolution_px,
        "init_weights": m.weights,
    }
    (cfg.out_dir / "detector.json").write_text(json.dumps(record, indent=2) + "\n", "utf-8")
    write_metadata(
        cfg.out_dir / "train_detector.meta.json",
        stage="train-detector",
        config=cfg,
        inputs={"data_yaml": cfg.views_dir / "yolo" / "data.yaml"},
        outputs={"best": best, "last": last},
        extra=record,
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
