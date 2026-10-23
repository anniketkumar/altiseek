"""fit-camera: focal length per camera group from the open-sea horizon (CLAUDE.md §7 step 4).

Run from the repository root after prepare-data. Missing images are fetched from the dataset's
WebDAV share (about 0.7 MB each for the compressed version):

    uv run python scripts/fit_camera.py [--config configs/data/default.yaml] [key=value ...]
"""

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from altiseek.config import load_config
from altiseek.data.camera_fit import (
    FocalFit,
    HorizonSample,
    fit_focal,
    spread_per_video,
    steady_frames,
)
from altiseek.data.config import DataConfig, HorizonFitConfig
from altiseek.data.frames import Frame
from altiseek.data.horizon import detect_horizon
from altiseek.data.images import fetch_images, image_path, load_rgb
from altiseek.data.tables import read_csv, write_csv
from altiseek.geometry.camera import hfov_deg
from altiseek.registry.run import write_metadata


@dataclass(frozen=True)
class HorizonRecord:
    """One examined frame: the detected horizon and whether it entered the fit."""

    image_id: int
    camera_group: str
    source_folder: str | None
    gimbal_pitch_deg: float | None
    altitude_m: float | None
    row_px: float | None
    roll_deg: float | None
    n_inliers: int
    inlier_fraction: float | None
    offset_px: float | None
    accepted: bool


def select(frames: Sequence[Frame], cfg: HorizonFitConfig) -> list[Frame]:
    """Steady-pitch open-sea frames in the pitch range, spread over each video."""
    site = [
        f
        for f in frames
        if f.camera_group in cfg.camera_groups
        and f.latitude_deg is not None
        and cfg.latitude_deg[0] <= f.latitude_deg <= cfg.latitude_deg[1]
        and f.altitude_m is not None
    ]
    steady = steady_frames(site, cfg.steady_window_s, cfg.steady_tol_deg)
    in_range = [
        f
        for f in steady
        if f.gimbal_pitch_deg is not None
        and cfg.pitch_deg[0] <= f.gimbal_pitch_deg <= cfg.pitch_deg[1]
    ]
    return spread_per_video(in_range, cfg.frames_per_video)


def plot(path: Path, samples: Sequence[HorizonSample], fit: FocalFit, group: str) -> None:
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    videos = sorted({s.video for s in samples}, key=lambda v: v[1] or "")
    x = np.array([s.tan_angle for s in samples])
    intercept = fit.focal_px * np.radians(fit.pitch_bias_deg)
    for video in videos:
        sel = [s for s in samples if s.video == video]
        xs = np.array([s.tan_angle for s in sel])
        ys = np.array([s.offset_px for s in sel])
        top.scatter(xs, ys, s=14, label=video[1])
        bottom.scatter(xs, ys - (intercept + fit.focal_px * xs), s=14)
    grid = np.linspace(0, x.max() * 1.05, 50)
    top.plot(grid, intercept + fit.focal_px * grid, color="black", lw=1)
    top.set_ylabel("horizon above centre (px)")
    top.set_title(
        f"{group}: f = {fit.focal_px:.0f} px "
        f"[{fit.focal_ci95_px[0]:.0f}, {fit.focal_ci95_px[1]:.0f}], "
        f"pitch bias {fit.pitch_bias_deg:+.2f} deg, n = {fit.n_frames} ({fit.n_videos} videos)",
        fontsize=9,
    )
    top.legend(fontsize=6, ncol=2)
    bottom.axhline(0, color="black", lw=1)
    bottom.set_xlabel("tan(logged pitch - horizon dip)")
    bottom.set_ylabel("residual (px)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit focal lengths from the sea horizon.")
    parser.add_argument("--config", type=Path, default=Path("configs/data/default.yaml"))
    parser.add_argument("overrides", nargs="*", help="config overrides, dotted.key=value")
    args = parser.parse_args(argv)
    config = load_config(args.config, DataConfig, args.overrides)
    cfg = config.horizon_fit

    frames_csv = config.out_dir / "frames.csv"
    chosen = select(read_csv(frames_csv, Frame), cfg)
    print(f"{len(chosen)} frames selected; fetched {fetch_images(config.images, chosen)} images")

    records: list[HorizonRecord] = []
    samples: list[HorizonSample] = []
    for frame in chosen:
        horizon = detect_horizon(load_rgb(image_path(config.images, frame)))
        offset = None if horizon is None else (frame.height_px - 1) / 2 - horizon.row_px
        accepted = (
            horizon is not None
            and horizon.n_inliers >= cfg.min_inlier_strips
            and horizon.inlier_fraction >= cfg.min_inlier_fraction
            and abs(horizon.roll_deg) <= cfg.max_roll_deg
        )
        records.append(
            HorizonRecord(
                frame.image_id,
                frame.camera_group,
                frame.source_folder,
                frame.gimbal_pitch_deg,
                frame.altitude_m,
                None if horizon is None else horizon.row_px,
                None if horizon is None else horizon.roll_deg,
                0 if horizon is None else horizon.n_inliers,
                None if horizon is None else horizon.inlier_fraction,
                offset,
                accepted,
            )
        )
        if accepted and offset is not None:
            samples.append(HorizonSample(frame, offset))

    models = {}
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    for group in cfg.camera_groups:
        group_samples = [s for s in samples if s.frame.camera_group == group]
        fit = fit_focal(group_samples, cfg.bootstrap_reps, cfg.seed)
        width = group_samples[0].frame.width_px
        per_video = {}
        for video in sorted({s.video[1] or "" for s in group_samples}):
            sel = [s for s in group_samples if (s.video[1] or "") == video]
            # Zero-bias focal length of each frame; a mode change would split the videos.
            focal = [s.offset_px / s.tan_angle for s in sel]
            per_video[video] = {"n": len(sel), "median_focal_px": float(np.median(focal))}
        models[group] = {
            **asdict(fit),
            "width_px": width,
            "hfov_deg": hfov_deg(fit.focal_px, width),
            "hfov_ci95_deg": [hfov_deg(f, width) for f in reversed(fit.focal_ci95_px)],
            "nadir_gsd_per_altitude": 1.0 / fit.focal_px,
            "per_video": per_video,
            "method": "open-sea horizon, Theil-Sen, video-clustered bootstrap",
        }
        plot(config.figures_dir / f"horizon_fit_{group}.png", group_samples, fit, group)

    outputs = {
        "camera_models": config.out_dir / "camera_models.json",
        "horizon_samples": config.out_dir / "horizon_samples.csv",
    }
    outputs["camera_models"].write_text(json.dumps(models, indent=2) + "\n", encoding="utf-8")
    write_csv(outputs["horizon_samples"], HorizonRecord, records)
    counts = Counter("accepted" if r.accepted else "rejected" for r in records)
    write_metadata(
        config.out_dir / "fit_camera.meta.json",
        stage="fit-camera",
        config=config,
        inputs={"frames": frames_csv},
        outputs=outputs,
        extra={"frames_examined": dict(counts)},
    )
    print(dict(counts))
    print(json.dumps(models, indent=2))


if __name__ == "__main__":
    main()
