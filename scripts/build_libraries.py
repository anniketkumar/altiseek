"""build-libraries: sea backgrounds and object crops, one library per split (CLAUDE.md §11, D26).

For each split of ``splits`` (``conditions.fit_split`` first), every ``frame_stride``-th mavic_4k
frame of each video that has pixels within theta_lib of nadir and no coarser than g0 is
rectified and cut into clean background patches. Each annotated object of ``objects.classes``
becomes a crop with a pixel-based alpha (``altiseek.world.library``). Appearance conditions are
fitted on the first split's patches and assigned to the others'. Per split, writes

    <out_dir>/<split>/backgrounds/<patch_id>.png, objects/<crop_id>.png (RGBA),
    backgrounds.csv, objects.csv, conditions.json, library.meta.json,
    contact_backgrounds.png, contact_objects.png

All of it is dataset imagery and stays out of git (CLAUDE.md §2). The library ID is the first 16
hex digits of the sha256 of the index files (``backgrounds.csv``, ``objects.csv``,
``conditions.json``); each library is registered in the artifact manifest. Run from the
repository root:

    uv run python scripts/build_libraries.py [--config configs/world/library.yaml] [--overwrite]
"""

import argparse
import hashlib
import io
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, replace
from datetime import UTC, datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

from altiseek.config import load_config
from altiseek.data.frames import Box, Frame
from altiseek.data.images import fetch_images, image_path, load_rgb
from altiseek.data.splits import SplitRow
from altiseek.data.tables import read_csv, write_csv
from altiseek.detector.views import CameraModel, footprint_m, frame_camera
from altiseek.geometry.rectify import FILL
from altiseek.registry.manifest import Artifact, register
from altiseek.registry.run import git_state, sha256_file, write_metadata
from altiseek.world.appearance import FEATURES, Conditions, fit_conditions, stats
from altiseek.world.config import LibraryConfig
from altiseek.world.library import (
    BackgroundRow,
    ObjectRow,
    background_patches,
    box_centres_m,
    object_crop,
    select_frames,
)

INDEX_FILES = ("backgrounds.csv", "objects.csv", "conditions.json")
Task = tuple[Frame, list[Box], SplitRow, LibraryConfig, CameraModel, Path]


def r4(value: float) -> float:
    """Statistics are stored to 4 decimals, so float noise does not change the library ID."""
    return round(float(value), 4)


def save_png(path: Path, array: np.ndarray) -> str:
    """Write ``array`` as PNG and return the sha256 of the file."""
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    path.write_bytes(buffer.getvalue())
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def process(task: Task) -> tuple[int, list[BackgroundRow], list[ObjectRow], Counter[str]]:
    """One frame's patches and crops, written under ``out``; their rows and the crop outcomes
    (``<category>: <reason>``)."""
    frame, boxes, row, cfg, model, out = task
    camera = frame_camera(frame, model)
    rgb = load_rgb(image_path(cfg.data.images, frame))
    geometry = {
        "gsd_mpp": cfg.gsd_mpp,
        "max_off_nadir_deg": cfg.max_off_nadir_deg,
        "supersample": cfg.supersample,
    }
    video = frame.source_folder or ""
    patches = []
    for name, _, image in background_patches(
        rgb, camera, boxes, **geometry, settings=cfg.background
    ):
        patch_id = f"{frame.image_id}-{name}"
        sha = save_png(out / "backgrounds" / f"{patch_id}.png", image)
        s = stats(image)
        patches.append(
            BackgroundRow(
                patch_id,
                frame.image_id,
                row.session_id,
                video,
                row.split,
                sha,
                *(r4(v) for v in (*s.mean, *s.std, s.texture)),
                condition=-1,
            )
        )
    crops, outcomes = [], Counter()
    centres = box_centres_m(camera, boxes)
    for i, box in enumerate(boxes):
        if box.category not in cfg.objects.classes:
            continue
        others = np.delete(centres, i, axis=0)
        crop, reason = object_crop(rgb, camera, box, others, **geometry, settings=cfg.objects)
        outcomes[f"{box.category}: {reason}"] += 1
        if crop is None:
            continue
        crop_id = f"{frame.image_id}-{box.annotation_id}"
        sha = save_png(out / "objects" / f"{crop_id}.png", crop.rgba)
        crops.append(
            ObjectRow(
                crop_id,
                box.category,
                frame.image_id,
                box.annotation_id,
                row.session_id,
                video,
                row.split,
                sha,
                crop.rgba.shape[0],
                *(r4(v) for v in (*crop.centre_px, *crop.size_m)),
                *(r4(v) for v in (*crop.water.mean, *crop.water.std)),
            )
        )
    return frame.image_id, patches, crops, outcomes


def condition_summary(patches: Sequence[BackgroundRow], k: int) -> list[dict[str, float]]:
    """Per condition: patch count and the mean of the patches' Lab means, Lab spreads and
    texture (the renderer's harmonisation target)."""
    names = ("l_mean", "a_mean", "b_mean", "l_std", "a_std", "b_std", "texture")
    out: list[dict[str, float]] = []
    for c in range(k):
        members = [p for p in patches if p.condition == c]
        row: dict[str, float] = {"count": len(members)}
        if members:
            row |= {n: r4(np.mean([getattr(p, n) for p in members])) for n in names}
        out.append(row)
    return out


def contact_sheet(tiles: Sequence[np.ndarray], cols: int, cell_px: int) -> np.ndarray:
    """Images (RGB, or RGBA composited over letterbox grey) in a grid of ``cell_px`` cells."""
    rows = max(1, math.ceil(len(tiles) / cols))
    sheet = np.full((rows * cell_px, cols * cell_px, 3), 255, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        if tile.shape[2] == 4:
            alpha = tile[..., 3:].astype(np.float32) / 255
            tile = np.rint(tile[..., :3] * alpha + np.array(FILL) * (1 - alpha)).astype(np.uint8)
        image = Image.fromarray(tile)
        image.thumbnail((cell_px, cell_px), Image.Resampling.LANCZOS)
        r, c = divmod(i, cols)
        sheet[r * cell_px : r * cell_px + image.height, c * cell_px : c * cell_px + image.width] = (
            np.asarray(image)
        )
    return sheet


def write_contact_sheets(out: Path, patches: list[BackgroundRow], crops: list[ObjectRow]) -> None:
    rng = np.random.default_rng(0)
    tiles = []
    for c in sorted({p.condition for p in patches}):
        members = [p for p in patches if p.condition == c]
        for k in sorted(rng.choice(len(members), min(8, len(members)), replace=False)):
            tiles.append(np.asarray(Image.open(out / "backgrounds" / f"{members[k].patch_id}.png")))
        tiles += [np.full((8, 8, 3), 255, dtype=np.uint8)] * (-len(tiles) % 8)
    Image.fromarray(contact_sheet(tiles, 8, 128)).save(out / "contact_backgrounds.png")
    tiles = []
    for category in sorted({c.category for c in crops}):
        members = [c for c in crops if c.category == category]
        for k in sorted(rng.choice(len(members), min(6, len(members)), replace=False)):
            rgba = np.asarray(Image.open(out / "objects" / f"{members[k].crop_id}.png"))
            tiles += [rgba[..., :3], rgba]  # the window as cut, then what gets pasted
        tiles += [np.full((8, 8, 3), 255, dtype=np.uint8)] * (-len(tiles) % 12)
    Image.fromarray(contact_sheet(tiles, 12, 96)).save(out / "contact_objects.png")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the world libraries.")
    parser.add_argument("--config", type=Path, default=Path("configs/world/library.yaml"))
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--manifest", type=Path, default=Path("registry/manifest.json"))
    parser.add_argument("--overwrite", action="store_true", help="replace existing libraries")
    parser.add_argument("overrides", nargs="*", help="config overrides, dotted.key=value")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, LibraryConfig, args.overrides)
    if cfg.splits[0] != cfg.conditions.fit_split:
        raise SystemExit(
            f"splits must start with conditions.fit_split ({cfg.conditions.fit_split})"
        )
    data = cfg.data.out_dir
    frames = read_csv(data / "frames.csv", Frame)
    split_rows = {r.image_id: r for r in read_csv(data / "splits.csv", SplitRow)}
    boxes: defaultdict[int, list[Box]] = defaultdict(list)
    for box in read_csv(data / "boxes.csv", Box):
        boxes[box.image_id].append(box)
    models = json.loads((data / "camera_models.json").read_text(encoding="utf-8"))
    split_id = json.loads((data / "make_splits.meta.json").read_text(encoding="utf-8"))["split_id"]
    parents = {
        "split_id": split_id,
        "camera_models_sha256": sha256_file(data / "camera_models.json")[:16],
        "code_commit": (git_state() or {}).get("commit", ""),
    }

    conditions: Conditions | None = None
    fitted_library = ""
    for split in cfg.splits:
        out = cfg.out_dir / split
        if out.exists():
            if not args.overwrite:
                raise SystemExit(f"{out} exists; pass --overwrite to rebuild it")
            shutil.rmtree(out)
        (out / "backgrounds").mkdir(parents=True)
        (out / "objects").mkdir()
        chosen = []
        for frame in select_frames(frames, split_rows, split, cfg.camera_groups, cfg.frame_stride):
            m = models[frame.camera_group]
            model = CameraModel(m["focal_px"], m["pitch_bias_deg"])
            camera = frame_camera(frame, model)
            if len(footprint_m(camera, cfg.gsd_mpp, cfg.max_off_nadir_deg)):
                chosen.append((frame, model))
        print(f"{split}: {len(chosen)} frames; fetching images ...", flush=True)
        fetch_images(cfg.data.images, [f for f, _ in chosen], workers=16)
        tasks: list[Task] = [
            (f, boxes[f.image_id], split_rows[f.image_id], cfg, model, out) for f, model in chosen
        ]
        results = []
        with Pool(args.workers) as pool:
            for i, result in enumerate(pool.imap_unordered(process, tasks, chunksize=1), 1):
                results.append(result)
                if i % 50 == 0:
                    print(f"{split}: {i}/{len(tasks)} frames", flush=True)
        results.sort(key=lambda r: r[0])
        patches = [p for r in results for p in r[1]]
        crops = [c for r in results for c in r[2]]
        outcomes = sum((r[3] for r in results), Counter())

        features = np.array([[getattr(p, name) for name in FEATURES] for p in patches])
        if conditions is None:
            conditions = fit_conditions(features, cfg.conditions.clusters, cfg.conditions.seed)
        labels = conditions.assign(features)
        patches = [replace(p, condition=int(c)) for p, c in zip(patches, labels, strict=True)]
        write_csv(out / "backgrounds.csv", BackgroundRow, patches)
        write_csv(out / "objects.csv", ObjectRow, crops)
        summary = condition_summary(patches, cfg.conditions.clusters)
        record = {
            "fitted_on": cfg.conditions.fit_split,
            "fitted_library": fitted_library,
            "features": list(FEATURES),
            **asdict(conditions),
            "conditions": summary,
        }
        (out / "conditions.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        digest = hashlib.sha256()
        for name in INDEX_FILES:
            digest.update((out / name).read_bytes())
        library_id = digest.hexdigest()[:16]
        if split == cfg.conditions.fit_split:
            fitted_library = library_id
        write_contact_sheets(out, patches, crops)

        counts = {
            "frames": len(chosen),
            "videos": len({f.source_folder for f, _ in chosen}),
            "backgrounds": len(patches),
            "backgrounds_per_condition": [s["count"] for s in summary],
            "objects": dict(sorted(Counter(c.category for c in crops).items())),
            "crop_outcomes": dict(sorted(outcomes.items())),
        }
        write_metadata(
            out / "library.meta.json",
            stage="build-libraries",
            config=cfg,
            inputs={
                "frames": data / "frames.csv",
                "boxes": data / "boxes.csv",
                "splits": data / "splits.csv",
                "camera_models": data / "camera_models.json",
            },
            outputs={name: out / name for name in INDEX_FILES},
            extra={"library_id": library_id, "split": split, **parents, **counts},
        )
        lineage = dict(parents)
        if split != cfg.conditions.fit_split:
            lineage["conditions_library"] = fitted_library
        register(
            args.manifest,
            Artifact(
                kind="library",
                artifact_id=library_id,
                sha256=digest.hexdigest(),
                location=f"local:{out.as_posix()}",
                parents=lineage,
                created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
                notes={"split": split, **counts},
            ),
        )
        print(json.dumps({"split": split, "library_id": library_id, **counts}, indent=2))


if __name__ == "__main__":
    main()
