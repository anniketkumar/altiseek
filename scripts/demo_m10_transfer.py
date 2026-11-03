"""Predetermined local M10 demo video (D56); never committed, it shows dataset imagery.

The case is fixed in the frozen protocol (first selected T world, B5, original
sensor) before any outcome is seen. Left: top-down exported world with belief
overlay, path coloured by level, current footprint and true targets. Inset: the
live drone camera with its swimmer detections. Writes data/m10/demo/<protocol>/
and a numeric manifest (no imagery) to docs/m10-demo.json.
"""

import argparse
import json
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

from altiseek.cache.config import CacheConfig
from altiseek.calibration.model import digest
from altiseek.config import load_config
from altiseek.deploy.transfer import load_protocol
from altiseek.registry.run import sha256_file
from altiseek.world.spec import WorldSpec

MAP_PX = 720
INSET_PX = 360
FPS = 4
HOLD_FRAMES = 4
LEVEL_COLOURS = ((80, 220, 80), (60, 200, 230), (40, 160, 255), (60, 80, 255), (200, 80, 220))


def top_down(export: Path) -> np.ndarray:
    """Exported Gazebo ground texture, verified against its export checksums."""
    metadata = json.loads((export / "export.json").read_text())
    height, width, _ = metadata["shape"]
    scale = MAP_PX / width
    canvas = np.zeros((MAP_PX, MAP_PX, 3), dtype=np.uint8)
    for name, checksum in metadata["sha256"].items():
        if not name.endswith(".png"):
            continue
        path = export / name
        if sha256_file(path) != checksum:
            raise ValueError(f"Exported texture changed: {path}")
        row, col = map(int, path.stem.split("_")[1:])
        tile = cv2.imread(str(path), cv2.IMREAD_COLOR)
        r0, c0 = round(row * scale), round(col * scale)
        r1 = round(min(height, row + tile.shape[0]) * scale)
        c1 = round(min(width, col + tile.shape[1]) * scale)
        canvas[r0:r1, c0:c1] = cv2.resize(tile, (c1 - c0, r1 - r0), interpolation=cv2.INTER_AREA)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/m10/demo"))
    parser.add_argument("--case", type=json.loads, help="Test override; writes no docs manifest")
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    case = args.case or protocol["demo_case"]
    stage = "preflight" if case["smoke"] else "live"
    saved = json.loads((args.root / stage / f"{digest(case)}.json").read_text())
    report_path = Path(saved["report_path"])
    if saved["key"] != case or sha256_file(report_path) != saved["sha256"]:
        raise ValueError("Demo case is incomplete or its result changed")
    flight = report_path.parent
    result = json.loads(report_path.read_text())
    cache = load_config("configs/cache/default.yaml", CacheConfig)
    folder = "rl-val" if case["smoke"] else "test"
    world = WorldSpec.load(cache.worlds_dir / folder / f"{case['world_id']}.json")
    table = cache.render.levels
    area_m = world.area_m
    px_per_m = MAP_PX / area_m

    def to_px(x: float, y: float) -> tuple[int, int]:
        return round(x * px_per_m), round((area_m - y) * px_per_m)

    background = top_down(Path.home() / "altiseek-m10/worlds" / world.world_id)
    output = args.out / protocol["protocol_id"] / digest(case)
    output.mkdir(parents=True, exist_ok=True)
    video_path = output / "demo.mp4"
    width, height = MAP_PX + INSET_PX, MAP_PX
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open an MP4 writer")
    path_points: list[tuple[tuple[int, int], int]] = []
    frames = 0
    views, records = result["views"], result["search"]
    for i, view in enumerate(views):
        record = records[i]
        belief_file = flight / f"belief-{i + 1}.npz"
        with np.load(belief_file) as arrays:
            p = arrays["probabilities"]
        # Belief p[ix, iy] -> image rows from the north edge.
        # Three decades below this frame's most likely cell; cell probabilities are small.
        heat = np.log10(np.clip(p.T[::-1], 1e-12, 1.0) / max(float(p.max()), 1e-12))
        heat = np.clip((heat + 3) / 3, 0, 1).astype(np.float32)
        heat = cv2.resize(heat, (MAP_PX, MAP_PX), interpolation=cv2.INTER_NEAREST)
        colour = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        # Opacity follows belief, so likely cells stand out and the sea shows elsewhere.
        alpha = (0.75 * heat)[..., None]
        frame_map = (background * (1 - alpha) + colour * alpha).astype(np.uint8)
        level = view["level"]
        x, y = view["camera_enu_m"][:2]
        path_points.append((to_px(x, y), level))
        for (a, la), (b, _) in pairwise(path_points):
            cv2.line(frame_map, a, b, LEVEL_COLOURS[la], 2, cv2.LINE_AA)
        half = table.footprint_m(table.levels[level]) / 2
        gx, gy = view["goal_enu_m"][:2]
        cv2.rectangle(
            frame_map, to_px(gx - half, gy + half), to_px(gx + half, gy - half), (255, 255, 255), 2
        )
        for target in world.targets:
            cv2.circle(frame_map, to_px(target.x_m, target.y_m), 6, (0, 255, 0), 1, cv2.LINE_AA)
        image = cv2.imread(str(flight / f"detector-{view['stamp_s']:.6f}.png"), cv2.IMREAD_COLOR)
        with np.load(flight / f"detections-{view['stamp_s']:.6f}.npz") as arrays:
            detections = arrays["detections"]
        for d in detections[detections["class_id"] == 0]:
            x0, y0, x1, y1 = (int(v) for v in d["box_px"])
            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 255), 2)
        inset = cv2.resize(image, (INSET_PX, INSET_PX), interpolation=cv2.INTER_AREA)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :MAP_PX] = frame_map
        frame[:INSET_PX, MAP_PX:] = inset
        text = [
            f"{world.split} world {world.world_id}",
            f"{case['method']}, {case['sensor']} sensor",
            f"view {i + 1}/{len(views)}  L{level + 1}",
            f"time {record['time_s']:.0f} s  energy {record['energy_s']:.0f}/500",
            f"found {record['found']}/{len(world.targets)}",
            f"false declarations {record['false_declarations']}",
        ]
        for j, line in enumerate(text):
            cv2.putText(
                frame,
                line,
                (MAP_PX + 12, INSET_PX + 34 + 30 * j),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
        for _ in range(HOLD_FRAMES * (3 if i == len(views) - 1 else 1)):
            writer.write(frame)
            frames += 1
        if i == 0:
            cv2.imwrite(str(output / "first-frame.png"), frame)
    writer.release()
    cv2.imwrite(str(output / "last-frame.png"), frame)
    manifest = {
        "protocol_id": protocol["protocol_id"],
        "case": case,
        "flight": str(flight),
        "video": str(video_path),
        "video_sha256": sha256_file(video_path),
        "frames": frames,
        "fps": FPS,
        "views": len(views),
        "metrics": result["metrics"],
        "source_sha256": sha256_file(Path(__file__)),
        "note": "Local only: contains dataset-derived imagery. Case fixed before outcomes.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if case == protocol["demo_case"]:
        Path("docs/m10-demo.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
