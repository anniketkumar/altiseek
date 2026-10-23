"""prepare-data: build the SeaDronesSee frame and box tables (CLAUDE.md §7).

Run from the repository root:

    uv run python scripts/prepare_data.py [--config configs/data/default.yaml] [key=value ...]
"""

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from altiseek.config import load_config
from altiseek.data.config import DataConfig
from altiseek.data.frames import Box, Frame, load_odv2
from altiseek.data.tables import write_csv
from altiseek.registry.run import write_metadata


def summarise(frames: Sequence[Frame]) -> dict[str, Any]:
    """Per camera group: frame counts per official split, metadata coverage and ranges."""
    groups: defaultdict[str, list[Frame]] = defaultdict(list)
    for frame in frames:
        groups[frame.camera_group].append(frame)
    out: dict[str, Any] = {}
    for name, members in sorted(groups.items()):
        altitudes = [f.altitude_m for f in members if f.altitude_m is not None]
        pitches = [f.gimbal_pitch_deg for f in members if f.gimbal_pitch_deg is not None]
        splits: defaultdict[str, int] = defaultdict(int)
        for f in members:
            splits[f.official_split] += 1
        out[name] = {
            "frames": dict(sorted(splits.items())),
            "with_altitude": len(altitudes),
            "with_pitch": len(pitches),
            "with_timestamp": sum(f.timestamp is not None for f in members),
            "swimmer_boxes": sum(f.n_swimmer for f in members),
            "altitude_m": [min(altitudes), max(altitudes)] if altitudes else None,
            "gimbal_pitch_deg": [min(pitches), max(pitches)] if pitches else None,
        }
    return out


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the SeaDronesSee frame and box tables.")
    parser.add_argument("--config", type=Path, default=Path("configs/data/default.yaml"))
    parser.add_argument("overrides", nargs="*", help="config overrides, dotted.key=value")
    args = parser.parse_args(argv)
    config = load_config(args.config, DataConfig, args.overrides)

    frames, boxes = load_odv2(config.annotations, config.camera_groups)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {"frames": config.out_dir / "frames.csv", "boxes": config.out_dir / "boxes.csv"}
    write_csv(outputs["frames"], Frame, frames)
    write_csv(outputs["boxes"], Box, boxes)
    summary = summarise(frames)
    write_metadata(
        config.out_dir / "prepare_data.meta.json",
        stage="prepare-data",
        config=config,
        inputs={f"annotations_{split}": path for split, path in config.annotations.items()},
        outputs=outputs,
        extra={"summary": summary},
    )
    print(f"{len(frames)} frames, {len(boxes)} boxes -> {config.out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
