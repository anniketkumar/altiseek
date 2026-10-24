"""render-preview: each level's view around the first target of a few worlds, for checking the
composition by eye (CLAUDE.md §11).

For every chosen world, writes ``<out>/<world_set>-<world_id>.png``: one view per level, finest
first, centred on the lattice point nearest the world's first target. Targets are ringed in
yellow and distractors in cyan, and a thumbnail of the whole composed world (with every object
ringed) is at the end. These are dataset imagery, so they stay under ``data/``, out of git.

    uv run python scripts/render_preview.py [--world-set test] [--count 2] [--out data/previews]
"""

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from altiseek.config import load_config
from altiseek.geometry.levels import Level, LevelTable
from altiseek.geometry.raster import Raster
from altiseek.world.config import RenderConfig, WorldConfig
from altiseek.world.library import Library
from altiseek.world.render import compose_world, render_view
from altiseek.world.spec import WorldSpec

PANEL_PX = 320
YELLOW, CYAN = (255, 230, 0), (0, 220, 255)


def nearest_cell(table: LevelTable, level: Level, x_m: float, y_m: float) -> tuple[int, int]:
    """The lattice cell whose footprint is centred nearest the point."""
    starts = np.arange(0, table.grid_cells - level.footprint_cells + 1, level.stride_cells)

    def best(v_m: float) -> int:
        centres = (starts + level.footprint_cells / 2) * table.cell_m
        return int(starts[np.abs(centres - v_m).argmin()])

    return best(x_m), best(y_m)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render preview views of a few worlds.")
    parser.add_argument("--render", type=Path, default=Path("configs/world/render.yaml"))
    parser.add_argument("--worlds", type=Path, default=Path("configs/world/default.yaml"))
    parser.add_argument("--world-set", default="test")
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--family", default="group-high", help="substring of the family name")
    parser.add_argument("--out", type=Path, default=Path("data/previews"))
    args = parser.parse_args(argv)
    render_cfg = load_config(args.render, RenderConfig)
    world_cfg = load_config(args.worlds, WorldConfig)
    folder = world_cfg.out_dir / args.world_set
    with (folder / "worlds.csv").open(newline="", encoding="utf-8") as fh:
        ids = [r["world_id"] for r in csv.DictReader(fh) if args.family in r["family"]]
    args.out.mkdir(parents=True, exist_ok=True)
    table = render_cfg.levels
    for world_id in ids[: args.count]:
        spec = WorldSpec.load(folder / f"{world_id}.json")
        library = Library.load(world_cfg.libraries[spec.split])
        composed = compose_world(spec, library)
        first = spec.targets[0]
        panels = []
        for level in table.levels:
            cell = nearest_cell(table, level, first.x_m, first.y_m)
            view, truth = render_view(
                spec,
                composed,
                table=table,
                camera=render_cfg.camera,
                level=level,
                cell=cell,
                variant=0,
                settings=render_cfg.variants,
            )
            marked = np.array(view)
            for obj in truth:
                cv2.circle(
                    marked,
                    (round(obj.col_px), round(obj.row_px)),
                    12,
                    YELLOW if obj.target else CYAN,
                    1,
                )
            panels.append(np.asarray(Image.fromarray(marked).resize((PANEL_PX, PANEL_PX))))
        raster = Raster(spec.gsd_mpp, composed.shape[1], composed.shape[0])
        thumb = np.array(
            Image.fromarray(composed).resize((PANEL_PX, PANEL_PX), Image.Resampling.BOX)
        )
        for obj in spec.objects:
            col, row = raster.to_pixel(obj.x_m, obj.y_m)
            scale = PANEL_PX / composed.shape[1]
            centre = (round(float(col) * scale), round(float(row) * scale))
            cv2.circle(thumb, centre, 4, YELLOW if obj.target else CYAN, 1)
        sheet = np.concatenate([*panels, thumb], axis=1)
        path = args.out / f"{args.world_set}-{world_id}.png"
        Image.fromarray(sheet).save(path)
        print(
            f"{path}: family {spec.family}, condition {spec.condition}, {len(spec.targets)} targets"
        )


if __name__ == "__main__":
    main()
