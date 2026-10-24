"""make-splits: capture-day sessions and the session-level A/B/T split (CLAUDE.md §7 steps 5-6).

Run from the repository root after prepare-data:

    uv run python scripts/make_splits.py [--config configs/data/default.yaml] [key=value ...]
"""

import argparse
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from altiseek.config import load_config
from altiseek.data.config import DataConfig, SplitConfig
from altiseek.data.frames import CATEGORIES, Frame
from altiseek.data.sessions import build_sessions
from altiseek.data.splits import (
    SPLITS,
    SplitRow,
    assign_splits,
    near_nadir,
    split_id,
    stratum,
)
from altiseek.data.tables import read_csv, write_csv
from altiseek.registry.run import write_metadata

CLASSES = [c for c in CATEGORIES if c != "ignored"]


@dataclass(frozen=True)
class SessionRow:
    session_id: str
    campaign: str
    day: date | None
    n_frames: int
    split: str


def band_label(edges: Sequence[float], band: int) -> str:
    return f"{edges[band]:g}-{edges[band + 1]:g} m"


def tables(frames: Sequence[Frame], split_of: dict[int, str], cfg: SplitConfig) -> str:
    """Markdown: usable frames and boxes per split x altitude band x class, plus group totals."""
    edges = cfg.altitude_bands_m
    usable = [f for f in frames if f.camera_group in cfg.usable_camera_groups]
    header = "| Split | Altitude | Frames | With swimmers | " + " | ".join(CLASSES) + " |"
    lines = [
        f"Usable frames ({', '.join(cfg.usable_camera_groups)}) and their boxes:",
        "",
        header,
        "|" + "---|" * (4 + len(CLASSES)),
    ]
    for split in SPLITS:
        rows = [f for f in usable if split_of[f.image_id] == split]
        for band in range(len(edges) - 1):
            sel = [f for f in rows if stratum(f, edges)[0] == band]
            boxes = [sum(getattr(f, f"n_{c}") for f in sel) for c in CLASSES]
            cells = [split, band_label(edges, band), len(sel), sum(f.n_swimmer > 0 for f in sel)]
            lines.append("| " + " | ".join(map(str, cells + boxes)) + " |")
        total = [sum(getattr(f, f"n_{c}") for f in rows) for c in CLASSES]
        cells = [f"**{split}**", "all", len(rows), sum(f.n_swimmer > 0 for f in rows)]
        lines.append("| " + " | ".join(map(str, cells + total)) + " |")
    groups = sorted({f.camera_group for f in frames})
    lines += [
        "",
        f"Near-nadir (off-nadir <= {cfg.near_nadir_deg:g} deg) usable swimmer boxes and all frames"
        " by camera group:",
        "",
        "| Split | Near-nadir swimmer boxes | " + " | ".join(groups) + " |",
        "|" + "---|" * (2 + len(groups)),
    ]
    for split in SPLITS:
        nadir = sum(
            f.n_swimmer
            for f in usable
            if split_of[f.image_id] == split and near_nadir(f, cfg.near_nadir_deg)
        )
        per_group = Counter(f.camera_group for f in frames if split_of[f.image_id] == split)
        lines.append(
            f"| {split} | {nadir} | " + " | ".join(str(per_group[g]) for g in groups) + " |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build sessions and the A/B/T split.")
    parser.add_argument("--config", type=Path, default=Path("configs/data/default.yaml"))
    parser.add_argument("overrides", nargs="*", help="config overrides, dotted.key=value")
    args = parser.parse_args(argv)
    config = load_config(args.config, DataConfig, args.overrides)

    frames_csv = config.out_dir / "frames.csv"
    frames = read_csv(frames_csv, Frame)
    session_of, sessions = build_sessions(
        frames, config.sessions.sites, config.sessions.undated_campaign
    )
    result = assign_splits(frames, session_of, sessions, config.splits)
    split_of = {i: result.split_of_session[s] for i, s in session_of.items()}

    outputs = {
        "splits": config.out_dir / "splits.csv",
        "sessions": config.out_dir / "sessions.csv",
        "split_tables": config.out_dir / "split_tables.md",
    }
    write_csv(
        outputs["splits"],
        SplitRow,
        [SplitRow(i, session_of[i], split_of[i]) for i in sorted(session_of)],
    )
    write_csv(
        outputs["sessions"],
        SessionRow,
        [
            SessionRow(s.session_id, s.campaign, s.day, s.n_frames, result.split_of_session[sid])
            for sid, s in sorted(sessions.items())
        ],
    )
    markdown = tables(frames, split_of, config.splits)
    outputs["split_tables"].write_text(markdown, encoding="utf-8")
    sid = split_id(split_of)
    write_metadata(
        config.out_dir / "make_splits.meta.json",
        stage="make-splits",
        config=config,
        inputs={"frames": frames_csv},
        outputs=outputs,
        extra={
            "split_id": sid,
            "split_of_session": result.split_of_session,
            "loss": result.loss,
            "runner_up_loss": result.runner_up_loss,
            "assignments_tried": result.n_assignments,
            "assignments_feasible": result.n_feasible,
        },
    )
    print(f"split_id {sid}; loss {result.loss:.4f}, runner-up {result.runner_up_loss}")
    print(f"feasible assignments {result.n_feasible}/{result.n_assignments}")
    for s in sessions.values():
        split = result.split_of_session[s.session_id]
        print(f"  {s.session_id:30s} {s.campaign:22s} {s.n_frames:5d} -> {split}")
    print(markdown)


if __name__ == "__main__":
    main()
