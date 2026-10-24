"""scan-images: one streaming pass over every labelled image (CLAUDE.md §7 steps 1 and 7).

Each image yields a perceptual hash and a thumbnail (for the leakage check) and blackened-region
statistics (ignore regions). Images on disk are read from there; the rest are downloaded and
discarded after analysis unless the keep rule matches, so the full dataset never has to be
stored. Resumable: rerun the same command to continue an interrupted scan.

    uv run python scripts/scan_images.py [--config configs/data/default.yaml] [key=value ...]
"""

import argparse
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from altiseek.config import load_config
from altiseek.data.config import DataConfig, KeepRule
from altiseek.data.frames import Frame
from altiseek.data.image_stats import ImageStats, run_scan
from altiseek.data.images import download, image_path, save_atomic
from altiseek.data.tables import read_csv
from altiseek.registry.run import write_metadata


def keeps(rule: KeepRule, frame: Frame) -> bool:
    """Whether a scanned image is kept on disk for later stages."""
    off_nadir = frame.off_nadir_deg
    return (
        frame.camera_group in rule.camera_groups
        and off_nadir is not None
        and abs(off_nadir) <= rule.max_off_nadir_deg
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Scan every labelled image once.")
    parser.add_argument("--config", type=Path, default=Path("configs/data/default.yaml"))
    parser.add_argument("overrides", nargs="*", help="config overrides, dotted.key=value")
    args = parser.parse_args(argv)
    config = load_config(args.config, DataConfig, args.overrides)

    frames_csv = config.out_dir / "frames.csv"
    frames = {f.image_id: f for f in read_csv(frames_csv, Frame)}
    out_dir = config.out_dir / "image_scan"

    def read(image_id: int) -> bytes:
        path = image_path(config.images, frames[image_id])
        return path.read_bytes() if path.exists() else download(config.images, frames[image_id])

    def keep_copy(image_id: int, data: bytes) -> None:
        frame = frames[image_id]
        path = image_path(config.images, frame)
        if keeps(config.scan.keep, frame) and not path.exists():
            save_atomic(path, data)

    start = time.time()

    def progress(done: int, total: int) -> None:
        rate = done / max(time.time() - start, 1e-9)
        eta_min = (total - done) / rate / 60 if rate > 0 else float("inf")
        print(f"{done}/{total} images, {rate:.2f}/s, ~{eta_min:.0f} min left", flush=True)

    analysed = run_scan(
        sorted(frames),
        read,
        out_dir,
        workers=config.scan.workers,
        chunk_size=config.scan.chunk_size,
        after_read=keep_copy,
        progress=progress,
    )

    stats = read_csv(out_dir / "stats.csv", ImageStats)
    missing = sorted(set(frames) - {s.image_id for s in stats})
    size_mismatch = [
        s.image_id
        for s in stats
        if (s.width_px, s.height_px) != (frames[s.image_id].width_px, frames[s.image_id].height_px)
    ]
    summary = {
        "analysed_this_run": analysed,
        "scanned": len(stats),
        "missing": len(missing),
        "size_mismatch": size_mismatch,
        "kept_on_disk": sum(keeps(config.scan.keep, f) for f in frames.values()),
        "black_blocks_by_group": {
            group: dict(
                Counter(
                    "0" if s.black_blocks == 0 else "1-9" if s.black_blocks < 10 else "10+"
                    for s in stats
                    if frames[s.image_id].camera_group == group
                )
            )
            for group in sorted({f.camera_group for f in frames.values()})
        },
    }
    write_metadata(
        out_dir / "scan_images.meta.json",
        stage="scan-images",
        config=config,
        inputs={"frames": frames_csv},
        outputs={"stats": out_dir / "stats.csv"},
        extra={"summary": summary},
    )
    print(summary)
    if missing or size_mismatch:
        sys.exit(f"incomplete scan: {len(missing)} missing, {len(size_mismatch)} size mismatches")


if __name__ == "__main__":
    main()
