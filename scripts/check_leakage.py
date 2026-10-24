"""check-leakage: perceptual-hash near-duplicate search across A/B/T (CLAUDE.md §7 step 7).

Thresholds are derived from known duplicates (consecutive video frames), never from the
cross-split result, and their false-alarm rate is measured on a full search across two campaigns
that cannot share content. Run after prepare-data, scan-images and make-splits; exits non-zero if
the scan is incomplete or any cross-split pair is a near-duplicate.

    uv run python scripts/check_leakage.py [--config configs/data/default.yaml] [key=value ...]
"""

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from altiseek.config import load_config
from altiseek.data.config import DataConfig
from altiseek.data.frames import Frame
from altiseek.data.image_stats import ImageStats, load_thumbnails
from altiseek.data.leakage import (
    Pair,
    close_pairs,
    consecutive_pairs,
    correlation,
    cross_split_pairs,
    hash_array,
    random_pairs,
    video_runs,
)
from altiseek.data.sessions import build_sessions
from altiseek.data.splits import SplitRow
from altiseek.data.tables import read_csv, write_csv
from altiseek.registry.run import write_metadata


@dataclass(frozen=True)
class CandidateRow:
    """A cross-split pair whose hashes are within the distance threshold."""

    image_a: int
    split_a: str
    image_b: int
    split_b: str
    hamming: int
    correlation: float
    near_duplicate: bool


def plot(
    path: Path, positives: Sequence[Pair], negatives: Sequence[Pair], max_h: int, min_r: float
) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8))
    for pairs, label in ((negatives, "different campaigns"), (positives, "consecutive frames")):
        left.hist(
            [p.hamming for p in pairs],
            bins=np.arange(0, 66, 2),
            alpha=0.6,
            density=True,
            label=label,
        )
        right.hist(
            [p.correlation for p in pairs], bins=np.linspace(-1, 1, 41), alpha=0.6, density=True
        )
    left.axvline(max_h + 0.5, color="black", lw=1)
    left.set_xlabel("perceptual-hash distance (bits)")
    left.set_ylabel("density")
    left.legend(fontsize=8)
    right.axvline(min_r, color="black", lw=1)
    right.set_xlabel("thumbnail correlation")
    fig.suptitle(
        f"Leakage-check calibration: near-duplicate if distance <= {max_h} and r >= {min_r:.3f}",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Near-duplicate search across A/B/T.")
    parser.add_argument("--config", type=Path, default=Path("configs/data/default.yaml"))
    parser.add_argument("overrides", nargs="*", help="config overrides, dotted.key=value")
    args = parser.parse_args(argv)
    config = load_config(args.config, DataConfig, args.overrides)
    cfg = config.leakage

    frames_csv = config.out_dir / "frames.csv"
    splits_csv = config.out_dir / "splits.csv"
    scan_dir = config.out_dir / "image_scan"
    frames = read_csv(frames_csv, Frame)
    split_of = {r.image_id: r.split for r in read_csv(splits_csv, SplitRow)}
    stats = {s.image_id: s for s in read_csv(scan_dir / "stats.csv", ImageStats)}
    thumbs = load_thumbnails(scan_dir)
    ids = {f.image_id for f in frames}
    if not ids <= stats.keys() or not ids <= thumbs.keys() or split_of.keys() != ids:
        sys.exit(f"incomplete inputs: {len(ids - stats.keys())} frames without scan results")
    hash_of = {i: int(s.phash, 16) for i, s in stats.items()}

    # Calibration on pairs of known status.
    positives = consecutive_pairs(video_runs(frames, cfg.positive_max_gap_s), hash_of, thumbs)
    session_of, sessions = build_sessions(
        frames, config.sessions.sites, config.sessions.undated_campaign
    )
    first, second = cfg.negative_campaigns
    neg_a = sorted(i for i, s in session_of.items() if sessions[s].campaign == first)
    neg_b = sorted(i for i, s in session_of.items() if sessions[s].campaign == second)
    negatives = random_pairs(neg_a, neg_b, cfg.negative_samples, cfg.seed, hash_of, thumbs)
    q = cfg.calibration_quantile
    max_h = math.ceil(float(np.quantile([p.hamming for p in positives], q)))
    min_r = float(np.quantile([p.correlation for p in positives], 1.0 - q))

    def near_duplicate(distance: int, r: float) -> bool:
        return distance <= max_h and r >= min_r

    cross_campaign = close_pairs(
        np.array(neg_a),
        hash_array(neg_a, hash_of),
        np.array(neg_b),
        hash_array(neg_b, hash_of),
        max_h,
    )
    false_alarms = sum(
        near_duplicate(d, correlation(thumbs[a], thumbs[b])) for a, b, d in cross_campaign
    )

    # The check itself.
    rows = [
        CandidateRow(
            p.image_a,
            split_of[p.image_a],
            p.image_b,
            split_of[p.image_b],
            p.hamming,
            p.correlation,
            near_duplicate(p.hamming, p.correlation),
        )
        for p in cross_split_pairs(split_of, hash_of, thumbs, max_h)
    ]
    near_dups = [r for r in rows if r.near_duplicate]
    sizes = [sum(1 for s in split_of.values() if s == k) for k in ("A", "B", "T")]

    report = {
        "thresholds": {"max_hamming": max_h, "min_correlation": min_r, "quantile": q},
        "calibration": {
            "positive_pairs": len(positives),
            "positive_flagged": sum(near_duplicate(p.hamming, p.correlation) for p in positives),
            "positive_hamming_median": float(np.median([p.hamming for p in positives])),
            "negative_sample_pairs": len(negatives),
            "negative_sample_flagged": sum(
                near_duplicate(p.hamming, p.correlation) for p in negatives
            ),
            "negative_hamming_median": float(np.median([p.hamming for p in negatives])),
            "cross_campaign_pairs": len(neg_a) * len(neg_b),
            "cross_campaign_hash_candidates": len(cross_campaign),
            "cross_campaign_false_alarms": false_alarms,
        },
        "check": {
            "cross_split_pairs": sizes[0] * sizes[1] + sizes[0] * sizes[2] + sizes[1] * sizes[2],
            "hash_candidates": len(rows),
            "near_duplicates": len(near_dups),
            "near_duplicate_pairs": [asdict(r) for r in near_dups[:50]],
            "passed": not near_dups,
        },
    }
    outputs = {
        "candidates": config.out_dir / "leakage_candidates.csv",
        "report": config.out_dir / "leakage_report.json",
    }
    write_csv(outputs["candidates"], CandidateRow, rows)
    outputs["report"].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    plot(config.figures_dir / "leakage_calibration.png", positives, negatives, max_h, min_r)
    write_metadata(
        config.out_dir / "check_leakage.meta.json",
        stage="check-leakage",
        config=config,
        inputs={"frames": frames_csv, "splits": splits_csv, "scan_stats": scan_dir / "stats.csv"},
        outputs=outputs,
        extra={"passed": not near_dups},
    )
    print(json.dumps(report, indent=2))
    if near_dups:
        sys.exit(f"leakage check FAILED: {len(near_dups)} cross-split near-duplicates")
    print("leakage check passed")


if __name__ == "__main__":
    main()
