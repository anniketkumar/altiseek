"""check-leakage: perceptual-hash near-duplicate search across A/B/T (CLAUDE.md §7 step 7).

The near-duplicate rule is the most sensitive grid point (recall on known duplicates:
consecutive video frames) that flags none of the known non-duplicates: every pair across two
campaigns that cannot share content. The check then searches the pairs where leakage could
hide, across splits within one campaign. Thresholds never depend on that result. Run after
prepare-data, scan-images and make-splits. Exits non-zero if the scan is incomplete or any
cross-split pair is a near-duplicate.

    uv run python scripts/check_leakage.py [--config configs/data/default.yaml] [key=value ...]
"""

import argparse
import io
import json
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from altiseek.config import load_config
from altiseek.data.config import DataConfig
from altiseek.data.frames import Frame
from altiseek.data.image_stats import ImageStats, load_thumbnails
from altiseek.data.images import download, image_path
from altiseek.data.leakage import (
    Pairs,
    Thumbnails,
    choose_thresholds,
    close_pairs,
    consecutive_pairs,
    cross_split_pairs,
    geometric_inliers,
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
    #: RANSAC homography inliers; computed for look-alikes only (0 otherwise).
    inliers: int
    #: Passed the hash and thumbnail tests.
    look_alike: bool
    #: Passed all three tests.
    near_duplicate: bool


def matching_image(config: DataConfig, frame: Frame, width_px: int = 1280) -> NDArray[np.uint8]:
    """A frame at ``width_px`` wide for feature matching, read from disk or the share."""
    path = image_path(config.images, frame)
    data = path.read_bytes() if path.exists() else download(config.images, frame)
    with Image.open(io.BytesIO(data)) as image:
        image.draft("RGB", (width_px, width_px * image.height // image.width))
        rgb = image.convert("RGB")
        height = round(rgb.height * width_px / rgb.width)
        return np.array(rgb.resize((width_px, height), Image.Resampling.BOX))


def plot(path: Path, positives: Pairs, negatives: Pairs, max_h: int, min_r: float) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8))
    for pairs, label in ((negatives, "different campaigns"), (positives, "consecutive frames")):
        left.hist(pairs.hamming, bins=np.arange(0, 66, 2), alpha=0.6, density=True, label=label)
        right.hist(pairs.correlation, bins=np.linspace(-1, 1, 41), alpha=0.6, density=True)
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
    thumb_of = load_thumbnails(scan_dir)
    ids = {f.image_id for f in frames}
    if not ids <= stats.keys() or not ids <= thumb_of.keys() or split_of.keys() != ids:
        sys.exit(f"incomplete inputs: {len(ids - stats.keys())} frames without scan results")
    hash_of = {i: int(s.phash, 16) for i, s in stats.items()}
    thumbs = Thumbnails(thumb_of)

    # Calibration on pairs of known status.
    positives = consecutive_pairs(video_runs(frames, cfg.positive_max_gap_s), hash_of, thumbs)
    session_of, sessions = build_sessions(
        frames, config.sessions.sites, config.sessions.undated_campaign
    )
    first, second = cfg.negative_campaigns
    neg_a = np.array(sorted(i for i, s in session_of.items() if sessions[s].campaign == first))
    neg_b = np.array(sorted(i for i, s in session_of.items() if sessions[s].campaign == second))
    negatives = random_pairs(neg_a, neg_b, cfg.negative_samples, cfg.seed, hash_of, thumbs)
    a, b, d = close_pairs(
        neg_a, hash_array(neg_a, hash_of), neg_b, hash_array(neg_b, hash_of), cfg.max_hamming_limit
    )
    cross_campaign = Pairs(a, b, d, thumbs.correlation(a, b))
    levels = np.round(np.arange(cfg.correlation_min, 1.0 - 1e-9, cfg.correlation_step), 4)
    rule = choose_thresholds(positives, cross_campaign, cfg.max_hamming_limit, levels.tolist())
    max_h, min_r = rule.max_hamming, rule.min_correlation
    print(f"thresholds: distance <= {max_h} bits and correlation >= {min_r:.2f}", flush=True)

    def flagged(pairs: Pairs) -> NDArray[np.bool_]:
        return (pairs.hamming <= max_h) & (pairs.correlation >= min_r)

    # The check itself: pairs across splits within one campaign. Cross-campaign pairs were the
    # known non-duplicates that set the rule, so none of them is flagged by construction.
    campaign_of = {i: sessions[s].campaign for i, s in session_of.items()}
    candidates = cross_split_pairs(split_of, hash_of, thumbs, max_h)
    same = np.array(
        [
            campaign_of[int(x)] == campaign_of[int(y)]
            for x, y in zip(candidates.a, candidates.b, strict=True)
        ],
        dtype=bool,
    )
    candidates = candidates.where(same)
    dup = flagged(candidates)

    # Stage 3: geometric verification of the flagged pairs. Its threshold also comes from pairs
    # of known status: a sample of known duplicates, and look-alikes that are known not to be
    # duplicates: pairs of one session, at least ``negative_min_gap_s`` apart, that pass stages
    # 1-2. They share the site's look like the pairs under test, but sit inside one split.
    rng = np.random.default_rng(cfg.seed)
    n = min(cfg.geometric_samples, len(positives))
    pos_pick = rng.choice(len(positives), size=n, replace=False)
    frame_of = {f.image_id: f for f in frames}
    seconds = {
        f.image_id: f.timestamp.timestamp()
        for f in frames
        if f.timestamp is not None and f.timestamp_source == "image"
    }
    session_ids = sorted(set(session_of.values()))
    alike_parts = []
    for sid in session_ids:
        members = np.array(sorted(i for i, s in session_of.items() if s == sid and i in seconds))
        if len(members) < 2:
            continue
        hashes = hash_array(members, hash_of)
        a_s, b_s, d_s = close_pairs(members, hashes, members, hashes, max_h)
        t_a = np.array([seconds[int(i)] for i in a_s])
        t_b = np.array([seconds[int(i)] for i in b_s])
        apart = (a_s < b_s) & (np.abs(t_a - t_b) >= cfg.negative_min_gap_s)
        alike_parts.append(
            Pairs(a_s[apart], b_s[apart], d_s[apart], thumbs.correlation(a_s[apart], b_s[apart]))
        )
    within = Pairs(
        np.concatenate([p.a for p in alike_parts]),
        np.concatenate([p.b for p in alike_parts]),
        np.concatenate([p.hamming for p in alike_parts]),
        np.concatenate([p.correlation for p in alike_parts]),
    )
    within = within.where(flagged(within))
    hard = np.argsort(-within.correlation)[: cfg.geometric_samples]
    needed = {
        int(i)
        for arr in (
            positives.a[pos_pick],
            positives.b[pos_pick],
            within.a[hard],
            within.b[hard],
            candidates.a[dup],
            candidates.b[dup],
        )
        for i in arr
    }
    print(f"geometric verification: loading {len(needed)} images", flush=True)
    order = sorted(needed)
    with ThreadPoolExecutor(max_workers=16) as pool:
        loaded = pool.map(lambda i: matching_image(config, frame_of[i]), order)
        small = dict(zip(order, loaded, strict=True))

    def inliers(a: NDArray[np.int64], b: NDArray[np.int64]) -> NDArray[np.int64]:
        pairs = list(zip(a.tolist(), b.tolist(), strict=True))
        with ThreadPoolExecutor(max_workers=4) as pool:
            counts = pool.map(lambda ab: geometric_inliers(small[ab[0]], small[ab[1]]), pairs)
            return np.array(list(counts), dtype=np.int64)

    pos_inliers = inliers(positives.a[pos_pick], positives.b[pos_pick])
    neg_inliers = inliers(within.a[hard], within.b[hard])
    min_inliers = int(neg_inliers.max()) + 1
    cand_inliers = np.zeros(len(candidates), dtype=np.int64)
    cand_inliers[dup] = inliers(candidates.a[dup], candidates.b[dup])
    print(
        f"min inliers {min_inliers}; known duplicates passing: "
        f"{float((pos_inliers >= min_inliers).mean()):.2f}",
        flush=True,
    )
    looks_alike = dup.copy()
    dup &= cand_inliers >= min_inliers

    keep = looks_alike.copy()
    keep[np.argsort(-candidates.correlation)[:1000]] = True  # the closest calls, for inspection
    rows = [
        CandidateRow(
            int(x),
            split_of[int(x)],
            int(y),
            split_of[int(y)],
            int(h),
            float(r),
            int(g),
            bool(s),
            bool(f),
        )
        for x, y, h, r, g, s, f in zip(
            candidates.a[keep],
            candidates.b[keep],
            candidates.hamming[keep],
            candidates.correlation[keep],
            cand_inliers[keep],
            looks_alike[keep],
            dup[keep],
            strict=True,
        )
    ]
    near_dups = [r for r in rows if r.near_duplicate]
    in_campaign = {
        c: [i for i, k in campaign_of.items() if k == c] for c in sorted(set(campaign_of.values()))
    }
    checked = 0
    for members in in_campaign.values():
        per_split = [sum(1 for i in members if split_of[i] == k) for k in ("A", "B", "T")]
        checked += per_split[0] * per_split[1] + per_split[0] * per_split[2]
        checked += per_split[1] * per_split[2]

    report = {
        "thresholds": {
            "max_hamming": max_h,
            "min_correlation": min_r,
            "rule": "highest recall on known duplicates with no cross-campaign false alarm",
        },
        "calibration": {
            "positive_pairs": len(positives),
            "positive_recall": rule.recall,
            "positive_hamming_median": float(np.median(positives.hamming)),
            "positive_correlation_median": float(np.median(positives.correlation)),
            "negative_sample_pairs": len(negatives),
            "negative_hamming_median": float(np.median(negatives.hamming)),
            "negative_correlation_median": float(np.median(negatives.correlation)),
            "cross_campaign_pairs": len(neg_a) * len(neg_b),
            "cross_campaign_within_limit": len(cross_campaign),
            "cross_campaign_false_alarms": int(flagged(cross_campaign).sum()),
            "geometric_min_inliers": min_inliers,
            "geometric_look_alike_negatives_found": len(within),
            "geometric_hard_negatives": len(hard),
            "geometric_hard_negative_max_inliers": int(neg_inliers.max()),
            "geometric_positive_sample": n,
            "geometric_positive_recall": float((pos_inliers >= min_inliers).mean()),
            "geometric_positive_median_inliers": float(np.median(pos_inliers)),
        },
        "check": {
            "same_campaign_cross_split_pairs": checked,
            "hash_candidates": len(candidates),
            "look_alikes": int(looks_alike.sum()),
            "look_alike_max_inliers": int(cand_inliers[looks_alike].max())
            if looks_alike.any()
            else 0,
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
