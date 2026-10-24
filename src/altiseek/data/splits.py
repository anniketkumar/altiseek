"""Session-level A/B/T split, stratified by altitude band and swimmer presence (CLAUDE.md §7.6).

Sessions are grouped into blocks that must share a split; with ``tie_undated_to_campaign`` an
undated session joins every session of its campaign. With this few sessions every assignment of
blocks to A, B and T can be tried, and the feasible one with the lowest loss wins:

    loss = sum over strata s of  w_s * sum over splits k of (n_sk / n_s - target_k)^2

The strata are altitude band x swimmer presence over the usable frames (camera groups with a
GSD), n_sk counts stratum s's usable frames in split k, and w_s = n_s / n is the stratum's share.
An assignment is feasible if every split has usable frames, B and T draw only on sessions of the
``eval_campaigns`` (sensor models are fitted on B worlds and judged on T worlds, so both must
come from the target domain), and B and T each have at least ``min_near_nadir_swimmers``
swimmer boxes on near-nadir usable frames (M4 libraries and the fidelity check need them). Ties
go to the first assignment in ``itertools.product`` order.
"""

import hashlib
import itertools
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from altiseek.data.config import SplitConfig
from altiseek.data.frames import Frame
from altiseek.data.sessions import Session

SPLITS = ("A", "B", "T")


@dataclass(frozen=True)
class SplitRow:
    """One labelled image's session and split: a row of ``splits.csv``."""

    image_id: int
    session_id: str
    split: str


@dataclass(frozen=True)
class SplitResult:
    split_of_session: dict[str, str]
    loss: float
    #: Loss of the next-best feasible assignment (None if there is no other).
    runner_up_loss: float | None
    n_assignments: int
    n_feasible: int


def blocks(sessions: Mapping[str, Session], tie_undated_to_campaign: bool) -> list[list[str]]:
    """Sessions that must share a split, as sorted lists, in sorted order."""
    if not tie_undated_to_campaign:
        return [[sid] for sid in sorted(sessions)]
    tied = {s.campaign for s in sessions.values() if s.day is None}
    by_campaign: defaultdict[str, list[str]] = defaultdict(list)
    out = []
    for sid in sorted(sessions):
        campaign = sessions[sid].campaign
        if campaign in tied:
            by_campaign[campaign].append(sid)
        else:
            out.append([sid])
    out.extend(by_campaign[c] for c in sorted(by_campaign))
    return sorted(out)


def stratum(frame: Frame, band_edges_m: Sequence[float]) -> tuple[int, bool]:
    """(altitude band index, swimmer present) of a usable frame."""
    if frame.altitude_m is None:
        raise ValueError(f"usable image {frame.image_id} has no altitude")
    band = int(np.searchsorted(band_edges_m, frame.altitude_m, side="right")) - 1
    if not 0 <= band < len(band_edges_m) - 1:
        raise ValueError(
            f"altitude {frame.altitude_m} m of image {frame.image_id} is outside the bands"
        )
    return band, frame.n_swimmer > 0


def near_nadir(frame: Frame, limit_deg: float) -> bool:
    off_nadir = frame.off_nadir_deg
    return off_nadir is not None and abs(off_nadir) <= limit_deg


def assign_splits(
    frames: Sequence[Frame],
    session_of: Mapping[int, str],
    sessions: Mapping[str, Session],
    cfg: SplitConfig,
) -> SplitResult:
    """Exhaustive search over block-to-split assignments (see the module docstring)."""
    if abs(sum(cfg.target_share.values()) - 1.0) > 1e-9 or set(cfg.target_share) != set(SPLITS):
        raise ValueError(f"target shares must cover {SPLITS} and sum to 1: {cfg.target_share}")
    groups = blocks(sessions, cfg.tie_undated_to_campaign)
    block_of = {sid: b for b, members in enumerate(groups) for sid in members}
    n_bands = len(cfg.altitude_bands_m) - 1
    counts = np.zeros((len(groups), n_bands * 2))
    nadir_swimmers = np.zeros(len(groups))
    for frame in frames:
        if frame.camera_group not in cfg.usable_camera_groups:
            continue
        b = block_of[session_of[frame.image_id]]
        band, swimmer = stratum(frame, cfg.altitude_bands_m)
        counts[b, band * 2 + swimmer] += 1
        if near_nadir(frame, cfg.near_nadir_deg):
            nadir_swimmers[b] += frame.n_swimmer
    weight = counts.sum(axis=0) / counts.sum()
    target = np.array([cfg.target_share[k] for k in SPLITS])
    present = counts.sum(axis=0) > 0
    in_eval = [all(sessions[sid].campaign in cfg.eval_campaigns for sid in g) for g in groups]
    held_out = [SPLITS.index("B"), SPLITS.index("T")]

    losses: list[tuple[float, tuple[int, ...]]] = []
    n_feasible = 0
    for choice in itertools.product(range(len(SPLITS)), repeat=len(groups)):
        pick = np.zeros((len(SPLITS), len(groups)))
        pick[list(choice), range(len(groups))] = 1.0
        per_split = pick @ counts  # splits x strata
        swimmers = pick @ nadir_swimmers
        if (per_split.sum(axis=1) == 0).any():
            continue
        if any(choice[b] in held_out and not in_eval[b] for b in range(len(groups))):
            continue
        if (
            min(swimmers[SPLITS.index("B")], swimmers[SPLITS.index("T")])
            < cfg.min_near_nadir_swimmers
        ):
            continue
        n_feasible += 1
        share = per_split[:, present] / per_split[:, present].sum(axis=0)
        losses.append(
            (float((weight[present] * ((share - target[:, None]) ** 2).sum(axis=0)).sum()), choice)
        )
    if not losses:
        raise ValueError("no feasible split: relax min_near_nadir_swimmers or the session rules")
    ranked = sorted(losses, key=lambda item: item[0])  # stable: ties keep product order
    loss, choice = ranked[0]
    split_of_session = {
        sid: SPLITS[choice[b]] for b, members in enumerate(groups) for sid in members
    }
    runner_up = ranked[1][0] if len(ranked) > 1 else None
    return SplitResult(split_of_session, loss, runner_up, len(SPLITS) ** len(groups), n_feasible)


def split_id(split_of_image: Mapping[int, str]) -> str:
    """Content ID of a split: SHA-256 of the sorted ``image_id,split`` lines, 16 hex digits."""
    lines = "".join(f"{i},{split_of_image[i]}\n" for i in sorted(split_of_image))
    return hashlib.sha256(lines.encode()).hexdigest()[:16]
