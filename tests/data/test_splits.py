"""Session blocks and the exhaustive, constrained A/B/T assignment."""

from dataclasses import replace
from datetime import date

import pytest

from altiseek.data.config import SplitConfig
from altiseek.data.sessions import Session
from altiseek.data.splits import assign_splits, blocks, split_id, stratum

from ._frames import FRAME

CFG = SplitConfig(
    target_share={"A": 0.6, "B": 0.2, "T": 0.2},
    usable_camera_groups=("mavic_4k",),
    altitude_bands_m=(0.0, 20.0, 40.0, 200.0),
    near_nadir_deg=30.0,
    min_near_nadir_swimmers=1,
    tie_undated_to_campaign=True,
    eval_campaigns=("sea",),
)


def session(sid, campaign, dated=True):
    return Session(sid, campaign, date(2021, 8, 20) if dated else None, 0)


SESSIONS = {
    "lake1": session("lake1", "lake"),
    "lake2": session("lake2", "lake"),
    "undated": session("undated", "lake", dated=False),
    "sea1": session("sea1", "sea"),
    "sea2": session("sea2", "sea"),
    "sea3": session("sea3", "sea"),
}


def frames_for(sizes, nadir_swimmers=1):
    """``sizes[sid]`` usable frames per session, each session with some near-nadir swimmers."""
    frames, of = [], {}
    for sid, n in sizes.items():
        for k in range(n):
            image_id = len(frames)
            nadir = k < nadir_swimmers
            frames.append(
                replace(
                    FRAME,
                    image_id=image_id,
                    altitude_m=10.0 + 30.0 * (k % 3),
                    gimbal_pitch_deg=85.0 if nadir else 40.0,
                    n_swimmer=1 if nadir or k % 2 else 0,
                )
            )
            of[image_id] = sid
    return frames, of


def test_blocks_tie_undated_sessions_to_their_campaign():
    assert blocks(SESSIONS, True) == [["lake1", "lake2", "undated"], ["sea1"], ["sea2"], ["sea3"]]
    assert blocks(SESSIONS, False) == [[s] for s in sorted(SESSIONS)]


def test_held_out_splits_come_from_eval_campaigns_and_shares_are_balanced():
    sizes = {"lake1": 20, "lake2": 20, "undated": 0, "sea1": 20, "sea2": 20, "sea3": 20}
    frames, of = frames_for(sizes)
    result = assign_splits(frames, of, SESSIONS, CFG)
    split = result.split_of_session
    assert split["lake1"] == split["lake2"] == split["undated"] == "A"
    assert sorted(split[s] for s in ("sea1", "sea2", "sea3")) == ["A", "B", "T"]
    assert result.n_assignments == 3**4


def test_min_near_nadir_swimmers_is_enforced():
    sizes = {"lake1": 10, "lake2": 10, "undated": 0, "sea1": 10, "sea2": 10, "sea3": 10}
    frames, of = frames_for(sizes, nadir_swimmers=0)
    with pytest.raises(ValueError, match="no feasible split"):
        assign_splits(frames, of, SESSIONS, CFG)


def test_assignment_is_deterministic():
    sizes = {"lake1": 7, "lake2": 3, "undated": 0, "sea1": 5, "sea2": 5, "sea3": 5}
    frames, of = frames_for(sizes)
    assert assign_splits(frames, of, SESSIONS, CFG) == assign_splits(frames, of, SESSIONS, CFG)


def test_target_shares_must_cover_the_splits_and_sum_to_one():
    bad = CFG.model_copy(update={"target_share": {"A": 0.7, "B": 0.2, "T": 0.2}})
    with pytest.raises(ValueError, match="sum to 1"):
        assign_splits([], {}, SESSIONS, bad)


def test_stratum_bands_and_swimmer_presence():
    assert stratum(replace(FRAME, altitude_m=5.0), CFG.altitude_bands_m) == (0, False)
    assert stratum(replace(FRAME, altitude_m=40.0, n_swimmer=2), CFG.altitude_bands_m) == (2, True)
    with pytest.raises(ValueError, match="outside the bands"):
        stratum(replace(FRAME, altitude_m=500.0), CFG.altitude_bands_m)
    with pytest.raises(ValueError, match="no altitude"):
        stratum(replace(FRAME, altitude_m=None), CFG.altitude_bands_m)


def test_split_id_depends_on_content_not_order():
    assert split_id({1: "A", 2: "T"}) == split_id({2: "T", 1: "A"})
    assert split_id({1: "A", 2: "T"}) != split_id({1: "A", 2: "B"})
    assert len(split_id({1: "A"})) == 16
